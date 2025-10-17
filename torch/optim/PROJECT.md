# Muon Distributed Training

Modifying the Muon optimizer to work with distributed training setups.

**Related Documents**:
- `DESIGN.md` - Detailed API specification and implementation details
- `EXAMPLES.md` - Concrete usage patterns for DTensor and process group configurations
- `TESTING.md` - Testing strategy and validation approach

## Problem

### Background

The Muon optimizer performs this update step:
```python
muon_step(params, grads, momentum_buffers, lr, beta, ns_steps):
    for param, grad, momentum_buffer in zip(params, grads, momentum_buffers):
        momentum_buffer = beta * momentum_buffer + grad
        update = newton_schulz_orthogonalize(momentum_buffer, ns_steps)
        param = param - lr * update
```

### Challenge 1: Correctness

Muon's orthogonalization step (`_zeropower_via_newtonschulz` in `_muon.py`) requires operating on the **full tensor** to maintain mathematical correctness. However, distributed training (FSDP, HSDP, Tensor Parallel, Expert Parallel, Context Parallel, Pipeline Parallel, etc.) shards parameters and their updates across devices. Orthogonalizing partial tensors separately produces **incorrect results**.

**Required:** In the distributed setting, the full `momentum_buffer` must be gathered from shards, orthogonalized as a full tensor, then the resulting update must be redistributed back to shards.

### Challenge 2: Efficiency

Without optimization, `newton_schulz_orthogonalize` would operate on the full `momentum_buffer` **on every GPU**, creating redundant computation.

**Required:** Each parameter should be assigned to exactly one rank. That rank gathers the full tensor, performs orthogonalization once, then redistributes the result.

## Solution Overview

Add a `distributed_config: DistributedConfig` parameter to `Muon.__init__()` that enables distributed orthogonalization with **zero redundancy** - each parameter is processed exactly once across all devices.

The configuration supports two approaches:
1. **DTensor-based** (recommended for new code): Automatic configuration using PyTorch's native DTensor API
2. **Process group-based** (for advanced setups): Manual configuration with three user-defined functions

## Quick Start with DTensor (Recommended)

For distributed setups using `torch.distributed.tensor.DTensor`, use the simpler `create_dtensor_config()` function:

```python
from torch.optim.muon import create_dtensor_config

# DTensors already encapsulate distributed metadata (device mesh, placement, etc.)
config = create_dtensor_config()

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

**Why DTensor?**
- **Simpler API**: No need to specify process groups - DTensor parameters already contain distributed metadata
- **Automatic detection**: DTensor placement and device mesh are extracted from parameters automatically
- **Works with any placement**: Automatically handles `Shard()`, `Replicate()`, `Partial()`, and hybrid placements
- **Future-proof**: PyTorch is moving toward DTensor as the standard distributed tensor API

## Verifying Your Setup

Before running long training jobs, verify your configuration is correct:

```python
# Print assignments to verify each parameter assigned exactly once
if hasattr(optimizer, 'assignments'):
    rank = optimizer.current_rank
    assigned_params = optimizer.assignments.get(rank, [])
    print(f"Rank {rank}: Processing params {assigned_params}")

# Run a few steps and check for errors
for i in range(3):
    loss = model(input_data).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {i} completed successfully")
```

## When to Use Distributed Config

- **Always use** when training with FSDP, TP, or any sharding strategy
  - Required for correctness (orthogonalization needs full tensors)
- **Recommended** for DDP/CP to eliminate redundant computation
  - Without it: every GPU redundantly orthogonalizes all parameters
  - With it: each parameter orthogonalized exactly once (zero redundancy)
- **Skip** for single-device training
  - Use `distributed_config=None` (default behavior)

## Advanced: Process Group Configuration

For advanced scenarios or non-DTensor distributed frameworks, use `create_auto_config()` with explicit process groups:

```python
from torch.optim.muon import create_auto_config

# Specify process groups explicitly
config = create_auto_config(
    fsdp_process_group=fsdp_pg,
    tp_process_group=tp_pg,
    tp_dim_per_param={0: 0, 1: 1, 2: 0, 3: 1},  # Per-parameter TP dimensions
)

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

**Use process group configuration when:**
- Your model uses process groups directly (FSDP, DDP, custom setups)
- You need fine-grained control over gather/redistribute operations
- You're working with custom or proprietary distributed frameworks
- You're using Expert Parallel or other specialized patterns

The process group approach uses **three user-defined functions**:
1. **assign_fn**: Maps each rank to the list of parameter indices it will process
2. **gather_fn**: Gathers the full momentum buffer from shards (on assigned rank)
3. **redistribute_fn**: Redistributes the orthogonalized update back to shards

`create_auto_config()` automatically creates these functions for common PyTorch parallelism strategies (FSDP, TP, DDP, EP, CP).

## Key Design Principles

**Flexibility**: Both approaches support **any distributed setup** - PyTorch built-in strategies, custom frameworks, or proprietary parallelism.

**Zero Redundancy**: Each parameter is assigned to exactly one rank. Each parameter is processed exactly once with no redundant computation.

**Backward Compatibility**: When `distributed_config=None`, Muon uses the original single-device implementation.

**Special Cases**:
- **Independent Parameter Groups** (PP, EP): Some parameters should be orthogonalized separately, not gathered together. For example, in Expert Parallel (EP), each expert is independent.
- **Replicated Strategies** (CP, DDP, HSDP-DDP): Parameters are fully replicated across some dimensions. The `gather_fn` is a no-op along these dimensions since tensors are already complete. The `redistribute_fn` broadcasts the update to maintain replica consistency.

## Troubleshooting

### Common Issues

#### "Parameters sharded incorrectly" or Shape Mismatches
**Symptom**: Runtime errors about tensor shape mismatches during gather/redistribute.

**Causes & Solutions**:
- **DTensor**: Ensure all DTensor parameters use compatible device meshes. Check: `all(p.data.device_mesh == params[0].data.device_mesh for p in params if isinstance(p.data, DTensor))`
- **Process Groups**: Verify `tp_dim_per_param` matches actual model sharding. Use the helper in `EXAMPLES.md` to auto-detect TP dimensions.
- **FSDP**: Ensure FSDP process group is correctly configured and matches model wrapping.

#### Training Diverges or Produces NaN
**Symptom**: Loss becomes NaN or training diverges from single-device baseline.

**Causes & Solutions**:
- Verify distributed config is correct for your parallelism strategy.
- Compare with single-device training using same seed (see `TESTING.md` for comparison tests).
- Check that all ranks participate in collectives (no deadlocks).
- Ensure learning rate is appropriate for distributed training.

#### Slower Than Expected Performance
**Symptom**: Training is slower than single-device or shows poor scaling.

**Causes & Solutions**:
- **Enable optimizations**: Set `async_gpu_parallelism=True` and `prefetch_count=1-2`.
- **Check redundant computation**: Without distributed config, every GPU orthogonalizes all parameters (expected behavior for DDP/CP without config).
- **Profile communication**: Use PyTorch profiler to identify communication bottlenecks.
- **Balance assignment**: Ensure parameters are evenly distributed across ranks.

**Expected Performance**:
- Prefetch (count=1-2): 1.5-2x speedup for communication-bound workloads
- Async GPU parallelism: 1.2-1.5x speedup for large models
- Without distributed config (DDP/CP): All GPUs redundantly compute (expected behavior)

#### "Assignment coverage error" During Initialization
**Symptom**: `AssertionError` about incomplete or overlapping parameter assignments.

**Causes & Solutions**:
- **Using custom assign_fn**: Ensure all parameters are assigned to exactly one rank.
- **Process group mismatch**: Verify process groups match expected world size.
- **Check validation**: Review assignment validation in [`DESIGN.md` section "Validation at Initialization"](#validation-at-initialization) (`DESIGN.md#validation-at-initialization`).

### Verifying Correctness

To verify your distributed configuration is working correctly:

1. **Compare with single-device**: Run identical training (same seed, data) on single device and distributed. Final parameters should match within tolerance (see [Distributed Integration Tests in TESTING.md](TESTING.md#4-distributed-integration-tests-process-groups) for `test_fsdp_vs_single_device` example).

2. **Check assignment**: Print assignments during initialization to verify each parameter is assigned exactly once:
   ```python
   if distributed_config:
       print(f"Assignments: {optimizer.assignments}")
   ```

3. **Monitor memory**: Ensure gather/redistribute aren't causing OOM. Reduce `prefetch_count` if needed.

4. **Use debug mode**: Start with `async_gpu_parallelism=False` and `prefetch_count=0` to isolate issues.

### Getting Help

- **Error messages**: Check [Error Handling in DESIGN.md](DESIGN.md#error-handling) for detailed error handling documentation.
- **Test examples**: See `TESTING.md` for comprehensive test examples covering all parallelism strategies.
- **Comparison table**: See [Choosing Between DTensor and Process Group Approaches in EXAMPLES.md](EXAMPLES.md#choosing-between-dtensor-and-process-group-approaches) for a detailed comparison table.

## Documentation Guide

- **DESIGN.md**: Detailed API specification and implementation details
- **EXAMPLES.md**: Concrete usage patterns for both DTensor and process group configurations
- **TESTING.md**: Testing strategy and validation approach

For a detailed comparison of DTensor vs process group approaches, see the comparison table in `EXAMPLES.md`.
