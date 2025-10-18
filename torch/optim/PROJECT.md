# Muon Distributed Training

Design document for adding distributed training support to the Muon optimizer.

## Summary

This document describes adding distributed training support to the Muon optimizer through a `distributed_config` parameter. The key design principle is **zero-redundancy orthogonalization**: each parameter is assigned to exactly one rank, which gathers the full tensor from shards, performs orthogonalization once, then redistributes the result back to all devices.

**Core API:** Users provide three functions via `DistributedConfig`:
- `assign_fn`: Maps each parameter to a rank responsible for orthogonalizing it
- `gather_fn`: Gathers full momentum buffer from shards/replicas to the assigned rank
- `redistribute_fn`: Distributes the orthogonalized update back to shards/replicas

**Quality of Life:** Helper functions (`create_processgroup_config`, `create_devicemesh_config`, `create_dtensor_config`) automatically generate these functions for common distributed setups.

**Performance:** Optional prefetching and async GPU parallelism enable overlapping communication with computation for improved training throughput.

## Terminology

- **Orthogonalization**: Mathematical transformation that converts a matrix to have orthonormal rows/columns (unit length, mutually perpendicular). Muon uses this to normalize gradient updates.
- **Newton-Schulz iteration**: Iterative algorithm for computing matrix orthogonalization. Requires operating on the complete tensor for mathematical correctness.
- **Sharding**: Partitioning a tensor across multiple devices, where each device holds a distinct subset (shard) of the data.
- **Replication**: Duplicating the full tensor across multiple devices, where each device holds an identical complete copy.
- **Process Group**: PyTorch's abstraction for a subset of distributed processes that communicate together.
- **Device Mesh**: PyTorch's multi-dimensional grid topology for organizing devices across different parallelism dimensions.

## Problem

### Background

The Muon optimizer performs this update step:
```python
muon_step(params, grads, momentum_buffers, lr, beta, ns_steps):
    for param, grad, momentum_buffer in zip(params, grads, momentum_buffers):
        momentum_buffer = beta * momentum_buffer + grad
        update = _zeropower_via_newtonschulz(momentum_buffer, ns_steps)
        param -= lr * update
```

**Note:** Muon's orthogonalization only applies to parameters with 2 or more dimensions (i.e., `len(param.shape) >= 2`). Scalars and 1D tensors (e.g., biases, layer norm parameters) are not handled by the Muon optimizer's orthogonalization step.

### Challenge 1: Correctness

Muon's orthogonalization step (via `_zeropower_via_newtonschulz` in `_muon.py`) requires operating on the **full tensor** to maintain mathematical correctness. However, distributed training (FSDP, HSDP, Tensor Parallel, Expert Parallel, Context Parallel, Pipeline Parallel, etc.) shards parameters and their updates across devices. Orthogonalizing partial tensors separately produces **incorrect results**.

**Required:** In the distributed setting, the full `momentum_buffer` must be gathered from shards, orthogonalized as a full tensor, then the resulting update must be redistributed back to shards.

### Challenge 2: Efficiency

In a naive implementation, `_zeropower_via_newtonschulz` would operate on the full `momentum_buffer` **on every GPU**, creating redundant computation.

**Required:** Each parameter should be assigned to exactly one rank. That rank gathers the full tensor, performs orthogonalization once, then redistributes the result.

## Solution Overview

Add a `distributed_config: DistributedConfig` parameter to `Muon.__init__()` that enables distributed orthogonalization with **zero redundancy** - each parameter is processed exactly once across all devices.

**Note:** When `distributed_config=None` (the default), Muon operates in non-distributed mode using the standard single-device optimizer behavior.

`DistributedConfig` has **three user-defined functions**:
1. **assign_fn**: Maps each param_idx to a rank that will orthogonalize the momentum buffer for that parameter
2. **gather_fn**: Gathers the full momentum buffer from shards (on assigned rank)
3. **redistribute_fn**: Redistributes (scatters) the orthogonalized update back to shards and/or replicas

Additionally, `DistributedConfig` includes a `state` dictionary that holds all metadata needed by the three functions above including but not limited to: Process groups (e.g., FSDP, TP, DP process groups), Device meshes, Tensor shapes/dtypes, Communication patterns, Any other distributed training metadata, etc.

Conceptually, the pseudocode is as follows:
```python
state = distributed_config.state
rank = torch.distributed.get_rank()  # current process rank
assignments = assign_fn(params, state=state)  # done once in `class Muon`'s `__init__`
muon_step(params, grads, momentum_buffers, assignments, lr, beta, ns_steps):
    for param_idx, (param, grad, momentum_buffer) in enumerate(zip(params, grads, momentum_buffers)):
        momentum_buffer = beta * momentum_buffer + grad

        # gather_fn returns the full tensor on dst_rank, None on other ranks
        momentum_buffer_full = gather_fn(momentum_buffer, dst_rank=assignments[param_idx], state=state)
        update_full = None
        if rank == assignments[param_idx]:
            update_full = _zeropower_via_newtonschulz(momentum_buffer_full, ns_steps)

        # redistribute_fn receives None on non-src_rank, but all ranks participate as destinations
        update = redistribute_fn(update_full, src_rank=assignments[param_idx], state=state)

        param -= lr * update
```

### Performance / Acceleration
**Prefetching**: When orthogonalizing the momentum buffer for parameter N to produce an update, the momentum buffer for parameter N+1 can be prefetched via `gather_fn` to overlap communication with computation.
- `prefetch_count`: Controls how many parameters' momentum buffers to prefetch ahead
- `prefetch_count=0`: Disables prefetching (sequential communication and computation)
- `prefetch_count=1`: Default behavior (prefetch next parameter's momentum buffer)

**Independent Asynchronous Processing**: GPUs process different parameters simultaneously in parallel, with each GPU working on the subset of parameters assigned to its rank.
- `async_gpu_parallelism`: Enable/disable parallel processing (default: `True`)

### Notes on Parallelism:

| **Category** | **Examples** | **Parameter Storage** | **`gather_fn` Behavior** | **`redistribute_fn` Behavior** |
|--------------|--------------|----------------------|--------------------------|-------------------------------|
| **Sharded** | TP, FSDP, HSDP shard dimension | Parameters sharded across devices | Must gather shards into full tensor on dst_rank | Must scatter update back to shards on all ranks |
| **Replicated** | DDP, CP, HSDP replicate dimension | Full copy on each device | Return local tensor on dst_rank (already complete), None on others | Must broadcast update to all replicas |
| **Independent** | EP, PP | Different parameters on each device | Return local tensor (no gathering needed) | No distribution needed (params independent) |
| **Combined** | FSDP+TP, HSDP, etc. | Mix of above strategies | Compose gather operations across all dimensions | Compose redistribute operations across all dimensions |

**Key Points:**
- **Sharded strategies** require actual gather/scatter collectives to reconstruct/redistribute tensors
- **Replicated strategies** use `gather_fn` for API consistency (returns local tensor on dst_rank), then broadcast via `redistribute_fn`
- **Independent strategies** should still pass their process groups to `assign_fn` to ensure correct rank assignment and work deduplication
- **Combined strategies** require the `create_*_config` helpers to compose operations across all parallelism dimensions

### `DistributedConfig` API

```python
@dataclass
class DistributedConfig:
    """Configuration for distributed Muon training.

    This is user-generated, or can be generated via the helper functions in the
    "Quality of Life (QoL) Features" section.
    """
    assign_fn: Callable[[list[Tensor], dict[str, Any]], dict[int, int]]
    # Maps param_index → rank assigned to perform orthogonalization
    # Called once during __init__
    # Signature: assign_fn(params, state) -> {param_idx: rank}
    # Must return an entry for every param_idx in range(len(params))
    # Common strategy: round-robin assignment across ranks
    # Future work: load-balanced assignment by parameter size

    gather_fn: Callable[[Tensor, int, dict[str, Any]], Optional[Tensor]]
    # For parallelism strategies that need it, gathers the full momentum buffer from shards for orthogonalization
    # Signature: gather_fn(momentum_buffer, dst_rank, state) -> momentum_buffer_full or None
    # Input tensor can be either a shard (for sharded strategies like FSDP, TP) or a full tensor (for replicated strategies like DDP, CP)
    # Returns the full tensor on dst_rank, None on other ranks

    redistribute_fn: Callable[[Tensor | None, int, dict[str, Any]], Tensor]
    # Redistributes the orthogonalized update back to shards and replicas from the rank that orthogonalized the update
    # Signature: redistribute_fn(update, src_rank, state) -> update_shard
    # Receives None on non-src_rank, but all ranks participate as destinations

    state: dict[str, Any]
    # Holds all metadata needed by the functions above:
    # - Process groups (e.g., FSDP, TP, DP process groups)
    # - Device meshes
    # - Tensor shapes/dtypes
    # - Communication patterns
    # - Any other distributed training metadata

    async_gpu_parallelism: bool = True
    # If True: Each rank processes its assigned parameters asynchronously in parallel
    # If False: Each rank processes its assigned parameters one at a time (easier debugging)
    # Recommended: False during initial development/debugging, True for production

    prefetch_count: int = 1
    # Number of tensors to prefetch ahead while processing current tensor
    # 0: Disabled (sequential communication and computation)
    # 1-2: Recommended (overlaps communication with computation)
    # 3+: Higher memory usage, diminishing returns
    # Uses async_op=True in distributed collectives
    # Note: Prefetching works independently of async_gpu_parallelism
```

### Quality of Life (QoL) Features
Parallelism setups are often bespoke. `DistributedConfig` enables users to define `assign_fn`, `gather_fn`, and `redistribute_fn` for their specific implementation of distributed training.

For convenience, we provide helper functions to generate `DistributedConfig` for common distributed training setups:
- Use `create_processgroup_config()` if you manually created process groups for your parallelism strategy
- Use `create_devicemesh_config()` if you're using PyTorch's DeviceMesh API
- Use `create_dtensor_config()` if your model parameters are DTensors

#### Process Group Configuration
```python
def create_processgroup_config(
    fsdp_pg: Optional[ProcessGroup] = None,
    tp_pg: Optional[ProcessGroup] = None,
    dp_pg: Optional[ProcessGroup] = None,
    ep_pg: Optional[ProcessGroup] = None,
    cp_pg: Optional[ProcessGroup] = None,
    pp_pg: Optional[ProcessGroup] = None,
    async_gpu_parallelism: bool = True,
    prefetch_count: int = 1,
) -> DistributedConfig:
    """
    Create DistributedConfig from PyTorch process groups.

    Args:
        fsdp_pg: Fully Sharded Data Parallel process group
        tp_pg: Tensor Parallel process group
        dp_pg: Data Parallel process group
        ep_pg: Expert Parallel process group
        cp_pg: Context Parallel process group
        pp_pg: Pipeline Parallel process group
        async_gpu_parallelism: Enable async parallel processing
        prefetch_count: Number of tensors to prefetch ahead

    Returns:
        DistributedConfig with appropriate assign/gather/redistribute functions
    """
```

#### Device Mesh Configuration
```python
def create_devicemesh_config(
    device_mesh: DeviceMesh,
    mesh_dim_names: list[str],
    async_gpu_parallelism: bool = True,
    prefetch_count: int = 1,
) -> DistributedConfig:
    """
    Create DistributedConfig from PyTorch DeviceMesh.

    Args:
        device_mesh: PyTorch DeviceMesh defining parallelism topology
        mesh_dim_names: Names for each mesh dimension (e.g., ["dp", "fsdp", "tp"])
        async_gpu_parallelism: Enable async parallel processing
        prefetch_count: Number of tensors to prefetch ahead

    Returns:
        DistributedConfig with appropriate assign/gather/redistribute functions
    """
```

#### DTensor Configuration
```python
def create_dtensor_config(
    async_gpu_parallelism: bool = True,
    prefetch_count: int = 1,
) -> DistributedConfig:
    """
    Create DistributedConfig for training setups using PyTorch DTensor.

    Automatically infers parallelism strategy from DTensor placement specs.

    Args:
        async_gpu_parallelism: Enable async parallel processing
        prefetch_count: Number of tensors to prefetch ahead

    Returns:
        DistributedConfig that uses DTensor placement information for
        assign/gather/redistribute operations
    """
```

## Design Summary / Principles

- **Flexibility**: Works with any parallelism configuration through `DistributedConfig`.
- **Simple**: User only needs to specify 3 functions `assign_fn`, `gather_fn`, and `redistribute_fn`.
- **Quality of Life**: can just use `create_processgroup_config()`, `create_devicemesh_config()`, or `create_dtensor_config()`.
- **Performant**: `prefetch_count` and `async_gpu_parallelism` enable faster parallel computation.
- **Debug Mode**: disable `prefetch_count` and `async_gpu_parallelism` for slower but more debuggable computation; to disable set `prefetch_count=0` and `async_gpu_parallelism=False`.

## Quick Start

### Example 1: FSDP Training
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Muon
from torch.optim.muon import create_processgroup_config

# Setup FSDP model
model = MyModel()
model = FSDP(model, ...)

# Create Muon optimizer with distributed support
optimizer = Muon(
    model.parameters(),
    lr=0.02,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    distributed_config=create_processgroup_config(
        fsdp_pg=model.process_group,
    )
)
```

### Example 2: FSDP + Tensor Parallel
```python
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import Muon
from torch.optim.muon import create_devicemesh_config

# Setup device mesh for combined parallelism
device_mesh = init_device_mesh("cuda", (4, 8), mesh_dim_names=["dp", "tp"])
# Apply FSDP + TP to model...

optimizer = Muon(
    model.parameters(),
    lr=0.02,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    distributed_config=create_devicemesh_config(
        device_mesh=device_mesh,
        mesh_dim_names=["dp", "tp"]
    )
)
```

### Example 3: DTensor-based Training
```python
from torch.optim import Muon
from torch.optim.muon import create_dtensor_config

# Model parameters are DTensors with placement specs
optimizer = Muon(
    model.parameters(),
    lr=0.02,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    distributed_config=create_dtensor_config()
)
```

### Example 4: Non-distributed Training (Backward Compatible)
```python
from torch.optim import Muon

# No distributed_config parameter - uses standard single-device behavior
optimizer = Muon(
    model.parameters(),
    lr=0.02,
    momentum=0.95,
    nesterov=True,
    ns_steps=5
)
```


## Implementation Details

### Integration with Muon Optimizer

Changes will be made to `/data/users/vchiley/pytorch/torch/optim/_muon.py` to integrate the `distributed_config` parameter:

```python
class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        # ... existing initialization code ...

        # Setup distributed training if config is provided
        if distributed_config is not None:
            # Store assignments in distributed_config.state for use during step()
            distributed_config.state['assignments'] = distributed_config.assign_fn(
                self.param_groups[0]['params'],
                distributed_config.state
            )
            self.distributed_config.state['rank'] = torch.distributed.get_rank()
            self.distributed_config = distributed_config
        else:
            # Backward compatibility: None means standard non-distributed behavior
            self.distributed_config = None

    def step(self, closure=None):
        # ... existing code ...

        if self.distributed_config is None:
            # Standard non-distributed path
            for param, grad, momentum_buffer in zip(params, grads, momentum_buffers):
                momentum_buffer.lerp_(grad, 1 - beta)
                update = _zeropower_via_newtonschulz(momentum_buffer, ns_steps)
                param.add_(update, alpha=-lr)
        else:
            # Distributed path with gather/redistribute
            assignments = self.distributed_config.state['assignments']
            rank = self.distributed_config.state['rank']

            for param_idx, (param, grad, momentum_buffer) in enumerate(zip(params, grads, momentum_buffers)):
                momentum_buffer.lerp_(grad, 1 - beta)

                # Gather full momentum buffer on assigned rank
                momentum_buffer_full = self.distributed_config.gather_fn(
                    momentum_buffer,
                    dst_rank=assignments[param_idx],
                    state=self.distributed_config.state
                )

                # Orthogonalize only on assigned rank
                update_full = None
                if rank == assignments[param_idx]:
                    update_full = _zeropower_via_newtonschulz(momentum_buffer_full, ns_steps)

                # Redistribute update to all ranks
                update = self.distributed_config.redistribute_fn(
                    update_full,
                    src_rank=assignments[param_idx],
                    state=self.distributed_config.state
                )

                param.add_(update, alpha=-lr)
```

### Edge Cases and Requirements

#### 1. Parameter Dimension Constraints
Muon **does not operate** on parameters with fewer than 2 dimensions. This includes:
- **Scalars** (0D tensors): e.g., learnable temperature parameters
- **1D tensors**: e.g., biases, LayerNorm scales/shifts, embedding scales

These parameters are not accepted as input to the optimizer and are handled by a different optimizer.

#### 2. Dynamic Parameter Changes
If parameters are added or removed after optimizer initialization, we would ideally like to rerun `assign_fn` to update the assignments.

```python
# TODO: Future Enhancement
# When param_groups are modified, recompute assignments:
# if distributed_config is not None:
#     distributed_config.state['assignments'] = distributed_config.assign_fn(
#         self.param_groups[0]['params'],
#         distributed_config.state
#     )
```

**Note:** This is not implemented in the initial version. Users should avoid modifying `param_groups` after initialization when using `distributed_config`.

#### 3. Tensor Shape Validation
The `gather_fn` and helper functions (`create_processgroup_config`, etc.) should perform shape validation when possible.

#### 4. Rank Assignment Validation
The `assign_fn` must:
- Return an assignment for **every** parameter index in `range(len(params))`
- Assign each parameter to a **valid rank** within the process group
- Handle parameters that span multiple ranks (for replicated strategies)

Example validation in helper functions:
```python
def validate_assignments(assignments, params, world_size):
    assert len(assignments) == len(params), "Missing assignments for some parameters"
    for param_idx, rank in assignments.items():
        assert 0 <= rank < world_size, f"Invalid rank {rank} for param {param_idx}"
```

#### 5. Communication Failure Handling
If network communication fails during `gather_fn` or `redistribute_fn`, the **entire training job should fail** with an appropriate error message. PyTorch's distributed collectives will raise exceptions on communication failures - these should propagate up and terminate the job.

Do not attempt to:
- Retry failed communications automatically
- Continue training with partial data
- Silently fall back to non-distributed mode

## Performance and Memory Considerations

### Memory Overhead

**Prefetching Memory Cost:**
- Each prefetched tensor requires memory equal to the full parameter size
- Formula: `Additional Memory ≈ prefetch_count × (sum of prefetched parameter sizes)`
- Example: For a 7B parameter model with `prefetch_count=2`, if prefetching the two largest layers (500M params each), you need ~4GB additional memory per rank (assuming fp32)

**Recommendations:**
- Start with `prefetch_count=1` (default) for most workloads
- Increase to `prefetch_count=2` only if profiling shows significant communication stalls
- Avoid `prefetch_count > 2` unless you have very large memory headroom and confirmed bottlenecks

**Monitoring:** Use `torch.cuda.max_memory_allocated()` before and after optimizer initialization to measure actual overhead.

### Compute-Communication Overlap

**Prefetching** enables overlapping communication (gathering next tensor) with computation (orthogonalizing current tensor).

**Effectiveness depends on:**
- Ratio of computation time to communication time
- Network bandwidth vs GPU compute throughput
- Parameter sizes (larger parameters benefit more from prefetch)

**Tuning Guidelines:**
- If `orthogonalization_time >> communication_time`: Prefetching provides minimal benefit
- If `orthogonalization_time < communication_time`: Prefetching can provide 20-40% speedup
- Use PyTorch profiler to measure these timings for your specific setup

### Async GPU Parallelism

When `async_gpu_parallelism=True`, each rank processes its assigned parameters in parallel without waiting for other ranks.

**Memory Impact:**
- Minimal additional memory (only buffering for async operations)
- Same peak memory as synchronous mode

**Computation Impact:**
- Can reduce total wall-clock time by the number of GPUs used when parameters are well-balanced across ranks
- Less effective if assignment is imbalanced (some ranks idle while others work)

**Trade-offs:**
- **Enable (default)** for production training - faster iteration time
- **Disable** for debugging - easier to trace execution flow and identify issues

### Combined Prefetch + Async

Using both `prefetch_count=1` and `async_gpu_parallelism=True` (the defaults) typically provides the best performance:
- Each rank overlaps its own communication and computation
- All ranks work in parallel on different parameters

## Testing Strategy

### Unit Tests

**Test Individual Components:**
- **`assign_fn` validation**: Test round-robin distribution, validate all parameters are assigned, validate ranks are within bounds
- **`gather_fn` correctness**: Test gathering from shards produces correct full tensor on dst_rank, returns None on other ranks
- **`redistribute_fn` correctness**: Test distributing from src_rank produces correct shards/replicas on all ranks
- **Helper function outputs**: Verify `create_processgroup_config`, `create_devicemesh_config`, `create_dtensor_config` produce valid `DistributedConfig` instances

**Test each parallelism strategy in isolation:**
```python
# Test FSDP gather/redistribute
def test_fsdp_gather_redistribute():
    # Setup FSDP process group with 4 ranks
    # Create sharded tensor on each rank
    # Call gather_fn, verify full tensor on dst_rank
    # Call redistribute_fn, verify correct shards on all ranks

# Test TP gather/redistribute
def test_tp_gather_redistribute():
    # Setup TP process group
    # Create tensor sharded across TP dimension
    # Verify gather reconstructs full tensor
    # Verify redistribute produces correct shards

# Test DDP (replicated) behavior
def test_ddp_gather_redistribute():
    # Setup DDP process group
    # Create replicated tensor
    # Verify gather returns local tensor on dst_rank, None elsewhere
    # Verify redistribute broadcasts to all replicas
```

### Integration Tests

**Test Full Optimizer Step:**
```python
def test_muon_distributed_equivalence():
    """
    Verify distributed and non-distributed Muon produce same results.

    1. Initialize identical models and data
    2. Run single-GPU training with distributed_config=None
    3. Run multi-GPU training with distributed_config
    4. Compare final parameters (should match within numerical tolerance)
    """
    # Single-GPU baseline
    model_single = create_model()
    optimizer_single = Muon(model_single.parameters(), lr=0.02)
    train(model_single, optimizer_single, steps=100)

    # Multi-GPU distributed
    model_distributed = create_model()  # Same initialization
    optimizer_distributed = Muon(
        model_distributed.parameters(),
        lr=0.02,
        distributed_config=create_processgroup_config(fsdp_pg=...)
    )
    train(model_distributed, optimizer_distributed, steps=100)

    # Compare results
    for p1, p2 in zip(model_single.parameters(), model_distributed.parameters()):
        torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5)
```

**Test Combined Parallelism Strategies:**
```python
def test_fsdp_tp_combined():
    """Test FSDP + TP combination"""
    device_mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=["dp", "tp"])
    # Apply parallelism to model
    config = create_devicemesh_config(device_mesh, ["dp", "tp"])
    optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)
    # Train and verify correctness

def test_hsdp():
    """Test Hybrid Sharded Data Parallel (FSDP replicate + shard dimensions)"""
    # Similar structure to above
```

### Performance Tests

**Verify Optimizations Work:**
```python
def test_prefetch_speedup():
    """Verify prefetching reduces wall-clock time"""
    # Benchmark with prefetch_count=0
    time_no_prefetch = benchmark_training(prefetch_count=0)

    # Benchmark with prefetch_count=1
    time_with_prefetch = benchmark_training(prefetch_count=1)

    # Expect 20-40% speedup in bandwidth-limited scenarios
    assert time_with_prefetch < time_no_prefetch * 0.9

def test_async_parallelism_speedup():
    """Verify async GPU parallelism reduces wall-clock time"""
    # Similar benchmarking approach
```

### Edge Case Tests

**Test Error Handling:**
```python
def test_invalid_assignments():
    """Test that invalid rank assignments are caught"""
    # Assign parameter to rank >= world_size
    # Should raise ValueError

def test_missing_assignments():
    """Test that missing parameter assignments are caught"""
    # assign_fn returns dict missing some param_idx
    # Should raise AssertionError

def test_shape_mismatch():
    """Test that shape mismatches in gather are caught"""
    # gather_fn returns wrong shape
    # Should raise RuntimeError with helpful message

def test_communication_failure():
    """Test that communication failures propagate correctly"""
    # Simulate network failure during gather/redistribute
    # Should raise exception and fail training job
```

### Correctness Validation for Users

Users can validate their distributed setup produces correct results by comparing against single-GPU baseline:

```python
# 1. Single-GPU reference
model_ref = MyModel()
optimizer_ref = Muon(model_ref.parameters(), lr=0.02)
for data in dataloader:
    loss = model_ref(data)
    loss.backward()
    optimizer_ref.step()

# 2. Multi-GPU distributed
model_dist = MyModel()  # Same initialization as model_ref
model_dist = FSDP(model_dist, ...)
optimizer_dist = Muon(
    model_dist.parameters(),
    lr=0.02,
    distributed_config=create_processgroup_config(fsdp_pg=model_dist.process_group)
)
for data in dataloader:
    loss = model_dist(data)
    loss.backward()
    optimizer_dist.step()

# 3. Compare on rank 0
if torch.distributed.get_rank() == 0:
    # Gather parameters from all ranks if needed
    # Compare with model_ref parameters
    # Should match within tolerance: torch.testing.assert_close(..., rtol=1e-5, atol=1e-5)
```

**Expected Tolerance:**
- **FP32**: `rtol=1e-5, atol=1e-5`
- **BF16/FP16**: `rtol=1e-3, atol=1e-3` (lower precision accumulates more error)

## Implementation Roadmap

This feature will be implemented in phases to ensure correctness and maintainability:

### Phase 1: Basic Distributed Support (Core Functionality)
**Goal:** Get basic distributed orthogonalization working without optimizations

- [ ] Add `DistributedConfig` dataclass to `/data/users/vchiley/pytorch/torch/optim/_muon.py`
- [ ] Modify `Muon.__init__()` to accept `distributed_config` parameter
- [ ] Implement distributed path in `Muon.step()` using `gather_fn` and `redistribute_fn`
- [ ] Implement `create_processgroup_config()` for basic FSDP support
- [ ] Set `prefetch_count=0` and `async_gpu_parallelism=False` for this phase
- [ ] Add validation for parameter dimensions (filter out scalars and 1D tensors)
- [ ] Add rank assignment validation

**Success Criteria:** Training runs successfully with FSDP, produces same numerical results as non-distributed (within tolerance)

### Phase 2: Process Group Helpers
**Goal:** Support common distributed training configurations

- [ ] Extend `create_processgroup_config()` to handle TP, DP, EP, CP, PP
- [ ] Handle combined parallelism strategies (e.g., FSDP + TP)
- [ ] Add comprehensive shape validation in gather/redistribute functions
- [ ] Document expected behavior for each parallelism type

**Success Criteria:** Users can easily configure Muon for standard parallelism setups

### Phase 3: Prefetching Optimization
**Goal:** Overlap communication with computation

- [ ] Implement prefetch buffer management in `Muon.step()`
- [ ] Add `prefetch_count` parameter support
- [ ] Use `async_op=True` in distributed collectives
- [ ] Handle edge cases (first/last parameter, buffer wraparound)

**Success Criteria:** Prefetching reduces wall-clock time by 20-40% in bandwidth-limited scenarios

### Phase 4: Async GPU Parallelism
**Goal:** Enable parallel processing across ranks

- [ ] Implement async processing logic in `Muon.step()`
- [ ] Add synchronization points where necessary
- [ ] Ensure correctness with async execution

**Success Criteria:** Async mode reduces wall-clock time by additional 20-30% vs prefetch alone

### Phase 5: Additional Configuration Helpers
**Goal:** Support advanced distributed APIs

- [ ] Implement `create_devicemesh_config()`
- [ ] Implement `create_dtensor_config()`
- [ ] Add automatic strategy detection from DTensor placement

**Success Criteria:** DeviceMesh and DTensor users can easily configure Muon

### Phase 6: Optimization and Polish
**Goal:** Production-ready performance and usability

- [ ] Profile and optimize hot paths
- [ ] Add memory usage documentation and warnings
- [ ] Add performance tuning guide
- [ ] Consider load-balanced assignment strategies (by parameter size)
- [ ] Add telemetry/logging for debugging distributed issues

**Success Criteria:** Production-grade performance and user experience

## Backward Compatibility

**Backward compatibility is maintained** by making `distributed_config` an optional parameter with default value `None`.

**When `distributed_config=None` (default):**
- Muon operates in standard single-device mode
- No distributed communication occurs
- Existing user code continues to work without modification

**Migration path for existing users:**
```python
# Existing code (still works)
optimizer = Muon(model.parameters(), lr=0.02)

# New distributed code (opt-in)
optimizer = Muon(
    model.parameters(),
    lr=0.02,
    distributed_config=create_processgroup_config(fsdp_pg=model.process_group)
)
```

No breaking changes to existing APIs.
