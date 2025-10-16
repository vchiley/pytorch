# CLAUDE_DESIGN.md - Muon Distributed Training Architecture

This document details the design decisions, architecture, and implementation strategy for distributed Muon optimizer.

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Design Goals](#design-goals)
3. [Architecture Overview](#architecture-overview)
4. [Async Pipeline Architecture](#async-pipeline-architecture)
5. [API Design Rationale](#api-design-rationale)
6. [Process Group Topologies](#process-group-topologies)
7. [Memory Management](#memory-management)
8. [Alternative Designs Considered](#alternative-designs-considered)

## Problem Statement

### Core Challenge
The Muon optimizer's orthogonalization step requires operating on the **full matrix structure** to maintain mathematical correctness. However, distributed training shards parameters and their updates across devices:

- **FSDP**: Shards parameters across data-parallel ranks
- **Tensor Parallel (TP)**: Shards layers across tensor-parallel ranks
- **HSDP**: Two-level sharding (shard within node, replicate across nodes)
- **Expert Parallel (EP)**: Shards expert weights in MoE models across devices
- **Context Parallel (CP)**: Replicates parameters but partitions sequence across devices
- **Pipeline Parallel (PP)**: Partitions model layers across devices

Orthogonalizing partial matrices separately produces **incorrect results** - the orthogonalization operation is not distributive across matrix shards.

**Special case - Context Parallel**: CP replicates model parameters (not sharded), but still requires coordination to avoid redundant orthogonalization work across ranks.

### Performance Challenge
Naive solution: synchronously gather → orthogonalize → redistribute for each parameter sequentially.

**Problem**: Creates massive GPU idle time where devices wait for:
1. Other GPUs to finish their work
2. Network communication to complete
3. Single GPU to finish orthogonalization

For a model with 100 parameters and 8 GPUs, this means 7 GPUs sit idle while 1 works, then all 8 wait for redistribution before moving to the next parameter.

## Design Goals

1. **Correctness**: Guarantee mathematically correct orthogonalization
2. **Efficiency**: Maximize GPU utilization via parallelism
3. **Flexibility**: Support any parallelism strategy or hybrid combination
4. **Simplicity**: Single API that handles all cases automatically
5. **Backward Compatibility**: No changes for non-distributed users
6. **Memory Efficiency**: Bounded memory overhead per GPU

## Architecture Overview

### Unified Implementation Design

**Core principle**: ONE implementation function handles both single-GPU and distributed cases.

**Why orthogonalize updates instead of parameters?** More efficient - only the update (momentum buffer) needs orthogonalization and redistribution. Parameters stay sharded; we apply the orthogonalized update to them in-place. This avoids the overhead of gathering and redistributing the full sharded parameters.

```python
def _single_tensor_muon(..., distributed_config: Optional[DistributedConfig] = None):
    # Update momentum (always)
    for param, grad, buf in zip(params, grads, momentum_bufs):
        buf.lerp_(grad, 1 - momentum)

    # Distributed operations gated behind distributed_config
    if distributed_config is not None:
        # Distributed path: gather update → orthogonalize → redistribute → apply
        full_update = config.gather_fn(buf, rank, state)
        ortho_update = _zeropower_via_newtonschulz(full_update, ...)
        shard_update = config.redistribute_fn(ortho_update, rank, state)
        param.add_(shard_update, alpha=-lr)
    else:
        # Single-GPU path: orthogonalize update and apply
        ortho_update = _zeropower_via_newtonschulz(buf, ...)
        param.add_(ortho_update, alpha=-lr)
```

**Benefits:**
- No function proliferation (no `_muon_distributed`, `_muon_sequential`, etc.)
- Single code path to maintain and test
- Distributed logic isolated in config, core algorithm unified
- `async_gpu_parallelism` flag naturally lives in the config

### Three-Function Interface

```python
@dataclass
class DistributedConfig:
    assign_fn: Callable[[list[Tensor], dict[str, Any]], dict[int, list[int]]]
    gather_fn: Callable[[Tensor, int, dict[str, Any]], Tensor]
    redistribute_fn: Callable[[Tensor, int, dict[str, Any]], Tensor]
    state: dict[str, Any]
    async_gpu_parallelism: bool = True  # False for sequential debugging
    prefetch_count: int = 1  # Updates to prefetch (0=disabled, 1-3 recommended)
```

**Design rationale**: Separates three orthogonal concerns:
1. **Workload distribution** (assign_fn): Which GPU processes which parameters?
2. **Data gathering** (gather_fn): How to assemble full update on assigned GPU?
3. **Data redistribution** (redistribute_fn): How to scatter orthogonalized update back to shards?

**Gating mechanism**: These functions are only called when `distributed_config` is not None. For single-GPU training, the optimizer skips the distributed path entirely.

### Key Insight: Composability

By separating these concerns, we can **compose** gathering strategies:

```
HSDP = gather_across_shard_group() + gather_across_replicate_group()
TP+FSDP = gather_across_tp_group() + gather_across_fsdp_group()
TP+HSDP = gather_across_tp_group() + gather_across_shard_group() + gather_across_replicate_group()
```

The `state` dict provides all necessary process groups, and `create_auto_config()` automatically chains the right gather operations.

## Async Pipeline Architecture

### Sequential Baseline (Naive Approach)

```
GPU 0: [gather_p0] → [ortho_p0] → [redist_p0] → [gather_p1] → [ortho_p1] → [redist_p1] → ...
GPU 1: [   idle   ] → [  idle  ] → [  idle   ] → [gather_p1] → [ortho_p1] → [redist_p1] → ...
GPU 2: [   idle   ] → [  idle  ] → [  idle   ] → [   idle  ] → [  idle  ] → [  idle   ] → ...
...
```

**Problem**: O(N * num_params) time where N = number of GPUs. GPUs sit idle most of the time.

### Parallel Pipeline (Our Design)

```
Time →

GPU 0: [gather_p0] [ortho_p0] [redist_p0] [gather_p3] [ortho_p3] [redist_p3] ...
GPU 1: [gather_p1] [ortho_p1] [redist_p1] [gather_p4] [ortho_p4] [redist_p4] ...
GPU 2: [gather_p2] [ortho_p2] [redist_p2] [gather_p5] [ortho_p5] [redist_p5] ...
...
```

**Speedup**: O(num_params) time. All GPUs work in parallel on different parameters.

### Prefetching Pipeline (Optimized)

```
Time →

GPU 0: [gather_p0, p3] [ortho_p0] [redist_p0, gather_p6] [ortho_p3] [redist_p3, gather_p9] ...
       └─ prefetch ─┘   └─ work ─┘  └──── overlap ────┘
```

**Key optimizations**:
1. **Prefetch `prefetch_count` updates ahead** while working on current update
2. **Use `async_op=True`** in all collectives to overlap communication with computation
3. **Overlap redistribution with next gather** using separate CUDA streams

**Expected speedup**: 2-3x over non-prefetching parallel version.

### Two Levels of Asynchrony

The design has **two orthogonal async concepts**:

#### 1. GPU-Level Parallelism (`async_gpu_parallelism`)

Controls whether GPUs process parameters in parallel or sequentially.

**Parallel mode** (`async_gpu_parallelism=True`, default):
```
Time →

GPU 0: [gather_p0] [ortho_p0] [redist_p0] [gather_p3] [ortho_p3] ...
GPU 1: [gather_p1] [ortho_p1] [redist_p1] [gather_p4] [ortho_p4] ...
GPU 2: [gather_p2] [ortho_p2] [redist_p2] [gather_p5] [ortho_p5] ...
```

All GPUs work simultaneously on different parameters. Optimal performance.

**Sequential mode** (`async_gpu_parallelism=False`, debug only):
```
Time →

GPU 0: [gather_p0] [ortho_p0] [redist_p0] [gather_p3] [ortho_p3] [redist_p3] ✓
       [────── wait (barrier) ──────]
GPU 1:                                                              [gather_p1] [ortho_p1] ...
GPU 2:                                                              [   idle   ] [  idle  ] ...
```

GPUs process sequentially: GPU 0 finishes all its params, then GPU 1, etc. Clear execution order for debugging.

**Implementation:**
```python
if config.async_gpu_parallelism:
    # Parallel: Each GPU processes its assigned parameters independently
    for param_idx in assignments[rank]:
        full_update = gather_fn(update[param_idx], rank, state)
        ortho_update = orthogonalize(full_update)
        shard_update = redistribute_fn(ortho_update, rank, state)
        param[param_idx].add_(shard_update, alpha=-lr)
else:
    # Sequential: Wait for previous GPUs to finish
    for gpu_rank in range(world_size):
        if rank == gpu_rank:
            for param_idx in assignments[rank]:
                full_update = gather_fn(update[param_idx], rank, state)
                ortho_update = orthogonalize(full_update)
                shard_update = redistribute_fn(ortho_update, rank, state)
                param[param_idx].add_(shard_update, alpha=-lr)
        dist.barrier()  # All GPUs wait before next GPU proceeds
```

#### 2. Operation-Level Async (Prefetching)

Controls prefetching and overlapping communication with computation **within a single GPU's work**. This is controlled by `prefetch_count` in `DistributedConfig`.

**With prefetching enabled** (`prefetch_count > 0`, default: 1):
```
GPU 0 timeline (prefetch_count=1):
[gather_p0] [ortho_p0] [redist_p0, gather_p3] [ortho_p3] [redist_p3, gather_p6] ...
            └─ work ─┘  └──── overlap ────┘   └─ work ─┘  └──── overlap ────┘
```

Prefetch next `prefetch_count` updates while working on current one. The implementation uses `async_op=True` in distributed collectives. Works in both parallel and sequential GPU modes.

**With prefetching disabled** (`prefetch_count=0`, simplest debugging):
```
GPU 0 timeline:
[gather_p0] [ortho_p0] [redist_p0] [gather_p3] [ortho_p3] [redist_p3] ...
```

No prefetching. Simplest execution, slowest performance. The implementation uses `async_op=False` in distributed collectives.

**User control**:
- `async_gpu_parallelism`: Controls GPU-level parallelism (parallel vs sequential GPU execution)
- `prefetch_count`: Controls operation-level async (0=disabled, 1-3 recommended)

**Independence**: These two levels are orthogonal. You can have sequential GPU execution but still use operation-level async for prefetching within each GPU's work.

### Pipeline Phases

#### Phase 1: Assign Parameters to GPUs

```python
assignments = assign_fn(parameters, state)
# Returns: {0: [0, 3, 6, 9], 1: [1, 4, 7, 10], 2: [2, 5, 8, 11], ...}
```

Each GPU gets roughly equal number of parameters to orthogonalize.

#### Phase 2: Parallel Gather-Orthogonalize-Redistribute

```python
for param_idx in assignments[rank]:
    # Prefetch next prefetch_count updates (async)
    # Note: gather_fn/redistribute_fn use async_op=True internally when prefetching is enabled
    if param_idx + N in assignments[rank]:
        gather_handle = gather_fn(update[param_idx + N], rank, state)  # Internally uses async_op=True

    # Gather current update (if not prefetched)
    full_update = wait_for_gather() if prefetched else gather_fn(update[param_idx], rank, state)

    # Orthogonalize update (computation)
    ortho_update = _zeropower_via_newtonschulz(full_update)

    # Redistribute orthogonalized update (async, overlapped with next gather)
    shard_update = redistribute_fn(ortho_update, rank, state)  # Internally uses async_op=True when prefetching enabled

    # Apply orthogonalized update to sharded parameter
    param[param_idx].add_(shard_update, alpha=-lr)
```

**Note**: The `async_op` behavior is controlled by operation-level async settings, not passed as a parameter to gather_fn/redistribute_fn. The implementation internally uses `async_op=True` in distributed collectives when prefetching is enabled.

#### Phase 3: Synchronization

```python
# Wait for all outstanding async operations
torch.cuda.synchronize()
dist.barrier(process_group)  # Ensure all GPUs finished
```

## API Design Rationale

### Why Not Built-in Strategies?

**Considered**:
```python
Muon(..., parallelism="fsdp")  # or "tp", "hsdp", etc.
```

**Rejected because**:
1. PyTorch has dozens of parallelism strategies and combinations
2. New strategies emerge (e.g., Context Parallel, Expert Parallel)
3. User-specific hybrid configurations
4. Would require maintaining N! combinations

### Why Functional Interface?

**Chosen approach**: User provides three functions.

**Advantages**:
1. **Flexibility**: Supports any current or future parallelism strategy
2. **Composability**: Users can chain gather operations for hybrid parallelism
3. **Testability**: Easy to unit test individual functions
4. **Debuggability**: Users can instrument their own functions
5. **Zero maintenance**: PyTorch doesn't maintain parallelism-specific code

### Why `create_auto_config()`?

**Core API**: The three functions (`assign_fn`, `gather_fn`, `redistribute_fn`) are the fundamental interface. Users can always define these directly for maximum flexibility.

**Convenience helper**: `create_auto_config()` is provided to **generate** these functions for common PyTorch parallelism strategies, avoiding boilerplate for 95% of users.

```python
# Instead of writing custom functions for FSDP+TP...
config = create_auto_config(
    tp_process_group=tp_pg,
    fsdp_process_group=fsdp_pg,
    tp_shard_dim=1
)
# ...create_auto_config() generates the three functions automatically
```

**How it works**:
1. Detects which process groups are provided (TP, FSDP, HSDP, EP, CP, PP)
2. Determines correct gather order (TP → FSDP → HSDP layers)
3. Builds composed gather/redistribute pipelines
4. Sets up balanced assignment strategy
5. Returns a `DistributedConfig` with generated functions

**When NOT to use `create_auto_config()`**:
- Using non-PyTorch distributed frameworks (e.g., Horovod, custom MPI)
- Proprietary/custom parallelism strategies
- Need custom assignment logic (e.g., based on parameter sizes, GPU memory)
- Fine-grained control over communication patterns

For these cases, define your own three functions directly.

## Process Group Topologies

### FSDP Only

```
Data Parallel Dimension
┌────────┬────────┬────────┬────────┐
│ GPU 0  │ GPU 1  │ GPU 2  │ GPU 3  │
│ shard0 │ shard1 │ shard2 │ shard3 │
└────────┴────────┴────────┴────────┘
         ↑ FSDP Process Group ↑
```

**Gather strategy**: `all_gather` update across FSDP group

### Tensor Parallel Only

```
Tensor Parallel Dimension (column sharding, tp_shard_dim=1)
┌─────────────┬─────────────┐
│   GPU 0     │   GPU 1     │
│ weight[:,:C]│ weight[:,C:]│
└─────────────┴─────────────┘
    ↑ TP Process Group ↑
```

**Gather strategy**: `all_gather` update across TP group along `tp_shard_dim`

### HSDP (Hybrid Sharded Data Parallel)

```
         Replicate Dimension
         ┌───────────────┐
         │               │
         ↓               ↓
Shard   ┌───────────────────┐
Dim     │ GPU0   GPU1       │  Node 0
        │ shard0 shard1     │
        └───────────────────┘
        ┌───────────────────┐
        │ GPU2   GPU3       │  Node 1
        │ shard0 shard1     │
        └───────────────────┘
```

**Gather strategy**:
1. `all_gather` update within shard group (horizontal)
2. Optionally `all_gather` update across replicate group (vertical) - but usually replicas don't need gathering

### TP + FSDP Hybrid

```
           Tensor Parallel (inner)
           ┌──────────┬──────────┐
           │          │          │
FSDP  ┌────▼────┬─────▼────┬────▼────┬─────▼────┐
(outer)│ GPU0    │ GPU1     │ GPU2    │ GPU3     │
      │ TP0,FS0 │ TP1,FS0  │ TP0,FS1 │ TP1,FS1  │
      └─────────┴──────────┴─────────┴──────────┘
```

**Gather strategy**:
1. First gather update across TP group: GPU0←GPU1, GPU2←GPU3
2. Then gather update across FSDP group: GPU0←GPU2

### Expert Parallel (EP)

```
Expert Parallel Dimension (for MoE models)
┌─────────────┬─────────────┬─────────────┐
│   GPU 0     │   GPU 1     │   GPU 2     │
│ experts 0-3 │ experts 4-7 │ experts 8-11│
└─────────────┴─────────────┴─────────────┘
    ↑ EP Process Group ↑
```

**Gather strategy**: `all_gather` update across EP group along `ep_shard_dim`

**Use case**: Mixture-of-Experts (MoE) models where expert weights are sharded across devices. Each expert's update needs full gathering for orthogonalization.

### Context Parallel (CP)

```
Context Parallel Dimension (for long sequences)
┌─────────────┬─────────────┐
│   GPU 0     │   GPU 1     │
│ seq[:L/2]   │ seq[L/2:]   │
│ params (dup)│ params (dup)│
└─────────────┴─────────────┘
    ↑ CP Process Group ↑
```

**Gather strategy**: No actual gathering needed (parameters already replicated)

**Coordination strategy**:
1. **Gradients all-reduced** (standard DDP behavior) - each GPU has identical gradients
2. **Momentum updated locally** - each GPU computes same momentum state using all-reduced gradients
3. **Work distribution**: Use `assign_fn` to assign each parameter to exactly one GPU in the CP group
   - GPU 0 orthogonalizes param updates [0, 2, 4, ...]
   - GPU 1 orthogonalizes param updates [1, 3, 5, ...]
4. **Exchange results**: After orthogonalization, use broadcast/scatter (NOT all-reduce) to exchange orthogonalized updates
   - Each GPU broadcasts its orthogonalized params to all others
   - OR use all-gather pattern to collect all orthogonalized params
5. This avoids redundant orthogonalization work while maintaining correctness

**Concrete example**:
```python
# Context Parallel gather/redistribute implementation
def cp_gather_fn(update, rank, state):
    # Updates already replicated (computed from all-reduced gradients), no gathering needed
    # Just return the local copy
    return update.clone()

def cp_redistribute_fn(ortho_update, rank, state):
    cp_pg = state["cp_process_group"]
    assignments = state["assignments"]  # Which params this rank orthogonalized
    param_idx = state["current_param_idx"]

    # Find which rank owns this parameter's orthogonalization
    owner_rank = None
    for r, param_indices in assignments.items():
        if param_idx in param_indices:
            owner_rank = r
            break

    # Broadcast from owner to all ranks in CP group
    # (Each GPU only broadcasts updates it orthogonalized)
    dist.broadcast(ortho_update, src=owner_rank, group=cp_pg)

    # Now all ranks have the orthogonalized update
    return ortho_update

# Example workflow for 2 GPUs in CP group with 4 parameters:
# - GPU 0 assigned params [0, 2], GPU 1 assigned params [1, 3]
# - After backward: both GPUs have identical gradients (all-reduced)
# - Both GPUs update momentum locally (identical results)
# - GPU 0 orthogonalizes updates 0 and 2, GPU 1 orthogonalizes updates 1 and 3
# - GPU 0 broadcasts update 0 → both GPUs have ortho update 0
# - GPU 1 broadcasts update 1 → both GPUs have ortho update 1
# - GPU 0 broadcasts update 2 → both GPUs have ortho update 2
# - GPU 1 broadcasts update 3 → both GPUs have ortho update 3
# - Result: Both GPUs have all 4 orthogonalized updates, apply to replicated params
```

**Use case**: Long sequence models where sequence dimension is partitioned but model parameters are replicated across all CP ranks. Unlike other strategies, CP doesn't shard parameters - only the input sequence is partitioned.

**Key difference from DDP**: DDP all-reduces gradients and each GPU updates all parameters. CP all-reduces gradients but splits orthogonalization work, then exchanges (broadcast/scatter) the orthogonalized results.

### Pipeline Parallel (PP)

```
Stage 0      Stage 1      Stage 2      Stage 3
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ GPU 0  │→ │ GPU 1  │→ │ GPU 2  │→ │ GPU 3  │
│ layers │  │ layers │  │ layers │  │ layers │
│ 0-25   │  │ 26-50  │  │ 51-75  │  │ 76-100 │
└────────┘  └────────┘  └────────┘  └────────┘
```

**Special case**: Each stage optimizes independently!
- No cross-stage gathering needed (parameters don't span stages)
- Each GPU creates its own `Muon` instance with stage-specific parameters
- `pp_process_group` provided for metadata only

### Full Hybrid: PP + TP + HSDP (4D Parallelism)

```
Pipeline Stage 0                    Pipeline Stage 1
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  Node 0      Node 1          │    │  Node 2      Node 3          │
│  ┌─────┐    ┌─────┐          │    │  ┌─────┐    ┌─────┐          │
│  │TP0  │    │TP0  │ Shard 0 │    │  │TP0  │    │TP0  │ Shard 0 │
│  │FS0  │    │FS0  │          │    │  │FS0  │    │FS0  │          │
│  ├─────┤    ├─────┤          │    │  ├─────┤    ├─────┤          │
│  │TP1  │    │TP1  │ Shard 1 │    │  │TP1  │    │TP1  │ Shard 1 │
│  │FS1  │    │FS1  │          │    │  │FS1  │    │FS1  │          │
│  └─────┘    └─────┘          │    │  └─────┘    └─────┘          │
│  Replicas across nodes       │    │  Replicas across nodes       │
└─────────────────────────────┘    └─────────────────────────────┘
        layers 0-50                        layers 51-100
```

**Gather strategy per stage**:
1. Gather update across TP group (within node)
2. Gather update across shard group (within node)
3. Replicate group doesn't need gathering (replicas hold same data)

## Memory Management

### Memory Footprint Analysis

**Per GPU memory during orthogonalization**:
```
Base memory: Model shards + optimizer states + gradients
Peak memory: + (prefetch_count full updates being processed)
```

**Example**: 7B parameter model, FSDP across 8 GPUs (assuming uniform layer sizes - actual models may vary)
- Shard size per GPU: ~875M params
- Full update size: ~7B / num_layers ≈ 70M params per layer (assuming ~100 layers)
- Prefetch buffer: `prefetch_count` × 70M × 4 bytes
  - With `prefetch_count=1`: ~280MB additional
  - With `prefetch_count=2`: ~560MB additional
  - With `prefetch_count=3`: ~840MB additional

**Design choice**: No upfront memory checks because:
1. Memory usage is bounded and predictable
2. Updates processed sequentially per GPU
3. Prefetch buffer is small (1-3 updates)
4. OOM during gather will fail gracefully with PyTorch error

### Sequential Processing Per GPU

```python
# GPU 0's assigned parameters: [0, 8, 16, 24, ...]
for param_idx in [0, 8, 16, 24, ...]:
    full_update = gather_fn(update[param_idx])  # Hold temporarily
    ortho_update = orthogonalize(full_update)
    shard_update = redistribute_fn(ortho_update)
    param[param_idx].add_(shard_update, alpha=-lr)  # Apply to sharded param
    del full_update, ortho_update, shard_update  # Free before next iteration
```

**Key property**: Only holds 1 (or up to `prefetch_count`) full updates at a time.

### Prefetch Buffer Management

```python
prefetch_queue = []  # Max size controlled by config.prefetch_count

while param_idx in my_assignments:
    # Start prefetching ahead (if prefetch_count > 0)
    if config.prefetch_count > 0 and len(prefetch_queue) < config.prefetch_count:
        next_idx = param_idx + config.prefetch_count * world_size
        if next_idx in my_assignments:
            handle = gather_fn(..., async_op=True)
            prefetch_queue.append((next_idx, handle))

    # Get next update (from prefetch or gather now)
    if prefetch_queue and prefetch_queue[0][0] == param_idx:
        _, handle = prefetch_queue.pop(0)
        full_update = handle.wait()
    else:
        full_update = gather_fn(update[param_idx])

    # Process and free
    ortho_update = orthogonalize(full_update)
    shard_update = redistribute_fn(ortho_update)
    param[param_idx].add_(shard_update, alpha=-lr)
```

**Prefetch size tuning**: `prefetch_count` (default: 1, set to 0 to disable, recommended: 1-3) balances memory vs. overlap. Higher values increase overlap but consume more memory.

## Alternative Designs Considered

### 1. Parameter-Specific Strategies

```python
Muon(params, strategy_map={
    "transformer.layer.0.weight": "fsdp",
    "transformer.layer.1.weight": "tp",
})
```

**Rejected**: Too verbose, doesn't handle new parameters added to model.

### 2. Automatic Detection

```python
# Automatically detect from parameter attributes
Muon(params)  # Reads param._fsdp_shard_info, param._tp_shard_info
```

**Rejected**:
- Fragile coupling to internal PyTorch attributes
- Different frameworks store metadata differently
- Doesn't work with custom parallelism

### 3. Callback-Based

```python
def on_gather_needed(update, rank):
    # User implements gathering
    return full_update

Muon(params, on_gather=on_gather_needed)
```

**Rejected**: Single callback can't handle complex multi-stage gathering (TP + FSDP + HSDP).

### 4. Builder Pattern

```python
config = (DistributedConfigBuilder()
    .with_fsdp(fsdp_pg)
    .with_tp(tp_pg, shard_dim=1)
    .build())
```

**Rejected**: Less flexible than functional approach, more boilerplate.

### 5. Class Hierarchy

```python
class FSDPStrategy: ...
class TPStrategy: ...
class HybridStrategy(FSDPStrategy, TPStrategy): ...
```

**Rejected**: Explosion of classes for all combinations, hard to extend.

## Implementation Strategy

### Stage 1: Core Infrastructure
1. Add `DistributedConfig` dataclass
2. Implement unified `_single_tensor_muon()` function that gates distributed operations behind `distributed_config`
3. Modify `step()` to pass distributed config to unified function
4. Basic gather/redistribute for FSDP only

**Key decision**: Use ONE unified function instead of separate `_single_tensor_muon` and `_muon_distributed` variants. This avoids code duplication and makes the codebase easier to maintain.

### Stage 2: Auto Configuration
1. Implement `create_auto_config()` with process group detection
2. Add support for FSDP, TP, HSDP, PP
3. Implement composition logic for hybrid parallelism

### Stage 3: Async Optimization
1. Add `async_op=True` to gather/redistribute
2. Implement prefetching pipeline
3. Add CUDA stream management for overlap
4. Performance benchmarking

### Stage 4: Validation & Testing
1. Process group validation in `create_auto_config()`
2. Unit tests for each parallelism mode
3. Correctness tests (compare against single-GPU)
4. Performance regression tests

## Validation Logic

### Process Group Consistency Checks

**FSDP + TP validation**:
```python
def validate_fsdp_tp(fsdp_pg, tp_pg):
    # FSDP and TP groups must be orthogonal
    fsdp_ranks = dist.get_process_group_ranks(fsdp_pg)
    tp_ranks = dist.get_process_group_ranks(tp_pg)

    # Check: current rank appears in exactly one position in each group
    # Check: groups partition the world
    assert len(set(fsdp_ranks) & set(tp_ranks)) == 1  # Only current rank
```

**HSDP validation**:
```python
def validate_hsdp(shard_pg, replicate_pg):
    # Shard and replicate groups must partition world
    # Each rank in exactly one shard group, one replicate group
    shard_ranks = dist.get_process_group_ranks(shard_pg)
    replicate_ranks = dist.get_process_group_ranks(replicate_pg)

    # Shard groups should be node-local, replicate across nodes
    # (This is a heuristic, not strictly enforced)
```

**PP validation**:
```python
def validate_pp(pp_pg, pp_stage_id, pp_num_stages):
    # Stage ID must be in valid range
    assert 0 <= pp_stage_id < pp_num_stages

    # PP group size should equal num_stages
    assert dist.get_world_size(pp_pg) == pp_num_stages
```

## Performance Characteristics

### Expected Performance vs Baseline

**Baseline**: Sequential gather-orthogonalize-redistribute
- **Time complexity**: O(N × P) where N=num_GPUs, P=num_params
- **GPU utilization**: ~12.5% (1/8 GPUs working at a time on 8-GPU setup)

**Our design (parallel)**:
- **Time complexity**: O(P / N) — optimal!
- **GPU utilization**: ~100% (all GPUs working in parallel)
- **Speedup vs baseline**: ~N× (8× on 8 GPUs)

**With prefetching**:
- **Overlap factor**: ~50-70% of communication hidden by computation
- **Additional speedup**: 2-3× over non-prefetching parallel
- **Total speedup vs baseline**: ~16-24× on 8 GPUs

### When Distributed Config May Not Help

1. **Communication-bound regimes**: When network bandwidth << orthogonalization time
2. **Small models**: Overhead of distributed logic exceeds single-GPU time
3. **Few parameters**: Not enough parallelism to saturate all GPUs

**Rule of thumb**: Use distributed config when:
- Model has 1B+ parameters OR
- Using 4+ GPUs for data/model parallelism

## Future Extensions

1. **Hierarchical gather**: Gather within node first, then across nodes
2. **Compression**: Compress parameters during gather/redistribute
3. **Mixed precision**: FP16/BF16 gathering with FP32 orthogonalization
4. **Parameter grouping**: Group small parameters to amortize communication overhead
5. **Dynamic assignment**: Load-balance based on parameter sizes and GPU speeds

---

**Document Status**: Living document, updated as implementation evolves.
**Last Updated**: 2025-10-15
