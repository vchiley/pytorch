# Muon Distributed Training

Design document for adding distributed training support to the Muon optimizer.

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

### Challenge 1: Correctness

Muon's orthogonalization step (via `_zeropower_via_newtonschulz` in `_muon.py`) requires operating on the **full tensor** to maintain mathematical correctness. However, distributed training (FSDP, HSDP, Tensor Parallel, Expert Parallel, Context Parallel, Pipeline Parallel, etc.) shards parameters and their updates across devices. Orthogonalizing partial tensors separately produces **incorrect results**.

**Required:** In the distributed setting, the full `momentum_buffer` must be gathered from shards, orthogonalized as a full tensor, then the resulting update must be redistributed back to shards.

### Challenge 2: Efficiency

In a naive implementation, `_zeropower_via_newtonschulz` would operate on the full `momentum_buffer` **on every GPU**, creating redundant computation.

**Required:** Each parameter should be assigned to exactly one rank. That rank gathers the full tensor, performs orthogonalization once, then redistributes the result.

## Solution Overview

Add a `distributed_config: DistributedConfig` parameter to `Muon.__init__()` that enables distributed orthogonalization with **zero redundancy** - each parameter is processed exactly once across all devices.

`DistributedConfig` has **three user-defined functions**:
1. **assign_fn**: Maps each param_idx to a rank that will orthogonalize that update
2. **gather_fn**: Gathers the full momentum buffer from shards (on assigned rank)
3. **redistribute_fn**: Redistributes (scatters) the orthogonalized update back to shards and/or replicas

Additionally, `DistributedConfig` should have an attached `state` that holds all metadata needed by the three functions above including but not limited to: Process groups (e.g., FSDP, TP, DP process groups), Device meshes, Tensor shapes/dtypes, Communication patterns, Any other distributed training metadata, etc.

Conceptually, the pseudocode is as follows:
```python
state = distributed_config.state
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
**Prefetching**: When orthogonalizing an update for parameter N, the momentum buffer for parameter N+1 can be prefetched via `gather_fn` to overlap communication with computation.
- `prefetch_count`: Controls how many parameters' momentum buffers to prefetch ahead
- `prefetch_count=0`: Disables prefetching (sequential communication and computation)
- `prefetch_count=1`: Default behavior (prefetch next parameter's momentum buffer)

**Independent Asynchronous Processing**: Each GPU processes its assigned parameters in parallel.
- `async_gpu_parallelism`: Enable/disable parallel processing (default: `True`)

### Notes on Parallelism:
- **Sharded**: Parallelism strategies which shard params (such as TP, FSDP, and HSDP's shard dimension) must gather momentum buffer and scatter updates using `gather_fn` and `redistribute_fn` respectively.
- **Replicated**: Parallelism strategies which replicate params (such as DDP, CP, and HSDP's replicate dimension), do not need to gather momentum buffer since the full replica is already on each GPU, but must still broadcast the orthogonalized update in `redistribute_fn`.
- **Independent**: Parallelism strategies which hold independent parameter sets (such as EP, PP), do not need to gather momentum buffer nor distribute updates since the params are independent, but their process groups should still be passed to `assign_fn`'s state to guarantee rank assignment for deduplication of redundant work is done correctly.

### `DistributedConfig` API

```python
from dataclasses import dataclass
from typing import Any, Callable
from torch import Tensor

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

    gather_fn: Callable[[Tensor, int, dict[str, Any]], Tensor | None]
    # For parallelism strategies that need it, gathers the full momentum buffer from shards for orthogonalization
    # Signature: gather_fn(momentum_buffer, dst_rank, state) -> momentum_buffer_full or None
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
    # If False: Sequential processing (easier debugging)
    # Recommended: False during initial development/debugging, True for production

    prefetch_count: int = 1
    # Number of tensors to prefetch ahead while processing current parameter
    # 0: Disabled (sequential communication and computation)
    # 1-2: Recommended (overlaps communication with computation)
    # 3+: Higher memory usage, diminishing returns
    # Uses async_op=True in distributed collectives
    # Expected speedup: ~2x for models with many parameters
```

### Quality of Life (QoL) Features
Parallelism setups are often bespoke. `DistributedConfig` enables users to define `assign_fn`, `gather_fn`, and `redistribute_fn` for their specific implementation of distributed training.
For convenience, we provide helper functions to generate `DistributedConfig` for common distributed training setups:

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

- **Flexibility**: Works with any parallelism configuration thru `DistributedConfig`.
- **Simple**: User only needs to specify 3 functions `assign_fn`, `gather_fn`, and `redistribute_fn`.
- **Quality of Life**: can just use `create_processgroup_config()`, `create_devicemesh_config()`, or `create_dtensor_config()`.
- **Performant**: `prefetch_count` and `async_gpu_parallelism` enable faster parallel computation.
- **Debug Mode**: disable `prefetch_count` and `async_gpu_parallelism` for slower but more debuggable computation; to disable set `prefetch_count=0` and `async_gpu_parallelism=False`.
