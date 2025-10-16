# Muon Distributed Training

This file provides guidance for modifying the Muon optimizer to work with distributed training setups.

## Problem

Muon's orthogonalization step (`_zeropower_via_newtonschulz` in `_muon.py`) requires operating on the full matrix structure to maintain mathematical correctness. However, distributed training (FSDP, HSDP, Tensor Parallel, Expert Parallel, Context Parallel, Pipeline Parallel, etc.) shards parameters and their updates across devices. Orthogonalizing partial matrices separately produces incorrect results.

**Why gather updates instead of parameters?** More efficient - only the update (momentum buffer) needs orthogonalization and redistribution. Parameters stay sharded; we just apply the orthogonalized update to them.

Additionally, synchronous gather/orthogonalize/redistribute operations create bottlenecks where GPUs sit idle while waiting for others to complete their work.

## Solution

Add `distributed_config` parameter to Muon with **three user-defined functions** (the core API):

1. **assign_fn**: Assigns parameters to devices for orthogonalization
2. **gather_fn**: Gathers full update (momentum buffer) to assigned device
3. **redistribute_fn**: Redistributes orthogonalized update back

**Key Design Principle**: These three functions define how distributed orthogonalization works. They receive a flexible `state` dict containing process groups and metadata, enabling support for **any distributed setup** (PyTorch built-in strategies, custom frameworks, or proprietary parallelism).

**For convenience**, `create_auto_config()` provides pre-built implementations for common PyTorch strategies (FSDP, TP, HSDP, EP, CP, PP, and hybrids). For custom distributed setups, users define their own functions.

## When to Use distributed_config

**Required for correct orthogonalization:**
- **Sharded strategies**: FSDP, TP, HSDP, EP, PP, or any hybrid where parameters are sharded across devices
  - Without `distributed_config`, orthogonalization will operate on shards independently (mathematically incorrect)
- **Replicated strategies**: CP (Context Parallel) where parameters are replicated
  - Without `distributed_config`, each GPU will redundantly orthogonalize all updates (inefficient)

**Not needed:**
- Single GPU training

**Optional but recommended:**
- DDP (DistributedDataParallel) - model is fully replicated on each GPU
  - DDP works without `distributed_config` but creates redundant orthogonalization work on each GPU
  - Using `distributed_config` with DDP coordinates work: each GPU orthogonalizes subset of updates, then broadcasts results
  - Similar to CP (Context Parallel) - both replicate parameters but benefit from work coordination

**Warning:** Using Muon in distributed training (e.g., FSDP, TP) without `distributed_config` will produce incorrect results. Users must configure `distributed_config` when using any sharded parallelism strategy.

**Debug Mode:**
- Set `async_gpu_parallelism=False` in `DistributedConfig` to enable sequential debugging mode
- In debug mode, GPUs process parameters sequentially (GPU 0 completes all its params, then GPU 1, etc.)
- This makes debugging much easier: clear execution order, no interleaved output, easy to isolate issues
- Default is `True` for parallel execution (optimal performance)

## API

```python
from dataclasses import dataclass
from typing import Any, Callable
from torch import Tensor

@dataclass
class DistributedConfig:
    assign_fn: Callable[[list[Tensor], dict[str, Any]], dict[int, list[int]]]
    gather_fn: Callable[[Tensor, int, dict[str, Any]], Tensor]
    redistribute_fn: Callable[[Tensor, int, dict[str, Any]], Tensor]
    state: dict[str, Any]  # Process groups, device meshes, metadata
    async_gpu_parallelism: bool = True  # False for sequential debugging
    prefetch_count: int = 1  # Updates to prefetch (0=disabled, 1-3 recommended)

class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        # ... other params ...
        distributed_config: Optional[DistributedConfig] = None,
    ):
```

## Usage

### Option 1: Auto-Configuration (Common PyTorch Strategies)

For standard PyTorch parallelism strategies, use `create_auto_config()`:

```python
from torch.optim.muon import Muon, create_auto_config

# FSDP only
config = create_auto_config(fsdp_process_group=fsdp_pg)

# TP only
config = create_auto_config(tp_process_group=tp_pg, tp_shard_dim=1)

# TP + FSDP hybrid
config = create_auto_config(
    tp_process_group=tp_pg,
    fsdp_process_group=fsdp_pg,
    tp_shard_dim=1
)

# HSDP (shard + replicate)
config = create_auto_config(
    shard_process_group=shard_pg,
    replicate_process_group=replicate_pg
)

# TP + HSDP (three-way parallelism)
config = create_auto_config(
    tp_process_group=tp_pg,
    shard_process_group=shard_pg,
    replicate_process_group=replicate_pg,
    tp_shard_dim=1
)

# Expert Parallel only
config = create_auto_config(ep_process_group=ep_pg, ep_shard_dim=0)

# Context Parallel only
config = create_auto_config(cp_process_group=cp_pg)

# Pipeline Parallel only
config = create_auto_config(
    pp_process_group=pp_pg,
    pp_stage_id=stage_id,
    pp_num_stages=num_stages
)

# Full hybrid: PP + TP + HSDP (4D parallelism)
config = create_auto_config(
    pp_process_group=pp_pg,
    tp_process_group=tp_pg,
    shard_process_group=shard_pg,
    replicate_process_group=replicate_pg,
    pp_stage_id=stage_id,
    pp_num_stages=num_stages,
    tp_shard_dim=1
)

optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)
```

### Option 2: Custom Functions (Custom Distributed Setups)

For custom distributed frameworks, proprietary parallelism, or non-PyTorch setups, define your own functions:

```python
from torch.optim.muon import Muon, DistributedConfig

def my_assign_fn(params: list[Tensor], state: dict[str, Any]) -> dict[int, list[int]]:
    """Assign parameters to ranks for orthogonalization.

    Returns: {rank: [param_indices]} mapping
    Example: {0: [0, 2, 4], 1: [1, 3, 5]} for 2 ranks, 6 params
    """
    # Your custom logic here
    # Could use state["custom_process_group"] or other metadata
    rank = state["rank"]
    world_size = state["world_size"]
    return {r: list(range(r, len(params), world_size)) for r in range(world_size)}

def my_gather_fn(update: Tensor, rank: int, state: dict[str, Any]) -> Tensor:
    """Gather full update (momentum buffer) to the assigned rank.

    Args:
        update: The sharded update (momentum buffer) on this rank
        rank: The rank that will orthogonalize this update
        state: User-provided metadata (process groups, etc.)

    Returns: Full update (only valid on assigned rank)
    """
    # Your custom gather logic here
    # Example: all_gather across custom_process_group
    custom_pg = state["custom_process_group"]
    return my_custom_all_gather(update, group=custom_pg)

def my_redistribute_fn(ortho_update: Tensor, rank: int, state: dict[str, Any]) -> Tensor:
    """Redistribute orthogonalized update back to shards.

    Args:
        ortho_update: The orthogonalized full update
        rank: The rank that orthogonalized this update
        state: User-provided metadata

    Returns: This rank's shard of the update
    """
    # Your custom redistribute logic here
    custom_pg = state["custom_process_group"]
    return my_custom_scatter(ortho_update, group=custom_pg)

# Create config with custom functions
config = DistributedConfig(
    assign_fn=my_assign_fn,
    gather_fn=my_gather_fn,
    redistribute_fn=my_redistribute_fn,
    state={
        "rank": my_framework.get_rank(),
        "world_size": my_framework.get_world_size(),
        "custom_process_group": my_framework.get_process_group(),
        # Any other metadata your functions need
    }
)

optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)
```

**When to use custom functions:**
- Using a non-PyTorch distributed framework (e.g., Horovod, custom MPI setup)
- Proprietary parallelism strategies
- Hybrid setups not covered by `create_auto_config()`
- Need fine-grained control over gather/redistribute behavior
- Custom parameter assignment logic (e.g., based on parameter sizes)

## Implementation

Modify the Muon optimizer to include (see DESIGN.md for detailed implementation stages):

1. **`DistributedConfig` dataclass** in `_muon.py` - Holds assign_fn, gather_fn, redistribute_fn, and state dict

2. **`distributed_config` parameter** in `Muon.__init__` - Optional distributed configuration

3. **Unified `_single_tensor_muon()` function** that handles both single-GPU and distributed cases:
   - Updates momentum on sharded parameters (always)
   - **If `distributed_config` exists**: Uses `gather_fn` to get full update → orthogonalizes update → uses `redistribute_fn` to scatter back → applies orthogonalized update to sharded params
   - **If `distributed_config` is None**: Orthogonalizes update directly and applies to local parameter (single-GPU path)
   - Respects `async_gpu_parallelism` flag for parallel vs sequential execution when distributed

   **Why orthogonalize updates instead of parameters?** More efficient - only the update (momentum buffer) needs orthogonalization and redistribution. Parameters stay sharded; we apply the orthogonalized update to them in-place.

4. **Modified `step()` method** - Passes `distributed_config` to unified function

**Key design**: ONE unified implementation with distributed operations gated behind `distributed_config`. No separate function variants needed.

5. **`create_auto_config()` factory function** - Single function that handles all parallelism combinations via runtime process group detection:
   - Supports FSDP, TP, HSDP (shard + replicate), EP (Expert Parallel), CP (Context Parallel), and PP
   - Automatically detects which process groups are provided
   - Builds correct gather/redistribute pipeline
   - Handles many possible combinations with one function

6. **Operation-level async (prefetching)**:
   - Controlled by `prefetch_count` in `DistributedConfig` (default: 1, set to 0 to disable)
   - When `prefetch_count > 0`, uses `async_op=True` in distributed collectives to prefetch next parameters
   - Recommended values: 1-3 (higher = more overlap but more memory)
   - Overlaps communication with computation
   - Expected 2-3x speedup for models with many parameters
   - Note: This is independent of `async_gpu_parallelism` (GPU-level parallelism)

## Documentation

- **API details**: Docstrings in `_muon.py`
- **Architecture and design decisions**: See `DESIGN.md`
- **Usage examples**: See `EXAMPLES.md`
- **Testing strategy**: See `TESTING.md`

---

**Document Status**: Living document, updated as implementation evolves.
**Last Updated**: 2025-10-15

## Key Points

- **Backward compatible**: `distributed_config=None` uses original implementation
- **Flexible**: State dict holds any process groups/metadata needed
- **Efficient**: Each parameter orthogonalized exactly once
- **Works with any parallelism strategy**: FSDP, TP, HSDP, EP, CP, PP, or hybrids
- **Auto-configuration**: One function handles many parallelism combinations
- **Pipeline Parallelism**: Each stage runs optimizer independently, no cross-stage coordination needed
- **GPU-level parallelism**: `async_gpu_parallelism` controls parallel vs sequential GPU execution
- **Operation-level async**: `prefetch_count` controls communication overlap (default: 1, set to 0 to disable)
- **Tunable prefetch**: Values 1-3 recommended; higher values = more overlap but more memory
- **Debug mode**: Set `async_gpu_parallelism=False` and `prefetch_count=0` for simplest debugging
- **Correctness guaranteed**: PyTorch-tested gather/redistribute logic
