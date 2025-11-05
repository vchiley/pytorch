# Practical Implementation Guide: Prefetching and Async GPU Parallelism

This document provides concrete implementation details for the `prefetch_count` and `async_gpu_parallelism` features in the distributed Muon optimizer.

## Table of Contents
1. [Prefetching Implementation](#prefetching-implementation)
2. [Async GPU Parallelism Implementation](#async-gpu-parallelism-implementation)
3. [Combined Implementation](#combined-implementation)
4. [Performance Considerations](#performance-considerations)

---

## Prefetching Implementation

### Core Concept

Prefetching overlaps **communication** (gathering the next parameter's momentum buffer) with **computation** (orthogonalizing the current parameter's momentum buffer). This uses PyTorch's asynchronous collectives.

### Key PyTorch APIs

```python
# Async collective operations
work = dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True)
work.wait()  # Block until operation completes

# Check if operation is complete
if work.is_completed():
    # Safe to use output_tensor
```

### Implementation Structure

```python
def _single_tensor_muon_distributed_prefetch(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    distributed_config: DistributedConfig,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
) -> None:
    assignments = distributed_config.state['assignments']
    rank = distributed_config.state['rank']
    prefetch_count = distributed_config.prefetch_count

    # Prefetch buffer: stores (momentum_buffer_full, work_handle) for each prefetched param
    # Keys are param_idx, values are (tensor, Optional[Work])
    prefetch_buffer: dict[int, tuple[Optional[Tensor], Optional[Any]]] = {}

    for i in range(len(params)):
        param = params[i]
        grad = grads[i]
        momentum_buf = muon_momentum_bufs[i]

        # Step 1: Update momentum buffer locally
        momentum_buf.lerp_(grad, 1 - momentum)

        # Step 2: Get the full momentum buffer (either from prefetch or gather now)
        if i in prefetch_buffer:
            # Use prefetched result
            momentum_buf_full, work = prefetch_buffer.pop(i)
            if work is not None:
                work.wait()  # Ensure gather is complete
        else:
            # No prefetch available, gather synchronously
            momentum_buf_full = distributed_config.gather_fn(
                momentum_buf,
                dst_rank=assignments[i],
                state=distributed_config.state
            )

        # Step 3: Prefetch ahead (while we compute orthogonalization)
        # Prefetch parameters [i+1, i+2, ..., i+prefetch_count]
        for offset in range(1, prefetch_count + 1):
            next_idx = i + offset
            if next_idx >= len(params):
                break  # No more parameters to prefetch
            if next_idx in prefetch_buffer:
                continue  # Already prefetched

            # Start async gather for next parameter
            next_momentum_buf = muon_momentum_bufs[next_idx]

            # Allocate output buffer for gather (if needed)
            # For sharded strategies, we need a full-sized buffer
            if rank == assignments[next_idx]:
                # This rank will receive the full tensor
                # TODO: Determine full shape from state metadata
                output_buf = torch.empty_like(next_momentum_buf)  # Placeholder
            else:
                # Other ranks don't receive data, but still participate
                output_buf = None

            # Call gather_fn with async_op=True (requires modified gather_fn signature)
            # For now, assume gather_fn internally uses async_op
            momentum_buf_full_prefetch, work = _gather_fn_async(
                next_momentum_buf,
                dst_rank=assignments[next_idx],
                state=distributed_config.state
            )

            prefetch_buffer[next_idx] = (momentum_buf_full_prefetch, work)

        # Step 4: Compute orthogonalization (while prefetch happens in background)
        if rank == assignments[i]:
            update_full = _zeropower_via_newtonschulz(
                momentum_buf_full, ns_coefficients, ns_steps, eps
            )
        else:
            update_full = None

        # Step 5: Redistribute the update
        update = distributed_config.redistribute_fn(
            update_full,
            src_rank=assignments[i],
            state=distributed_config.state
        )

        # Step 6: Apply update with nesterov and weight decay
        if nesterov:
            update_to_apply = grad.lerp(momentum_buf, momentum)
            # Need to re-orthogonalize with nesterov? Or orthogonalize nesterov buffer?
            # This needs clarification in the design

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)

    # Cleanup: wait for any remaining prefetches
    for _, (_, work) in prefetch_buffer.items():
        if work is not None:
            work.wait()
```

### Modified gather_fn Signature for Async Support

The `gather_fn` needs to support async operations:

```python
def gather_fn_async(
    momentum_buffer: Tensor,
    dst_rank: int,
    state: dict[str, Any],
    async_op: bool = False
) -> tuple[Optional[Tensor], Optional[Any]]:
    """
    Gather momentum buffer to dst_rank.

    Args:
        momentum_buffer: Local shard/replica
        dst_rank: Rank that will receive full tensor
        state: Metadata including process groups
        async_op: If True, return immediately with work handle

    Returns:
        (full_tensor_or_None, work_handle_or_None)
        - full_tensor: Full tensor on dst_rank, None on others
        - work_handle: PyTorch distributed Work object if async_op=True
    """
    rank = state['rank']

    if 'fsdp_pg' in state:
        # FSDP: sharded strategy
        pg = state['fsdp_pg']
        world_size = dist.get_world_size(pg)

        # Allocate output buffer on dst_rank
        if rank == dst_rank:
            full_shape = _get_full_shape(momentum_buffer, state)
            output = torch.empty(full_shape, dtype=momentum_buffer.dtype, device=momentum_buffer.device)
        else:
            output = None

        # Perform gather
        # PyTorch doesn't have gather to single rank directly, use all_gather then select
        gather_list = [torch.empty_like(momentum_buffer) for _ in range(world_size)]
        work = dist.all_gather(gather_list, momentum_buffer, group=pg, async_op=async_op)

        if async_op:
            # Return work handle; user must call work.wait() before using output
            return (gather_list, work) if rank == dst_rank else (None, work)
        else:
            # Synchronous: concatenate shards
            if rank == dst_rank:
                output = torch.cat(gather_list, dim=0)  # Assuming dim 0 is sharded
                return (output, None)
            else:
                return (None, None)

    elif 'dp_pg' in state:
        # DDP: replicated strategy
        if rank == dst_rank:
            # Already have full tensor locally
            return (momentum_buffer, None)
        else:
            return (None, None)

    else:
        raise ValueError("Unknown parallelism strategy in state")
```

### Challenges with Current Design

**Issue 1:** The `gather_fn` signature in the PROJECT.md doesn't support `async_op` parameter:
```python
# Current signature
gather_fn: Callable[[Tensor, int, dict[str, Any]], Optional[Tensor]]

# Needed signature for prefetch
gather_fn: Callable[[Tensor, int, dict[str, Any], bool], tuple[Optional[Tensor], Optional[Any]]]
```

**Proposed Solution:**
- Add optional `async_op: bool = False` parameter to `gather_fn`
- Return `tuple[Optional[Tensor], Optional[Work]]` instead of just `Optional[Tensor]`
- This maintains backward compatibility (default async_op=False returns (tensor, None))

**Issue 2:** Prefetching requires knowing the full tensor shape before gathering:
- For FSDP, we need to allocate a full-sized output buffer
- This information should be stored in `state` during initialization

---

## Async GPU Parallelism Implementation

### Core Concept

**Async GPU parallelism** means each rank processes **its assigned parameters** independently, without waiting for other ranks to finish their assigned parameters.

**Key Insight:** Without async parallelism:
- Rank 0 processes param 0 (assigned to rank 0)
- Rank 1 waits idle while rank 0 works on param 0
- Then rank 1 processes param 1 (assigned to rank 1)
- Rank 0 waits idle while rank 1 works on param 1
- This serializes the work!

With async parallelism:
- Rank 0 processes param 0, 4, 8, ... (all assigned to rank 0) in parallel
- Rank 1 processes param 1, 5, 9, ... (all assigned to rank 1) in parallel
- Both ranks work simultaneously on different parameters

### Implementation Structure

```python
def _single_tensor_muon_distributed_async(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    distributed_config: DistributedConfig,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
) -> None:
    assignments = distributed_config.state['assignments']
    rank = distributed_config.state['rank']

    # Step 1: Partition parameters by assigned rank
    # This is the key: each rank only processes its assigned parameters
    my_params_indices = [i for i, assigned_rank in assignments.items() if assigned_rank == rank]

    # Step 2: Process my assigned parameters
    # All ranks do this in parallel without coordination
    for i in my_params_indices:
        param = params[i]
        grad = grads[i]
        momentum_buf = muon_momentum_bufs[i]

        # Update momentum buffer
        momentum_buf.lerp_(grad, 1 - momentum)

        # Gather full momentum buffer (I'm the dst_rank, so I get the full tensor)
        momentum_buf_full = distributed_config.gather_fn(
            momentum_buf,
            dst_rank=rank,  # Always me since this is my assigned param
            state=distributed_config.state
        )

        # Orthogonalize (only I do this)
        assert momentum_buf_full is not None, "Should have full tensor on assigned rank"
        update_full = _zeropower_via_newtonschulz(
            momentum_buf_full, ns_coefficients, ns_steps, eps
        )

        # Redistribute to all ranks
        update = distributed_config.redistribute_fn(
            update_full,
            src_rank=rank,  # I'm the source
            state=distributed_config.state
        )

        # Apply update
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)

    # Step 3: All ranks must synchronize before continuing
    # This ensures all parameters have been updated before the next training step
    if 'world_pg' in distributed_config.state:
        dist.barrier(distributed_config.state['world_pg'])
```

### Why This Works

**Without async_gpu_parallelism** (synchronous mode):
```python
for i in range(len(params)):
    # All ranks participate in gather
    gather_fn(...)  # <-- Synchronization point, all ranks wait here

    # Only assigned rank computes
    if rank == assignments[i]:
        orthogonalize(...)  # <-- Other ranks idle here

    # All ranks participate in redistribute
    redistribute_fn(...)  # <-- Synchronization point, all ranks wait here
```
**Result:** Parameters processed one-at-a-time globally. Total time = sum of all parameter times.

**With async_gpu_parallelism:**
```python
# Each rank processes only its assigned parameters
for i in my_assigned_params:
    gather_fn(...)
    orthogonalize(...)
    redistribute_fn(...)

# Barrier at end to sync before next training step
barrier()
```
**Result:** Parameters processed in parallel. Total time ≈ max time across ranks (assuming balanced assignment).

### Challenge: Ordering and Correctness

**Problem:** If rank 0 processes param 0 first, and rank 1 processes param 1 first, when rank 1 calls `gather_fn` for param 1, rank 0 might not have updated its shard of param 1 yet (since rank 0 is still working on param 0).

**Solution:** The gather/redistribute operations must use **the momentum buffer**, not the parameter itself:
- Each rank updates its **local momentum buffer shard** synchronously in Step 1
- Then each rank asynchronously gathers/orthogonalizes/redistributes **the momentum buffer** for its assigned params
- The parameter update happens locally after redistribute, so ordering doesn't matter

**Critical:** All ranks must call `momentum_buf.lerp_(grad, 1 - momentum)` **before** any async processing:
```python
# Step 0: ALL ranks update ALL their local momentum buffer shards (synchronous)
for i in range(len(params)):
    momentum_buf = muon_momentum_bufs[i]
    grad = grads[i]
    momentum_buf.lerp_(grad, 1 - momentum)

# Step 1: Each rank processes its assigned parameters (async)
for i in my_assigned_params:
    # Now safe to gather because all ranks have updated their local shards
    momentum_buf_full = gather_fn(momentum_buf, ...)
    # ... rest of processing
```

---

## Combined Implementation

Combining both prefetching and async GPU parallelism:

```python
def _single_tensor_muon_distributed_full(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    distributed_config: DistributedConfig,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
) -> None:
    assignments = distributed_config.state['assignments']
    rank = distributed_config.state['rank']
    prefetch_count = distributed_config.prefetch_count
    async_gpu = distributed_config.async_gpu_parallelism

    # Step 0: ALL ranks update ALL their local momentum buffers
    # This must be synchronous to ensure consistency
    for i in range(len(params)):
        momentum_buf = muon_momentum_bufs[i]
        grad = grads[i]
        momentum_buf.lerp_(grad, 1 - momentum)

    # Step 1: Determine which parameters this rank will process
    if async_gpu:
        # Process only assigned parameters
        param_indices_to_process = [i for i, r in assignments.items() if r == rank]
    else:
        # Process all parameters in order (synchronous mode)
        param_indices_to_process = list(range(len(params)))

    # Step 2: Process parameters with prefetching
    prefetch_buffer: dict[int, tuple[Optional[Tensor], Optional[Any]]] = {}

    for idx, i in enumerate(param_indices_to_process):
        param = params[i]
        momentum_buf = muon_momentum_bufs[i]

        # Get full momentum buffer (from prefetch or gather now)
        if i in prefetch_buffer:
            momentum_buf_full, work = prefetch_buffer.pop(i)
            if work is not None:
                work.wait()
        else:
            if async_gpu:
                # In async mode, I'm always the dst_rank for my assigned params
                momentum_buf_full = distributed_config.gather_fn(
                    momentum_buf, dst_rank=rank, state=distributed_config.state
                )
            else:
                # In sync mode, use the assignment
                momentum_buf_full = distributed_config.gather_fn(
                    momentum_buf, dst_rank=assignments[i], state=distributed_config.state
                )

        # Prefetch ahead
        if prefetch_count > 0:
            for offset in range(1, prefetch_count + 1):
                next_list_idx = idx + offset
                if next_list_idx >= len(param_indices_to_process):
                    break
                next_param_idx = param_indices_to_process[next_list_idx]

                if next_param_idx not in prefetch_buffer:
                    next_momentum_buf = muon_momentum_bufs[next_param_idx]

                    # Start async gather
                    if async_gpu:
                        next_dst_rank = rank
                    else:
                        next_dst_rank = assignments[next_param_idx]

                    buf_full_prefetch, work = _gather_fn_async(
                        next_momentum_buf,
                        dst_rank=next_dst_rank,
                        state=distributed_config.state,
                        async_op=True
                    )
                    prefetch_buffer[next_param_idx] = (buf_full_prefetch, work)

        # Orthogonalize
        if async_gpu or rank == assignments[i]:
            assert momentum_buf_full is not None
            update_full = _zeropower_via_newtonschulz(
                momentum_buf_full, ns_coefficients, ns_steps, eps
            )
        else:
            update_full = None

        # Redistribute
        if async_gpu:
            update = distributed_config.redistribute_fn(
                update_full, src_rank=rank, state=distributed_config.state
            )
        else:
            update = distributed_config.redistribute_fn(
                update_full, src_rank=assignments[i], state=distributed_config.state
            )

        # Apply update
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)

    # Step 3: Synchronize all ranks before next training step
    if async_gpu and 'world_pg' in distributed_config.state:
        dist.barrier(distributed_config.state['world_pg'])

    # Cleanup remaining prefetches
    for _, (_, work) in prefetch_buffer.items():
        if work is not None:
            work.wait()
```

---

## Performance Considerations

### Memory Usage

**Prefetching:**
- Each prefetched parameter requires a full-sized buffer
- Memory overhead = `prefetch_count × largest_parameter_size`
- For a 7B model, largest layers ~500M params, prefetch_count=2 → ~4GB extra memory

**Async GPU Parallelism:**
- Minimal memory overhead
- No additional buffers needed

**Recommendation:** Start with `prefetch_count=1`, `async_gpu_parallelism=True`

### Communication Patterns

**Gather Operations:**
- FSDP: `all_gather` - all ranks send their shards to all ranks
- TP: `all_gather` along TP dimension
- DDP: No gather needed, tensor already replicated

**Redistribute Operations:**
- FSDP: `scatter` or `reduce_scatter` - split full tensor back to shards
- TP: `scatter` along TP dimension
- DDP: `broadcast` from src_rank to all replicas

### Optimization Tips

1. **Profile First:** Use PyTorch profiler to measure:
   - `orthogonalize_time` per parameter
   - `gather_time` per parameter
   - `redistribute_time` per parameter

2. **Tune Prefetch Count:**
   - If `gather_time > orthogonalize_time`: Increase prefetch_count to 2
   - If `gather_time < orthogonalize_time`: Keep prefetch_count at 1
   - If `gather_time << orthogonalize_time`: Set prefetch_count to 0 (disable)

3. **Assignment Strategy:**
   - Round-robin by default: `param_idx % world_size`
   - Load-balanced: Assign by parameter size to balance compute time
   ```python
   def assign_fn_balanced(params, state):
       world_size = state['world_size']
       # Sort params by size
       param_sizes = [(i, p.numel()) for i, p in enumerate(params)]
       param_sizes.sort(key=lambda x: x[1], reverse=True)

       # Greedy assignment: assign largest param to least-loaded rank
       rank_loads = [0] * world_size
       assignments = {}
       for idx, size in param_sizes:
           min_rank = min(range(world_size), key=lambda r: rank_loads[r])
           assignments[idx] = min_rank
           rank_loads[min_rank] += size
       return assignments
   ```

4. **Debugging:**
   - Disable both features: `prefetch_count=0`, `async_gpu_parallelism=False`
   - Process parameters one-at-a-time globally, easier to trace
   - Add logging at each step to identify issues

---

## Example Usage

```python
from torch.optim import Muon
from torch.optim.muon import create_processgroup_config

# Create FSDP model
model = FSDP(model, ...)

# Default: prefetch=1, async=True (best performance)
optimizer = Muon(
    model.parameters(),
    lr=0.02,
    distributed_config=create_processgroup_config(
        fsdp_pg=model.process_group,
        prefetch_count=1,  # Overlap next gather with current compute
        async_gpu_parallelism=True  # Each rank works independently
    )
)

# Debug mode: disable optimizations
optimizer_debug = Muon(
    model.parameters(),
    lr=0.02,
    distributed_config=create_processgroup_config(
        fsdp_pg=model.process_group,
        prefetch_count=0,  # No prefetch
        async_gpu_parallelism=False  # Synchronous processing
    )
)

# High-memory mode: aggressive prefetching
optimizer_fast = Muon(
    model.parameters(),
    lr=0.02,
    distributed_config=create_processgroup_config(
        fsdp_pg=model.process_group,
        prefetch_count=2,  # Prefetch 2 ahead
        async_gpu_parallelism=True
    )
)
```

---

## Summary

**Prefetching:**
- Overlaps communication (gather) with computation (orthogonalize)
- Requires async collective operations
- Memory overhead: `prefetch_count × param_size`
- Benefit: 20-40% speedup when communication-bound

**Async GPU Parallelism:**
- Each rank processes only its assigned parameters
- All ranks work in parallel
- Minimal memory overhead
- Benefit: Near-linear speedup with number of GPUs (if balanced)

**Combined:**
- Best performance: both enabled (defaults)
- Start with `prefetch_count=1`, `async_gpu_parallelism=True`
- Profile and tune based on your model's characteristics
