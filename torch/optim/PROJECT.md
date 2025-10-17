# Muon Distributed Training

Modifying the Muon optimizer to work with distributed training setups.

## Problem

### Background

The Muon optimizer performs this update step:
```python
muon_step(params, grads, momentum_buffers, lr, beta, ns_steps):
    for param, grad, momentum_buffer in zip(params, grads, momentum_buffers):
        momentum_buffer = beta * momentum_buffer + grad
        update = _zeropower_via_newtonschulz(momentum_buffer, ns_steps)
        param = param - lr * update
```

### Challenge 1: Correctness

Muon's orthogonalization step (`_zeropower_via_newtonschulz` in `_muon.py`) requires operating on the **full tensor** to maintain mathematical correctness. However, distributed training (FSDP, HSDP, Tensor Parallel, Expert Parallel, Context Parallel, Pipeline Parallel, etc.) shards parameters and their updates across devices. Orthogonalizing partial tensors separately produces **incorrect results**.

**Required:** In the distributed setting, the full `momentum_buffer` must be gathered from shards, orthogonalized as a full tensor, then the resulting update must be redistributed back to shards.

### Challenge 2: Efficiency

Without optimization, `_zeropower_via_newtonschulz` would operate on the full `momentum_buffer` **on every GPU**, creating redundant computation.

**Required:** Each parameter should be assigned to exactly one rank. That rank gathers the full tensor, performs orthogonalization once, then redistributes the result.

## Solution Overview

Add a `distributed_config: DistributedConfig` parameter to `Muon.__init__()` that enables distributed orthogonalization with **zero redundancy** - each parameter is processed exactly once across all devices.

`DistributedConfig` has **three user-defined functions**:
1. **assign_fn**: Maps each param_idx to a rank that will orthogonalization that update
2. **gather_fn**: Gathers the full momentum buffer from shards (on assigned rank)
3. **redistribute_fn**: Redistributes (scatters) the orthogonalized update back to shards and/or replica's

Additionally, `DistributedConfig` should have an attached `state` that holds all metadata needed by the three functions above including but not limited to: Process groups (e.g., FSDP, TP, DP process groups), Device meshes, Tensor shapes/dtypes, Communication patterns, Any other distributed training metadata, etc.

Conceptually, the pseudocode is as follows:
```python
assignments = assign_fn(params)  # done once in `class Muon`'s `__init__`
muon_step(params, grads, momentum_buffers, assignments, lr, beta, ns_steps):
    for param_idx, (param, grad, momentum_buffer) in enumerate(zip(params, grads, momentum_buffers)):
        momentum_buffer = beta * momentum_buffer + grad

        momentum_buffer_full = gather_fn(momentum_buffer, dst=assignments[param_idx])
        if rank == assignments[param_idx]:
            update_full = _zeropower_via_newtonschulz(momentum_buffer_full, ns_steps)
        update = redistribute_fn(momentum_buffer.clone().detach(), update_full, src=assignments[param_idx])

        param = param - lr * update
```

### Performance / Acceleration
**Prefetching**: When `_zeropower_via_newtonschulz` is orthogonalizing an update, it can prefetch gathering (via `gather_fn`) the next tensor. Prefetch can accelerate the above. Use `prefetch_count` to set up to how many Tensors the operation should prefetch. Setting `prefetch_count=0` disables this feature, and `prefetch_count: int = 1` should be the default.
**Independent Asynchronous Processing**: To remove redundant computation, each GPU is assigned a set of param updates to orthogonalize via `_zeropower_via_newtonschulz`; the orthogonalization process is independent on each GPU and can done in parallel; allow this feature to be enabled or disabled via a `async_gpu_parallelism: bool = True` flag which defaults to `True`.
`DistributedConfig` should use `prefetch_count` and `async_gpu_parallelism`

### Quality of Life (QoL) Features
Parallelism setups are often bespoke. `DistributedConfig` enables users to define `assign_fn`, `gather_fn`, and `redistribute_fn` for their specific implementation of distributed training.
For convenience, we should implement a `create_auto_config()` functions which generates `DistributedConfig` for use with generic PyTorch process groups (defined per parallelism strategy); takes as input all the different parallelism strategies (FSDP, HSDP, TP, DDP, EP, CP, etc) to generate a `DistributedConfig`.
Additionally, we should implement a `create_auto_devicemesh_config()` functions which generates `DistributedConfig` for use with generic PyTorch device meshes (defined per parallelism strategy).
Additionally, we should implement a `create_dtensor_config()` functions which generates `DistributedConfig` for use with training setups which use PyTorch DTensor

### Notes on Parallelism:
- **Sharded**: Parallelism strategies which shard params (such as TP, FSDP, and one dim of HSDP (ie the non-DDP dim)) must gather momentum buffer and scatter updates using `gather_fn` and `redistribute_fn` respectively.
- **Replicated**: Parallelism strategies which replicate params (such as DDP, CP, and the other dim of HSDP (ie the DDP dim)), do not need to gather momentum buffer since the full replica is already on each GPU, but must still broadcast the orthogonalized update in `redistribute_fn`.
- **Independent**: Parallelism strategies which hold independent parameter sets (such as EP, PP), do not need to gather momentum buffer nor distribute updates since the params are independent, but their process groups should still participate in `assign_fn` to guarantee rank assignment for deduplication of redundant work is done correctly.

## Design Summary / Principles

- **Flexibility**: Works with any parallelism configuration if user specifies 3 functions `DistributedConfig`.
- **Simple**: User only needs to specify 3 functions `assign_fn`, `gather_fn`, and `redistribute_fn`.
- **Quality of Life**: can just use `create_auto_config()`, `create_auto_devicemesh_config()`, or `create_dtensor_config()`.
- **Performant**: `prefetch_count` and `async_gpu_parallelism` enable faster parallel computation.
- **Debug Mode**: disable `prefetch_count` and `async_gpu_parallelism` for slower but more debuggable computation; to disable set `prefetch_count=0` and `async_gpu_parallelism=False`.
