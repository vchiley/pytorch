# Muon Distributed Training: Detailed Design

This document provides the complete API specification and implementation details for distributed Muon training.

**Related Documents**:
- `PROJECT.md` - High-level overview, problem statement, and solution approach
- `EXAMPLES.md` - Concrete usage patterns and examples
- `TESTING.md` - Testing strategy and validation approach

## Module Structure

```
torch.optim.muon (public API)
├── DistributedConfig        # Configuration dataclass
├── create_auto_config()     # Convenience function for process group-based setups
├── create_dtensor_config()  # Convenience function for DTensor-based setups
└── Muon                     # Optimizer class

torch.optim._muon (private implementation)
└── _zeropower_via_newtonschulz()  # Newton-Schulz orthogonalization
```

**Import examples**:
```python
# Public API (recommended for users)
from torch.optim import Muon
from torch.optim.muon import DistributedConfig, create_auto_config, create_dtensor_config

# For DTensor-based distributed training
from torch.optim.muon import create_dtensor_config

# For process group-based distributed training
from torch.optim.muon import create_auto_config

# Private API (only needed for testing internal functions)
from torch.optim._muon import _zeropower_via_newtonschulz
```

## API Specification

```python
from dataclasses import dataclass
from typing import Any, Callable, Optional
from torch import Tensor
from torch.optim import Optimizer
from torch.distributed.tensor import DTensor

@dataclass
class DistributedConfig:
    """Configuration for distributed Muon training.

    The three functions (assign_fn, gather_fn, redistribute_fn) define how
    orthogonalization works in a distributed setting. Together with the state
    dict, they enable support for any distributed training strategy.
    """

    assign_fn: Callable[[list[Tensor], dict[str, Any]], dict[int, list[int]]]
    # Maps rank → list of parameter indices that rank will orthogonalize
    # Called once during __init__
    # Args: (params, state) -> {rank: [param_idx_1, param_idx_2, ...]}

    gather_fn: Callable[[Tensor, int, dict[str, Any]], Tensor]
    # Gathers the full momentum buffer from shards for orthogonalization
    # Called only on ranks assigned to process this parameter
    # Args: (momentum_buffer_shard, current_rank, state) -> momentum_buffer (full)

    redistribute_fn: Callable[[Tensor, int, dict[str, Any]], Tensor]
    # Redistributes the orthogonalized update back to shards
    # Called only on ranks that performed orthogonalization
    # Args: (update, current_rank, state) -> update_shard

    state: dict[str, Any]
    # Holds all metadata needed by the three functions above:
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
    # Number of parameters to prefetch while processing current parameter
    # 0: Disabled (sequential communication and computation)
    # 1-2: Recommended (overlaps communication with computation)
    # 3+: Higher memory usage, diminishing returns
    # Uses async_op=True in distributed collectives
    # Expected speedup: ~2x for models with many parameters


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = None,
        adamw_betas: tuple[float, float] = (0.95, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
        distributed_config: Optional[DistributedConfig] = None,
    ):
        """
        Args:
            ...existing args...
            distributed_config: Configuration for distributed training.
                If None, uses original single-device implementation (backward compatible).
                If provided, enables distributed orthogonalization with zero redundancy.
        """

def create_auto_config(
    fsdp_process_group=None,
    tp_process_group=None,
    ddp_process_group=None,
    ep_process_group=None,
    cp_process_group=None,
    tp_dim_per_param=None,
    expert_assignments=None,
    async_gpu_parallelism=True,
    prefetch_count=1,
) -> DistributedConfig:
    """
    Convenience function to automatically create DistributedConfig for common
    parallelism strategies.

    This function handles any combination of PyTorch's built-in distributed
    strategies (FSDP, TP, DDP, EP, CP) by inspecting the provided process
    groups and creating appropriate assign_fn, gather_fn, and redistribute_fn
    implementations.

    Args:
        fsdp_process_group: FSDP process group (sharding)
        tp_process_group: Tensor parallel process group
        ddp_process_group: DDP process group (replication)
        ep_process_group: Expert parallel process group
        cp_process_group: Context parallel process group
        tp_dim_per_param: dict[param_idx -> tp_dim] or single int for all params
            Maps each parameter to its TP sharding dimension. Different layers
            can have different TP dimensions (e.g., column-parallel vs row-parallel).
            If int, uses same dimension for all parameters.
        expert_assignments: dict[param_idx -> expert_id] for Expert Parallel
        async_gpu_parallelism: Enable async GPU parallelism (default: True)
        prefetch_count: Number of parameters to prefetch (default: 1)

    Returns:
        DistributedConfig configured for the provided parallelism strategies

    Examples:
        # FSDP only
        config = create_auto_config(fsdp_process_group=fsdp_pg)

        # FSDP + TP with different dimensions per layer
        config = create_auto_config(
            fsdp_process_group=fsdp_pg,
            tp_process_group=tp_pg,
            tp_dim_per_param={0: 0, 1: 1, 2: 0, 3: 1},  # Column/row parallel
        )

        # HSDP (FSDP + DDP)
        config = create_auto_config(
            fsdp_process_group=fsdp_pg,
            ddp_process_group=ddp_pg,
        )

        # 3D parallelism
        config = create_auto_config(
            fsdp_process_group=fsdp_pg,
            tp_process_group=tp_pg,
            ddp_process_group=ddp_pg,
            tp_dim_per_param=tp_dims,
        )
    """

def create_dtensor_config(
    async_gpu_parallelism=True,
    prefetch_count=1,
) -> DistributedConfig:
    """
    Convenience function to create DistributedConfig for DTensor-based distributed
    training setups.

    This function is designed for models where parameters are torch.distributed.tensor.DTensor
    instances. DTensors already encapsulate distributed metadata (device mesh, placement,
    sharding spec), so the configuration is simpler than process group-based approaches.

    The gather_fn uses DTensor.full_tensor() to materialize the full tensor for
    orthogonalization, and redistribute_fn converts the result back to a DTensor
    with the original placement.

    Args:
        async_gpu_parallelism: Enable async GPU parallelism (default: True)
        prefetch_count: Number of parameters to prefetch (default: 1)

    Returns:
        DistributedConfig configured for DTensor-based distributed training

    Examples:
        # Basic DTensor setup
        from torch.distributed.tensor import distribute_tensor
        from torch.optim.muon import create_dtensor_config

        # Model with DTensor parameters
        config = create_dtensor_config()
        optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

        # With performance tuning
        config = create_dtensor_config(
            async_gpu_parallelism=True,
            prefetch_count=2,
        )
        optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

    Key differences from create_auto_config():
        - No process groups needed: DTensors already contain distributed metadata
        - Simpler gather_fn: Just DTensor.full_tensor() to get the full tensor
        - Automatic: Device mesh and placement extracted from DTensor parameters
        - Flexible: Works with any DTensor placement strategy (Shard, Replicate, etc.)

    Requirements:
        - Model parameters must be DTensor instances
        - All DTensor parameters should use compatible device meshes
        - DTensor placement strategies are detected automatically

    Note on device mesh compatibility:
        - Parameters with the same device mesh work seamlessly
        - Parameters with different device meshes may cause errors during redistribution
        - Validation: all(p.data.device_mesh == params[0].data.device_mesh
                        for p in params if isinstance(p.data, DTensor))
    """
```

## Function Specifications

### assign_fn

**Purpose**: Assign each parameter to exactly one rank for orthogonalization.

**Signature**:
```python
def assign_fn(params: list[Tensor], state: dict[str, Any]) -> dict[int, list[int]]
```

**Arguments**:
- `params`: List of all parameters in the optimizer
- `state`: Metadata dict (process groups, device meshes, etc.)

**Returns**: Dictionary mapping `rank → [param_indices]`

**Requirements**:
- Each parameter index must appear in exactly one rank's list (zero redundancy)
- For independent parameter groups (EP, PP), assign related parameters to the same rank
- Balance load across ranks when possible

**Example**:
```python
# For 4 parameters and 2 ranks
{0: [0, 2], 1: [1, 3]}  # Rank 0 handles params 0,2; Rank 1 handles params 1,3
```

### gather_fn

**Purpose**: Gather the full momentum buffer from shards on the assigned rank.

**Signature**:
```python
def gather_fn(
    momentum_buffer_shard: Tensor,
    current_rank: int,
    state: dict[str, Any]
) -> Tensor
```

**Arguments**:
- `momentum_buffer_shard`: The local shard of the momentum buffer
- `current_rank`: The rank calling this function (from `state['rank']` or similar)
- `state`: Metadata dict

**Returns**: The full, gathered momentum buffer (ready for orthogonalization)

**When called**: Only on ranks assigned to process this parameter (via `assign_fn`)

**Special cases**:
- **FSDP**: Use `all_gather` to collect shards across FSDP process group
- **Replicated (DDP, CP)**: Return `momentum_buffer_shard` as-is (already full)
- **Tensor Parallel**: Gather along TP dimension(s)

### redistribute_fn

**Purpose**: Redistribute the orthogonalized update back to shards.

**Signature**:
```python
def redistribute_fn(
    update: Tensor,
    current_rank: int,
    state: dict[str, Any]
) -> Tensor
```

**Arguments**:
- `update`: The full orthogonalized update tensor
- `current_rank`: The rank calling this function
- `state`: Metadata dict

**Returns**: The local shard of the update

**When called**: Only on ranks that performed orthogonalization

**Special cases**:
- **FSDP**: Scatter or slice to get this rank's shard
- **Replicated (DDP, CP)**: Broadcast to all replicas (maintain consistency)
- **Tensor Parallel**: Distribute along TP dimension(s)

## Implementation Details

### Assignment at Initialization

```python
def __init__(self, ..., distributed_config: Optional[DistributedConfig] = None):
    ...
    if distributed_config is not None:
        # Get all parameters as a list
        param_list = list(chain.from_iterable(
            group['params'] for group in self.param_groups
        ))

        # Call assign_fn once to determine assignments
        self.assignments = distributed_config.assign_fn(
            param_list,
            distributed_config.state
        )

        # Store current rank for step()
        self.current_rank = distributed_config.state['rank']
```

### Step Implementation Overview

The `step()` method processes each parameter with the following flow:

**1. Momentum Update** (all ranks):
- Update momentum buffer: `momentum_buffer = beta * momentum_buffer + grad`
- Store in optimizer state

**2. Distributed Orthogonalization** (if `distributed_config` provided):
- **Set context**: Store `current_param_idx` in `state` for per-parameter TP dimension lookup
- **Assigned rank** (determined by `assign_fn`):
  1. Call `gather_fn(momentum_buffer, rank, state)` → full tensor
  2. Orthogonalize: `update = _zeropower_via_newtonschulz(full_tensor, ns_steps)`
  3. Call `redistribute_fn(update, rank, state)` → update shard
- **Non-assigned ranks**: Participate in collectives initiated by assigned rank (see below)

**3. Apply Update** (all ranks):
- `param.data.add_(update_shard, alpha=-lr)`

For concrete implementation examples, see `EXAMPLES.md`.

### Update Distribution to Non-Assigned Ranks

The `_receive_update_shard()` method handles how non-assigned ranks receive their update shards:

```python
def _receive_update_shard(self, param_idx: int) -> Tensor:
    """
    Receive update shard from the rank assigned to process this parameter.

    This is the counterpart to redistribute_fn on non-assigned ranks.
    The communication pattern depends on the distributed strategy:

    - DDP/CP: The assigned rank broadcasts the full update to all replicas
      All ranks receive the same full tensor (replicas stay in sync)

    - FSDP/TP: The assigned rank scatters/distributes shards
      Each rank receives its specific shard via the redistribute_fn mechanism

    Implementation note: This method doesn't need to explicitly call distributed
    collectives because redistribute_fn already handles broadcasting/scattering
    to all ranks (both assigned and non-assigned). Non-assigned ranks simply
    participate in the collective operation started by the assigned rank.

    Returns:
        The update shard for this rank's portion of the parameter
    """
    # In practice, this is handled implicitly by redistribute_fn's collectives
    # (broadcast for DDP/CP, scatter for FSDP/TP)
    # All ranks participate in the same collective operation
    pass
```

**Key insight**: With proper `redistribute_fn` implementation, non-assigned ranks don't need separate logic. They participate in the same broadcast/scatter collective that the assigned rank initiates, automatically receiving their shard.

### Auto-Configuration Implementation

The `create_auto_config()` function builds a `DistributedConfig` by:

1. **Detecting active strategies**: Inspects which process groups are provided
2. **Building state dict**: Collects rank info and process groups for all active strategies
3. **Creating universal functions**: Implements `assign_fn`, `gather_fn`, and `redistribute_fn` that handle all combinations

**Key implementation details**:

#### Handling per-parameter TP dimensions

```python
# In state dict setup
if tp_dim_per_param is None:
    state['tp_dim_per_param'] = {}
elif isinstance(tp_dim_per_param, int):
    # Convert single int to per-param dict during assign_fn
    state['tp_dim_default'] = tp_dim_per_param
    state['tp_dim_per_param'] = {}
else:
    state['tp_dim_per_param'] = tp_dim_per_param

# In gather_fn/redistribute_fn
param_idx = state.get('current_param_idx', 0)
tp_dim = state['tp_dim_per_param'].get(param_idx, 0)
```

**Important - State Mutation**: The `state` dict is mutable and shared across all function calls. The optimizer sets `state['current_param_idx']` before each `gather_fn`/`redistribute_fn` call (see "Step Implementation Overview" section) to enable per-parameter TP dimension lookup. This is an intentional side effect that allows the functions to know which parameter they're processing.

#### ⚠️ Critical: Operation Order

The order of operations in `gather_fn` and `redistribute_fn` is critical for correctness:

**gather_fn order** (build up full tensor):
1. **TP first** (if active): Gather along tensor-parallel dimension
2. **FSDP next** (if active): Gather along FSDP shard dimension (usually dim 0)
3. **EP**: No gather (experts are independent)
4. **DDP/CP**: No gather (already replicated)

**redistribute_fn order** (break down full tensor - **reversed!**):
1. **FSDP first** (if active): Slice along FSDP dimension ← **Reverse order!**
2. **TP next** (if active): Slice along TP dimension ← **Reverse order!**
3. **DDP/CP** (if active): Broadcast to maintain replica consistency
4. **EP**: No-op (experts stay local)

**Why reversed?** You must undo operations in reverse order when unpacking a nested structure. Think of it like unwrapping a gift: you remove the outer wrapping (FSDP) before the inner wrapping (TP).

**Example:**
```python
# gather_fn: TP → FSDP
def gather_fn(tensor, current_rank, state):
    # Step 1: Gather TP dimension (innermost)
    if 'tp_process_group' in state:
        tensor = gather_along_tp_dim(tensor, state)
    # Step 2: Gather FSDP dimension (outermost)
    if 'fsdp_process_group' in state:
        tensor = gather_along_fsdp_dim(tensor, state)
    return tensor

# redistribute_fn: FSDP → TP (reversed!)
def redistribute_fn(update, current_rank, state):
    # Step 1: Slice FSDP dimension (outermost first)
    if 'fsdp_process_group' in state:
        update = slice_along_fsdp_dim(update, state)
    # Step 2: Slice TP dimension (innermost last)
    if 'tp_process_group' in state:
        update = slice_along_tp_dim(update, state)
    # Step 3: Broadcast for replicated dimensions
    if 'ddp_process_group' in state:
        dist.broadcast(update, src=current_rank, group=state['ddp_process_group'])
    return update
```

#### DTensor Configuration Implementation

The `create_dtensor_config()` function provides a simpler alternative for DTensor-based setups:

```python
def create_dtensor_config(
    async_gpu_parallelism=True,
    prefetch_count=1,
):
    """Create DistributedConfig for DTensor-based distributed training."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    # Get rank information from default process group
    global_rank = dist.get_rank()
    global_world_size = dist.get_world_size()

    # Build minimal state dict
    state = {
        'global_rank': global_rank,
        'global_world_size': global_world_size,
        'is_dtensor': True,
        'param_metadata': {},  # Will store per-param DTensor metadata
    }

    def assign_fn(params, state):
        """
        Simple round-robin assignment for DTensor params.

        Called ONCE during __init__ to populate metadata.
        Stores DTensor device mesh and placements for each parameter.
        """
        global_rank = state['global_rank']
        global_world_size = state['global_world_size']

        assignments = {i: [] for i in range(global_world_size)}
        for param_idx, param in enumerate(params):
            # Store DTensor metadata for use in gather/redistribute
            # (see DTensor-Based Examples in EXAMPLES.md for details on param.data)
            if isinstance(param.data, DTensor):
                state['param_metadata'][param_idx] = {
                    'device_mesh': param.data.device_mesh,
                    'placements': param.data.placements,
                }

            assigned_rank = param_idx % global_world_size
            assignments[assigned_rank].append(param_idx)

        return assignments

    def gather_fn(tensor, current_rank, state):
        """
        Gather full tensor from DTensor shard.

        Called during EACH step() to gather momentum buffers.
        Reads metadata populated by assign_fn during __init__.

        For DTensor: calls full_tensor() which:
        - Returns a regular Tensor (not DTensor)
        - Automatically handles Shard/Replicate/Partial placements
        - Is a no-op for Replicate() (already full)

        For regular tensors: returns as-is (already full).

        Note: The optimizer sets state['current_param_idx'] before calling this.
        """
        if isinstance(tensor, DTensor):
            # DTensor.full_tensor() handles all the gathering
            return tensor.full_tensor()
        else:
            # Regular tensor - already full
            return tensor

    def redistribute_fn(update, current_rank, state):
        """
        Convert full update tensor back to DTensor with original placement.

        Called during EACH step() to redistribute orthogonalized updates.
        Reads metadata populated by assign_fn during __init__.

        For DTensor: recreate DTensor with same device mesh and placements.
        The distribute_tensor() function handles redistribution automatically.

        For regular tensors: returns as-is.

        Note: The optimizer sets state['current_param_idx'] before calling this,
        enabling per-parameter metadata lookup.
        """
        param_idx = state.get('current_param_idx', 0)
        metadata = state['param_metadata'].get(param_idx)

        if metadata is not None:
            # Recreate DTensor with original placement
            from torch.distributed.tensor import distribute_tensor
            return distribute_tensor(
                update,
                device_mesh=metadata['device_mesh'],
                placements=metadata['placements'],
            )
        else:
            # Regular tensor - return as is
            return update

    return DistributedConfig(
        assign_fn=assign_fn,
        gather_fn=gather_fn,
        redistribute_fn=redistribute_fn,
        state=state,
        async_gpu_parallelism=async_gpu_parallelism,
        prefetch_count=prefetch_count,
    )
```

**Key advantages of DTensor approach**:

1. **Simpler API**: No need to specify process groups or sharding dimensions - DTensor already has this
2. **Universal gather**: `DTensor.full_tensor()` works for any placement (Shard, Replicate, Partial)
3. **Automatic redistribution**: `distribute_tensor()` handles all communication patterns
4. **Type safety**: DTensor metadata is preserved through the optimization step

**DTensor placement handling**:
- **Shard(dim)**: `full_tensor()` gathers along sharded dimension(s)
- **Replicate()**: `full_tensor()` is a no-op (already full)
- **Partial()**: `full_tensor()` reduces partial values appropriately
- **Hybrid placements**: Automatically handled by DTensor's placement composition

### Expert Parallel special handling

Expert Parallel requires special logic because experts are independent:

- **assign_fn**: Assign parameters to ranks based on expert ownership
- **gather_fn**: No-op (don't gather across experts)
- **redistribute_fn**: No-op (each expert's update stays local)

```python
def assign_fn(params, state):
    if 'expert_assignments' in state and state['expert_assignments']:
        # Expert parallel: assign based on expert ownership
        ep_rank = state['ep_rank']
        assignments = {i: [] for i in range(state['global_world_size'])}

        for param_idx, expert_id in state['expert_assignments'].items():
            if expert_id == ep_rank:
                assignments[state['global_rank']].append(param_idx)

        return assignments
    else:
        # Regular round-robin assignment
        ...
```

## Acceleration Features

### async_gpu_parallelism

**Purpose**: Enable parallel processing of different parameters on different ranks.

**When enabled (`True`)**:
- Rank 0 processes param 0 while Rank 1 processes param 1 simultaneously
- Requires careful synchronization at parameter boundaries
- Maximum GPU utilization

**When disabled (`False`)**:
- All ranks process parameters in the same order
- Easier to debug (deterministic execution)
- Still zero redundancy (each rank only orthogonalizes its assigned params)

**Implementation approach**:
```python
if async_gpu_parallelism:
    # Use async operations and careful synchronization
    for param in my_assigned_params:
        launch_async_orthogonalize(param)
    synchronize()
else:
    # Process all params in order, but only work on assigned ones
    for param in all_params:
        if param in my_assigned_params:
            orthogonalize(param)
        synchronize()  # Wait for assigned rank to finish
```

### prefetch_count

**Purpose**: Overlap communication (gather/redistribute) with computation (orthogonalization).

**How it works**:
1. While orthogonalizing parameter N, start gathering parameter N+1
2. Uses `async_op=True` in distributed collectives
3. Requires buffering of in-flight operations

**Example with `prefetch_count=1`**:
```python
# Start gathering param 1
gather_handle = gather_fn(..., async_op=True)

for i in range(num_params):
    # Wait for previous gather
    if i > 0:
        momentum_buffer = gather_handle.wait()

    # Start prefetch for next param
    if i < num_params - 1:
        gather_handle = gather_fn(next_param, ..., async_op=True)

    # Compute while next gather happens
    update = orthogonalize(momentum_buffer)

    # Redistribute (could also be async)
    redistribute_fn(update, ...)
```

**Memory impact**: `prefetch_count=N` requires buffering N additional tensors (full size).

**Recommended values**:
- Development/debugging: 0 (disabled)
- Production: 1-2 (sweet spot for most models)
- Large models with slow communication: 2-3

## Error Handling

### Validation at Initialization

The `__init__` method should validate:

1. **Assignment coverage**: Every parameter assigned to exactly one rank
   ```python
   all_assigned = set(chain.from_iterable(assignments.values()))
   assert all_assigned == set(range(len(param_list)))
   ```

2. **No overlaps**: No parameter assigned to multiple ranks
   ```python
   total_assigned = sum(len(v) for v in assignments.values())
   assert total_assigned == len(param_list)
   ```

3. **State dict contains required keys**: Depends on implementation, but typically:
   - `rank`: Current rank ID
   - Process groups or device meshes

4. **DTensor device mesh compatibility** (for DTensor configs):
   ```python
   # Check that all DTensor parameters use compatible device meshes
   dtensor_params = [p for p in param_list if isinstance(p.data, DTensor)]
   if dtensor_params:
       first_mesh = dtensor_params[0].data.device_mesh
       for p in dtensor_params[1:]:
           if p.data.device_mesh != first_mesh:
               raise ValueError(
                   f"All DTensor parameters must use the same device mesh. "
                   f"Found incompatible meshes: {first_mesh} vs {p.data.device_mesh}"
               )
   ```

### Runtime Errors

**gather_fn failures**:
- Mismatched tensor shapes across ranks → Check FSDP sharding configuration
- Process group errors → Verify `state` contains correct process groups
- DTensor errors → Ensure DTensor parameters have compatible device meshes

**redistribute_fn failures**:
- Shape mismatches → Ensure redistribute produces same shape as original shard
- Synchronization deadlocks → Check that exactly one rank calls this per parameter
- DTensor placement errors → Verify placements are valid for the parameter shape

### DTensor-Specific Errors

**Incompatible device meshes**:
- **Error**: Parameters have different device meshes
- **Cause**: Model parameters were created with different `DeviceMesh` objects
- **Solution**: Ensure all DTensor parameters use the same device mesh during initialization

**Invalid placement for parameter shape**:
- **Error**: Cannot shard tensor along dimension that doesn't exist
- **Cause**: Placement specifies `Shard(dim)` where `dim >= param.ndim`
- **Solution**: Verify placement dimensions are valid for each parameter's shape

**Mixed DTensor/regular parameters**:
- **Behavior**: Supported - regular parameters are treated as already-full tensors
- **Performance**: May be suboptimal; consider converting all parameters to DTensor
- **Recommendation**: For best performance and simplicity, use DTensor for all parameters in distributed training

## Performance Considerations

### Communication Overhead

For a parameter of size `S` with `N` ranks:
- **FSDP shard size**: `S / N`
- **gather_fn communication**: `(N-1) * S/N = S * (N-1)/N` (all-gather)
- **redistribute_fn communication**: Similar magnitude

**Optimization**: Use `prefetch_count > 0` to overlap with computation.

### Computation Balance

With balanced assignment:
- Each rank processes `num_params / num_ranks` parameters
- Orthogonalization cost: `O(d^3)` for `d×d` matrix (Newton-Schulz iterations)

**Optimization**: Assign larger parameters to ranks with fewer total parameters.

### Memory Requirements

Per parameter being processed:
- Full momentum buffer: `S` (gathered)
- Update tensor: `S` (after orthogonalization)
- With `prefetch_count=N`: Additional `N * S` for prefetched buffers

## Design Decisions

### Why user-defined functions?

**Flexibility**: Different distributed strategies require different gather/scatter patterns. User-defined functions support:
- PyTorch built-in (FSDP, DDP, etc.)
- Custom frameworks (Megatron-LM, DeepSpeed, etc.)
- Proprietary distributed systems
- Future parallelism strategies

**Composability**: One set of functions can handle multiple parallelism strategies simultaneously (e.g., FSDP + TP + DP).

**Convenience with flexibility**: The `create_auto_config()` function provides pre-built implementations for common PyTorch strategies, while the underlying `DistributedConfig` API allows complete customization for advanced or proprietary setups. Users can:
- Use `create_auto_config()` for standard cases (recommended)
- Define custom functions for unusual distributed architectures
- Mix and match: use `create_auto_config()` then override specific functions if needed

### Why dict[str, Any] for state?

**Flexibility**: Different setups need different metadata. Examples:
- FSDP: `{'rank': 0, 'world_size': 4, 'fsdp_group': pg}`
- FSDP+TP: `{'rank': 0, 'fsdp_group': pg1, 'tp_group': pg2, 'mesh': device_mesh}`

**Extensibility**: New metadata can be added without API changes.

### Why separate gather and redistribute?

**Asymmetry**: Gather happens before orthogonalization, redistribute after. They may have different patterns:
- FSDP: `all_gather` for gather, then slice for redistribute
- Replicated: No-op for gather, `broadcast` for redistribute

**Clarity**: Explicit functions make the dataflow clear.
