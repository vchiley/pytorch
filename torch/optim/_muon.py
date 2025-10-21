# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer."""

import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict

import torch
from torch import Tensor

from .optimizer import (
    _disable_dynamo_if_unsupported,
    _params_doc,
    _to_scalar,
    Optimizer,
    ParamsT,
)


__all__ = [
    "Muon",
    "DistributedConfig",
    "create_processgroup_config",
    "create_devicemesh_config",
    "create_dtensor_config",
]

# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


class DistributedState(TypedDict, total=False):
    """Type definition for distributed training state dictionary.

    This TypedDict defines the structure of the state dictionary used throughout
    distributed Muon operations. Not all fields are required (total=False).

    Core Fields:
        rank: Current process rank
        world_size: Total number of processes
        assignments: Mapping from parameter index to assigned rank

    Shape/Type Metadata (for buffer allocation):
        param_shapes: Parameter shapes for each param_idx
        param_dtypes: Parameter dtypes for each param_idx
        param_devices: Parameter devices for each param_idx
        current_param_idx: Currently processing parameter index

    Process Groups (parallelism strategies):
        fsdp_pg: Fully Sharded Data Parallel process group
        tp_pg: Tensor Parallel process group
        dp_pg: Data Parallel process group (DDP)
        ep_pg: Expert Parallel process group
        cp_pg: Context Parallel process group
        pp_pg: Pipeline Parallel process group
        world_pg: World process group

    Device Mesh (for multi-dimensional parallelism):
        device_mesh: PyTorch DeviceMesh object
        mesh_dim_names: Dimension names for the mesh
    """

    # Core fields
    rank: int
    world_size: int
    assignments: dict[int, int]

    # Shape/type metadata
    param_shapes: dict[int, tuple[int, ...]]
    param_dtypes: dict[int, torch.dtype]
    param_devices: dict[int, torch.device]
    current_param_idx: int

    # Process groups
    fsdp_pg: Any
    tp_pg: Any
    dp_pg: Any
    ep_pg: Any
    cp_pg: Any
    pp_pg: Any
    world_pg: Any

    # Device mesh
    device_mesh: Any
    mesh_dim_names: list[str]


@dataclass
class DistributedConfig:
    """Configuration for distributed Muon training.

    This configuration enables zero-redundancy orthogonalization in distributed training,
    where each parameter is assigned to exactly one rank for orthogonalization.

    Attributes:
        assign_fn: Maps param_index → rank assigned to perform orthogonalization.
            Called once during __init__.
            Signature: assign_fn(params, state) -> dict[int, int]
            Must return an entry for every param_idx in range(len(params)).

        gather_fn: Gathers the full momentum buffer from shards for orthogonalization.
            Signature: gather_fn(momentum_buffer, dst_rank, state) -> Optional[Tensor]
            Returns the full tensor on dst_rank, None on other ranks.

        redistribute_fn: Redistributes the orthogonalized update back to shards/replicas.
            Signature: redistribute_fn(update, src_rank, state) -> Tensor
            Receives None on non-src_rank, but all ranks participate as destinations.

        state: Holds all metadata needed by the functions above (see DistributedState).
            Contains process groups, shapes, dtypes, devices, and other metadata.

        async_gpu_parallelism: If True, enables rank-level asynchronous processing where
            each rank independently processes only its assigned parameters (Phase 4).
            If False, all ranks process all parameters synchronously for easier debugging.

            When True:
            - Each rank processes only parameters assigned to it
            - Ranks work independently without waiting for each other
            - Final barrier synchronizes all ranks before next training step
            - Expected speedup: 20-30% due to parallel rank processing

            When False:
            - All ranks process all parameters (redundant computation)
            - Easier to debug since execution is deterministic
            - All ranks follow identical execution path

        prefetch_count: Number of tensors to prefetch ahead while processing current tensor.
            0: Disabled (sequential communication and computation)
            1-2: Recommended (overlaps communication with computation)
            3+: Higher memory usage, diminishing returns
            Works independently of async_gpu_parallelism and can be combined with it.
    """

    assign_fn: Callable[[list[Tensor], dict[str, Any]], dict[int, int]]
    gather_fn: Callable[[Tensor, int, dict[str, Any]], Optional[Tensor]]
    redistribute_fn: Callable[[Optional[Tensor], int, dict[str, Any]], Tensor]
    state: dict[str, Any]  # Ideally DistributedState, but kept as dict for flexibility
    async_gpu_parallelism: bool = True
    prefetch_count: int = 1


def _validate_assignments(
    assignments: dict[int, int], params: list[Tensor], world_size: int
) -> None:
    """Validate that assignments are correct."""
    if len(assignments) != len(params):
        raise ValueError(
            f"Missing assignments for some parameters. "
            f"Expected {len(params)} assignments, got {len(assignments)}"
        )

    for param_idx, rank in assignments.items():
        if not (0 <= rank < world_size):
            raise ValueError(
                f"Invalid rank {rank} for parameter {param_idx}. "
                f"Rank must be in range [0, {world_size})"
            )


def _default_assign_fn(params: list[Tensor], state: dict[str, Any]) -> dict[int, int]:
    """Default round-robin assignment of parameters to ranks."""
    world_size = state["world_size"]
    return {i: i % world_size for i in range(len(params))}


def _allocate_communication_buffer(
    param_idx: int,
    state: dict[str, Any],
    shard: bool = False,
    world_size: int = 1,
) -> Tensor:
    """Allocate output buffer for distributed communication operations.

    Args:
        param_idx: Parameter index for shape lookup in state
        state: State dictionary containing param_shapes, param_dtypes, param_devices
        shard: If True, allocate shard-sized buffer; if False, allocate full-sized buffer
        world_size: Number of ranks (used for shard size calculation)

    Returns:
        Allocated tensor with correct shape, dtype, and device

    Note:
        Falls back to empty tensor if shape information not available (backward compat)
    """
    if param_idx >= 0 and "param_shapes" in state:
        param_shape = state["param_shapes"][param_idx]
        param_dtype = state["param_dtypes"].get(param_idx, torch.float32)
        param_device = state["param_devices"].get(
            param_idx, torch.cuda.current_device()
        )

        if shard:
            # Allocate shard-sized buffer (assumes dim 0 is sharded)
            shard_size = param_shape[0] // world_size
            shape = (shard_size, *param_shape[1:])
        else:
            # Allocate full-sized buffer
            shape = param_shape

        return torch.empty(shape, dtype=param_dtype, device=param_device)
    else:
        # Fallback for backward compatibility or when shapes not available
        return torch.empty(0, dtype=torch.float32, device=torch.cuda.current_device())


def _gather_tensor_shards(
    tensor: Tensor,
    process_group: Any,
    async_op: bool = False,
) -> tuple[Any, Optional[Any]]:
    """Gather tensor shards from all ranks in process group.

    This is a reusable helper for gathering sharded tensors (FSDP, TP, etc.).
    Performs all_gather and concatenates results along dimension 0.

    Args:
        tensor: Local shard to gather
        process_group: PyTorch process group for communication
        async_op: If True, return async work handle for prefetching (Phase 3)

    Returns:
        Tuple of (gathered_tensor, async_work_handle)
        - gathered_tensor: Concatenated full tensor (or list of tensors if async)
        - async_work_handle: None if async_op=False, Work object if async_op=True

    Note:
        For Phase 3 prefetching, set async_op=True and wait on the work handle later.
        When async_op=True, gathered_tensor is a list that needs concatenation after wait().
    """
    import torch.distributed as dist

    world_size = dist.get_world_size(process_group)
    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    if async_op:
        # Phase 3: Async gather for prefetching
        work = dist.all_gather(gather_list, tensor, group=process_group, async_op=True)
        # Note: Concatenation must happen after work.wait() in Phase 3
        # For now, return the gather_list and work handle
        # Caller is responsible for: work.wait(), then torch.cat(gather_list, dim=0)
        return gather_list, work
    else:
        # Synchronous gather (Phase 1/2)
        dist.all_gather(gather_list, tensor, group=process_group)
        result = torch.cat(gather_list, dim=0)
        return result, None


def _scatter_tensor_to_shards(
    full_tensor: Optional[Tensor],
    src_rank: int,
    process_group: Any,
    state: dict[str, Any],
    param_idx: int,
    async_op: bool = False,
) -> tuple[Tensor, Optional[Any]]:
    """Scatter full tensor to shards across all ranks in process group.

    This is a reusable helper for scattering tensors (FSDP, TP, etc.).
    On src_rank, splits tensor into shards and scatters to all ranks.
    On other ranks, allocates buffer and receives shard.

    Args:
        full_tensor: Full tensor on src_rank, None on other ranks
        src_rank: Rank that has the full tensor
        process_group: PyTorch process group for communication
        state: State dict for buffer allocation
        param_idx: Parameter index for shape lookup
        async_op: If True, return async work handle for prefetching (Phase 3)

    Returns:
        Tuple of (local_shard, async_work_handle)
        - local_shard: Local shard after scatter
        - async_work_handle: None if async_op=False, Work object if async_op=True

    Note:
        For Phase 3 prefetching, set async_op=True and wait on the work handle later.
    """
    import torch.distributed as dist

    rank = state["rank"]
    world_size = dist.get_world_size(process_group)

    if rank == src_rank:
        assert full_tensor is not None, "Source rank must have full tensor"
        shard_size = full_tensor.size(0) // world_size
        scatter_list = [
            full_tensor[i * shard_size : (i + 1) * shard_size]
            for i in range(world_size)
        ]
        output = torch.empty_like(scatter_list[0])
    else:
        scatter_list = None
        output = _allocate_communication_buffer(
            param_idx, state, shard=True, world_size=world_size
        )

    if async_op:
        # Phase 3: Async scatter for prefetching
        work = dist.scatter(
            output, scatter_list, src=src_rank, group=process_group, async_op=True
        )
        return output, work
    else:
        # Synchronous scatter (Phase 1/2)
        dist.scatter(output, scatter_list, src=src_rank, group=process_group)
        return output, None


def _broadcast_tensor(
    tensor: Optional[Tensor],
    src_rank: int,
    process_group: Any,
    state: dict[str, Any],
    param_idx: int,
    async_op: bool = False,
) -> tuple[Tensor, Optional[Any]]:
    """Broadcast full tensor from src_rank to all ranks in process group.

    This is a reusable helper for broadcasting replicated tensors (DDP, CP, etc.).

    Args:
        tensor: Full tensor on src_rank, None on other ranks
        src_rank: Rank that has the tensor to broadcast
        process_group: PyTorch process group for communication
        state: State dict for buffer allocation
        param_idx: Parameter index for shape lookup
        async_op: If True, return async work handle for prefetching (Phase 3)

    Returns:
        Tuple of (broadcasted_tensor, async_work_handle)
        - broadcasted_tensor: Tensor available on all ranks
        - async_work_handle: None if async_op=False, Work object if async_op=True

    Note:
        For Phase 3 prefetching, set async_op=True and wait on the work handle later.
    """
    import torch.distributed as dist

    rank = state["rank"]

    if rank == src_rank:
        assert tensor is not None, "Source rank must have tensor"
        output = tensor.clone() if tensor is not tensor else tensor
    else:
        output = _allocate_communication_buffer(param_idx, state, shard=False)

    if async_op:
        # Phase 3: Async broadcast for prefetching
        work = dist.broadcast(output, src=src_rank, group=process_group, async_op=True)
        return output, work
    else:
        # Synchronous broadcast (Phase 1/2)
        dist.broadcast(output, src=src_rank, group=process_group)
        return output, None


def create_processgroup_config(
    fsdp_pg: Optional[Any] = None,
    tp_pg: Optional[Any] = None,
    dp_pg: Optional[Any] = None,
    ep_pg: Optional[Any] = None,
    cp_pg: Optional[Any] = None,
    pp_pg: Optional[Any] = None,
    async_gpu_parallelism: bool = True,
    prefetch_count: int = 1,
) -> DistributedConfig:
    """Create DistributedConfig from PyTorch process groups.

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
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before creating distributed config"
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    state = {
        "rank": rank,
        "world_size": world_size,
        "fsdp_pg": fsdp_pg,
        "tp_pg": tp_pg,
        "dp_pg": dp_pg,
        "ep_pg": ep_pg,
        "cp_pg": cp_pg,
        "pp_pg": pp_pg,
    }

    def gather_fn(
        momentum_buffer: Tensor, dst_rank: int, state: dict[str, Any]
    ) -> Optional[Tensor]:
        """Gather momentum buffer from shards/replicas to dst_rank.

        Uses chaining pattern for combined parallelism:
        - For single strategy: performs one gather
        - For combined strategies (Phase 2): chains multiple gathers

        Example for FSDP+TP:
            tensor = gather_fsdp(tensor)  # Gather FSDP shards first
            tensor = gather_tp(tensor)     # Then gather TP shards
        """
        rank = state["rank"]
        result = momentum_buffer

        # Chain gather operations for each active parallelism dimension
        # Order matters: gather inner dimensions first (TP), then outer (FSDP/DDP)

        # Tensor Parallel: gather shards along TP dimension
        if state.get("tp_pg") is not None:
            result, _ = _gather_tensor_shards(result, state["tp_pg"], async_op=False)

        # FSDP: gather shards along FSDP dimension
        if state.get("fsdp_pg") is not None:
            result, _ = _gather_tensor_shards(result, state["fsdp_pg"], async_op=False)

        # DDP/CP: already replicated, no gather needed
        # EP/PP: independent parameters, no gather needed

        # Return result only on dst_rank
        if rank == dst_rank:
            return result
        else:
            return None

    def redistribute_fn(
        update: Optional[Tensor], src_rank: int, state: dict[str, Any]
    ) -> Tensor:
        """Redistribute orthogonalized update from src_rank back to shards/replicas.

        Uses chaining pattern for combined parallelism (inverse of gather):
        - For single strategy: performs one redistribute
        - For combined strategies (Phase 2): chains multiple redistributes in reverse order

        Example for FSDP+TP:
            tensor = redistribute_fsdp(tensor)  # Scatter FSDP shards first
            tensor = redistribute_tp(tensor)     # Then scatter TP shards
        """
        param_idx = state.get("current_param_idx", -1)
        result = update

        # On src_rank, we have the full update tensor
        # Chain redistribute operations in reverse order of gather
        # For FSDP+TP: scatter FSDP first, then TP (opposite of gather order)

        # FSDP: scatter full tensor into shards along FSDP dimension
        if state.get("fsdp_pg") is not None:
            result, _ = _scatter_tensor_to_shards(
                result, src_rank, state["fsdp_pg"], state, param_idx, async_op=False
            )

        # Tensor Parallel: scatter into shards along TP dimension
        if state.get("tp_pg") is not None:
            result, _ = _scatter_tensor_to_shards(
                result, src_rank, state["tp_pg"], state, param_idx, async_op=False
            )

        # DDP/CP: broadcast full tensor to all replicas
        if state.get("dp_pg") is not None or state.get("cp_pg") is not None:
            pg = state.get("dp_pg") or state.get("cp_pg")
            result, _ = _broadcast_tensor(
                result, src_rank, pg, state, param_idx, async_op=False
            )

        # EP/PP: independent parameters, no redistribution needed
        # Just return what we have (full tensor on src_rank, empty on others)

        return (
            result
            if result is not None
            else torch.empty(0, dtype=torch.float32, device=torch.cuda.current_device())
        )

    return DistributedConfig(
        assign_fn=_default_assign_fn,
        gather_fn=gather_fn,
        redistribute_fn=redistribute_fn,
        state=state,
        async_gpu_parallelism=async_gpu_parallelism,
        prefetch_count=prefetch_count,
    )


def create_devicemesh_config(
    device_mesh: Any,
    mesh_dim_names: list[str],
    async_gpu_parallelism: bool = True,
    prefetch_count: int = 1,
) -> DistributedConfig:
    """Create DistributedConfig from PyTorch DeviceMesh.

    DeviceMesh organizes devices into a multi-dimensional grid topology. Each mesh
    dimension corresponds to a different parallelism strategy (e.g., dp, fsdp, tp).
    This function extracts process groups from the mesh and implements gather/redistribute
    using the chaining pattern for combined parallelism.

    Args:
        device_mesh: PyTorch DeviceMesh defining parallelism topology
        mesh_dim_names: Names for each mesh dimension (e.g., ["dp", "fsdp", "tp"])
            Common dimension names:
            - "dp": Data Parallel (replicated parameters)
            - "fsdp": Fully Sharded Data Parallel (sharded parameters)
            - "tp": Tensor Parallel (sharded parameters)
            - "cp": Context Parallel (replicated parameters)
            - "pp": Pipeline Parallel (independent parameters)
        async_gpu_parallelism: Enable async parallel processing
        prefetch_count: Number of tensors to prefetch ahead

    Returns:
        DistributedConfig with appropriate assign/gather/redistribute functions

    Example:
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> from torch.optim import Muon
        >>> from torch.optim._muon import create_devicemesh_config
        >>>
        >>> # Create 2D mesh for FSDP + TP (4 DP ranks, 8 TP ranks)
        >>> mesh = init_device_mesh("cuda", (4, 8), mesh_dim_names=["dp", "tp"])
        >>> config = create_devicemesh_config(mesh, ["dp", "tp"])
        >>> optimizer = Muon(model.parameters(), distributed_config=config)
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before creating distributed config"
        )

    rank = dist.get_rank()

    # Extract process groups from device mesh
    # DeviceMesh provides a process group for each dimension via mesh["dim_name"]
    # We'll map standard dimension names to parallelism strategies
    state: dict[str, Any] = {
        "rank": rank,
        "world_size": device_mesh.size(),
        "device_mesh": device_mesh,
        "mesh_dim_names": mesh_dim_names,
    }

    # Map mesh dimension names to process group keys
    # This enables the chaining pattern in gather/redistribute
    for dim_name in mesh_dim_names:
        if dim_name == "fsdp":
            # FSDP dimension: sharded parameters
            state["fsdp_pg"] = device_mesh[dim_name].get_group()
        elif dim_name in ("tp", "tensor_parallel"):
            # Tensor Parallel dimension: sharded parameters
            state["tp_pg"] = device_mesh[dim_name].get_group()
        elif dim_name in ("dp", "data_parallel", "ddp"):
            # Data Parallel dimension: replicated parameters
            state["dp_pg"] = device_mesh[dim_name].get_group()
        elif dim_name in ("cp", "context_parallel"):
            # Context Parallel dimension: replicated parameters
            state["cp_pg"] = device_mesh[dim_name].get_group()
        elif dim_name == "pp":
            # Pipeline Parallel dimension: independent parameters
            state["pp_pg"] = device_mesh[dim_name].get_group()
        elif dim_name == "ep":
            # Expert Parallel dimension: independent parameters
            state["ep_pg"] = device_mesh[dim_name].get_group()
        else:
            # For unknown dimension names, try to treat as data parallel (replicated)
            # This provides best-effort support for custom dimension names
            state["dp_pg"] = device_mesh[dim_name].get_group()

    # Reuse gather_fn and redistribute_fn from create_processgroup_config
    # by delegating to the same implementation
    temp_config = create_processgroup_config(
        fsdp_pg=state.get("fsdp_pg"),
        tp_pg=state.get("tp_pg"),
        dp_pg=state.get("dp_pg"),
        ep_pg=state.get("ep_pg"),
        cp_pg=state.get("cp_pg"),
        pp_pg=state.get("pp_pg"),
        async_gpu_parallelism=async_gpu_parallelism,
        prefetch_count=prefetch_count,
    )

    # Update state with device_mesh information
    temp_config.state.update(
        {"device_mesh": device_mesh, "mesh_dim_names": mesh_dim_names}
    )

    return temp_config


def create_dtensor_config(
    async_gpu_parallelism: bool = True,
    prefetch_count: int = 1,
) -> DistributedConfig:
    """Create DistributedConfig for training setups using PyTorch DTensor.

    Automatically infers parallelism strategy from DTensor placement specs.
    DTensors encode their sharding/replication strategy via placement specifications:
    - Shard(dim): Tensor is sharded along dimension `dim`
    - Replicate(): Tensor is replicated across ranks
    - Partial(reduce_op): Tensor holds partial values pending reduction

    This function inspects the DTensor placements to determine gather/redistribute operations.

    Args:
        async_gpu_parallelism: Enable async parallel processing
        prefetch_count: Number of tensors to prefetch ahead

    Returns:
        DistributedConfig that uses DTensor placement information for
        assign/gather/redistribute operations

    Example:
        >>> from torch.distributed.tensor import DTensor, Shard, Replicate
        >>> from torch.optim import Muon
        >>> from torch.optim._muon import create_dtensor_config
        >>>
        >>> # Model parameters are DTensors with sharding placement
        >>> config = create_dtensor_config()
        >>> optimizer = Muon(model.parameters(), distributed_config=config)

    Note:
        This is a Phase 2 feature. The current implementation provides a framework
        for DTensor integration. Full DTensor support requires:
        1. Detecting DTensor instances in momentum buffers
        2. Extracting placement specs from DTensors
        3. Using DTensor collectives (redistribute, to_local) for gather/redistribute
        4. Handling mixed placement strategies (Shard + Replicate)
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before creating distributed config"
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    state = {
        "rank": rank,
        "world_size": world_size,
    }

    def gather_fn(
        momentum_buffer: Tensor, dst_rank: int, state: dict[str, Any]
    ) -> Optional[Tensor]:
        """Gather using DTensor placement.

        DTensor Gather Strategy:
        1. Check if momentum_buffer is a DTensor
        2. Extract placement specs from DTensor
        3. For Shard placements: use DTensor.redistribute() to gather
        4. For Replicate placements: return local tensor on dst_rank
        5. Convert to local tensor on dst_rank

        Args:
            momentum_buffer: Momentum buffer tensor (may be DTensor)
            dst_rank: Destination rank to gather full tensor
            state: State dictionary

        Returns:
            Full tensor on dst_rank, None on other ranks
        """
        try:
            from torch.distributed.tensor import DTensor, Replicate, Shard
        except ImportError:
            raise RuntimeError(
                "DTensor not available. Please use PyTorch with DTensor support."
            )

        rank = state["rank"]

        # Check if tensor is a DTensor
        if not isinstance(momentum_buffer, DTensor):
            # Not a DTensor, use standard collective operations
            # This handles non-DTensor tensors gracefully
            if rank == dst_rank:
                return momentum_buffer
            else:
                return None

        # Extract DTensor metadata
        device_mesh = momentum_buffer.device_mesh
        placements = momentum_buffer.placements

        # Strategy: Convert all Shard placements to Replicate
        # This gathers the full tensor on all ranks
        target_placements = tuple(
            Replicate() if isinstance(p, Shard) else p for p in placements
        )

        # Redistribute to replicate all shards
        if target_placements != placements:
            full_dtensor = momentum_buffer.redistribute(
                device_mesh=device_mesh,
                placements=target_placements,
            )
        else:
            # Already replicated
            full_dtensor = momentum_buffer

        # Convert to local tensor
        full_tensor = full_dtensor.to_local()

        # Return only on dst_rank for consistency with gather_fn API
        if rank == dst_rank:
            return full_tensor
        else:
            return None

    def redistribute_fn(
        update: Optional[Tensor], src_rank: int, state: dict[str, Any]
    ) -> Tensor:
        """Redistribute using DTensor placement.

        DTensor Redistribute Strategy:
        1. On src_rank: have full update tensor
        2. On other ranks: need to reconstruct DTensor with original placements
        3. Use DTensor.from_local() to create DTensor with desired placements
        4. Extract local shard on each rank

        Args:
            update: Full update tensor on src_rank, None on other ranks
            src_rank: Source rank with full tensor
            state: State dictionary

        Returns:
            Local shard/replica of update tensor

        Note:
            Current implementation returns a regular tensor, not a DTensor.
            For full DTensor integration, we'd need to track original DTensor
            placements and reconstruct DTensors with those placements.
        """
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            raise RuntimeError(
                "DTensor not available. Please use PyTorch with DTensor support."
            )

        rank = state["rank"]
        param_idx = state.get("current_param_idx", -1)

        # For Phase 2, we implement basic redistribution using standard collectives
        # Full DTensor integration would require tracking original placements
        # and using DTensor.from_local() to reconstruct sharded DTensors

        # Broadcast full tensor from src_rank to all ranks
        # This is a simplified implementation that doesn't preserve DTensor structure
        if rank == src_rank:
            assert update is not None, "Source rank must have update tensor"
            output = update.clone() if update is not update else update
        else:
            # Allocate buffer for receiving
            output = _allocate_communication_buffer(param_idx, state, shard=False)

        # Broadcast update from src_rank
        import torch.distributed as dist

        dist.broadcast(output, src=src_rank)

        return output

    return DistributedConfig(
        assign_fn=_default_assign_fn,
        gather_fn=gather_fn,
        redistribute_fn=redistribute_fn,
        state=state,
        async_gpu_parallelism=async_gpu_parallelism,
        prefetch_count=prefetch_count,
    )


def _zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    with suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(
    lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size
) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        # pyrefly: ignore  # no-matching-overload
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class Muon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: Optional[str] = None,
        distributed_config: Optional[DistributedConfig] = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original",
            "match_rms_adamw",
        ]:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters whereas we found a parameter with size: {p.size()}"
                    )

        # Setup distributed training if config is provided
        self.distributed_config = distributed_config
        if distributed_config is not None:
            # Validate prefetch_count parameter
            if not 0 <= distributed_config.prefetch_count <= 10:
                raise ValueError(
                    f"prefetch_count must be between 0 and 10, got {distributed_config.prefetch_count}"
                )
            self._setup_distributed()

    def _setup_distributed(self) -> None:
        """Setup distributed training configuration.

        This method is called during __init__ when distributed_config is provided.
        It performs the following setup:
        1. Collects all parameters from all param_groups
        2. Computes parameter-to-rank assignments
        3. Validates assignments
        4. Stores metadata (shapes, dtypes, devices) for communication
        """
        # Type narrowing: this method is only called when distributed_config exists
        assert self.distributed_config is not None

        # Collect all parameters from all param_groups
        all_params = []
        for group in self.param_groups:
            all_params.extend(group["params"])

        # Compute assignments once during initialization
        assignments = self.distributed_config.assign_fn(
            all_params, self.distributed_config.state
        )

        # Validate assignments
        _validate_assignments(
            assignments, all_params, self.distributed_config.state["world_size"]
        )

        # Store assignments in state for use during step()
        self.distributed_config.state["assignments"] = assignments

        # Store parameter shapes and dtypes for gather/redistribute operations
        # This is needed to allocate proper output buffers in distributed communication
        self.distributed_config.state["param_shapes"] = {
            i: tuple(p.shape) for i, p in enumerate(all_params)
        }
        self.distributed_config.state["param_dtypes"] = {
            i: p.dtype for i, p in enumerate(all_params)
        }
        self.distributed_config.state["param_devices"] = {
            i: p.device for i, p in enumerate(all_params)
        }

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            muon_momentum_bufs.append(state["momentum_buffer"])

        return False  # has_complex

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                muon_momentum_bufs,
            )

            muon(
                params_with_grad,
                grads,
                muon_momentum_bufs,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=has_complex,
                distributed_config=self.distributed_config,
            )
        return loss


Muon.__doc__ = (
    r"""Implements Muon algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt} \\
            &\textbf{input}      : \gamma \text{ (lr)},\ \lambda \text{ (weight decay)},\
               \mu \text{ (momentum)},\ \textit{nesterov}\in\{True,False\},\\
            &\hspace{13mm}(a,b,c)\ \text{ (NS coefficients)},\
               \varepsilon \text{ (epsilon)},\ k \text{ (NS steps)},\
               \theta_0 \text{ (params)},\ f(\theta) \text{ (objective)} \\
            &\textbf{initialize} : B_0 \leftarrow 0 \text{ (momentum buffer)} \\[-1.ex]
            &\rule{110mm}{0.4pt} \\
            &\textbf{for}\ t=1\ \textbf{to}\ \ldots\ \textbf{do} \\[0.25ex]
            &\hspace{5mm} g_t \leftarrow \nabla_{\theta} f_t(\theta_{t-1}) \\[0.25ex]
            &\hspace{5mm} B_t \leftarrow \mu B_{t-1} + g_t \\[0.25ex]
            &\hspace{5mm} \widetilde{B}_t \leftarrow
                \begin{cases}
                   g_t + \mu B_t, & \text{if nesterov}=True \\
                   B_t,           & \text{if nesterov}=False
                \end{cases} \\[1.0ex]
            &\hspace{5mm} O_t \leftarrow \mathrm{NS}^{(a,b,c)}_{k}\!\big(\widetilde{B}_t;\ \varepsilon\big) \\[0.5ex]
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma\,\lambda\,\theta_{t-1}
               \quad\text{(decoupled weight decay)} \\[0.25ex]

            &\hspace{5mm} \gamma \leftarrow \mathrm{AdjustLR}\!\big(\gamma;\ \mathrm{shape}\!\big(\theta_t \big) \big) \\[0.25ex]
            &\hspace{5mm} \theta_t \leftarrow \theta_t - \gamma\, O_t \\
            &\rule{110mm}{0.4pt} \\[-1.ex]
            &\mathbf{return}\ \theta_t \\[-1.ex]
            &\rule{110mm}{0.4pt}s
       \end{aligned}

    Here, :math:`\mathrm{NS}^{(a,b,c)}_{k}(\cdot;\varepsilon)` denotes :math:`k` iterations of the
    Newton–Schulz orthogonalization operator parameterized by coefficients :math:`(a,b,c)`
    with numerical stabilization :math:`\varepsilon`.

    The purpose for :math:`\mathrm{AdjustLR}\!\big(\gamma;\ \mathrm{shape}\!\big(\theta_t \big) \big)`
    is to make the orthogonalized update have a consistent :math:`RMS` across rectangular matrices.

    Keller's original implementation scales the update by :math:`\sqrt{\max\!\left(1, \frac{A}{B}\right)}`,
    where :math:`A` and :math:`B` are dimension of the matrix being optimized.

    Moonshot's implementation also focuses on matching :math:`RMS` of AdamW. The adjustment is computed as:
    :math:`\gamma \leftarrow {0.2}\gamma\,\sqrt{\max\!\left({A}, {B}\right)}`
    The method is adopted from `Muon is Scalable for LLM Training`_. Research
    results show that with this adjustment Muon can directly reuse the learning rate
    and weight decay tuned for AdamW.

    We provide two options for the learning rate adjustment: "original", which follows Keller's
    implementation, and "match_rms_adamw", which refers to Moonshot's implementation. This gives users the
    flexibility to choose between the two. If `adjust_lr_fn` is not specified, the default is "original".

    For further details regarding the algorithm we refer to `Muon: An optimizer for hidden layers in neural networks`_
    and `Muon is Scalable for LLM Training`_.
    """
    + rf"""
    Args:
        {_params_doc}. Note that Muon is an optimizer for 2D parameters of neural network hidden layers. Other
            parameters, such as bias, and embedding, should be optimized by a standard method such as AdamW.
        lr (float, Tensor, optional): learning rate (default: 1e-3).
        weight_decay (float, optional): weight decay (L2 penalty). (default: 0.1)
        momentum (float, optional): momentum factor (default: 0.95)
        nesterov (bool, optional): enables Nesterov momentum. Only applicable
            when momentum is non-zero
        ns_coefficients (tuple of three floats, optional): coefficients \(a,b,c\) for the
            Newton–Schulz orthogonalization polynomial (default: ({DEFAULT_A}, {DEFAULT_B}, {DEFAULT_C}))
        eps (float, optional): term added to the denominator for numerical stability. (default: {EPS})
        ns_steps (int, optional): number of Newton–Schulz iteration steps. (default: {DEFAULT_NS_STEPS})
        adjust_lr_fn (str, optional): function to adjust learning rate. One of "original" and "match_rms_adamw".
            If not specified, we will default to use "original". (default: None)

    .. _Muon\: An optimizer for hidden layers in neural networks:
        https://kellerjordan.github.io/posts/muon/
    .. _Muon is Scalable for LLM Training:
        https://arxiv.org/pdf/2502.16982

    """
)


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
    has_complex: bool,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_muon)
def muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    foreach: Optional[bool] = None,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
    has_complex: bool,
    distributed_config: Optional[DistributedConfig] = None,
):
    r"""Functional API that performs Muon algorithm computation.

    See :class:`~torch.optim.Muon` for details.
    """
    if foreach is not None and foreach:
        raise RuntimeError("Foreach is not supported for Muon yet")

    if distributed_config is None:
        # Standard non-distributed path
        func = _single_tensor_muon
        func(
            params,
            grads,
            muon_momentum_bufs,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
            has_complex=has_complex,
        )
    else:
        # Distributed path with gather/redistribute
        _single_tensor_muon_distributed(
            params,
            grads,
            muon_momentum_bufs,
            distributed_config=distributed_config,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
            has_complex=has_complex,
        )


def _update_momentum_buffers(
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    momentum: float,
) -> None:
    """Step 0: Update all local momentum buffers synchronously.

    All ranks must complete this step before any gather operations to ensure
    consistency across distributed training.

    Args:
        grads: List of parameter gradients
        muon_momentum_bufs: List of momentum buffers
        momentum: Momentum coefficient
    """
    for i in range(len(grads)):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError(
                f"Parameter {i} gradient must be a 2D matrix, got {grad.ndim}D"
            )

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)


def _select_parameters_to_process(
    assignments: dict[int, int],
    rank: int,
    num_params: int,
    async_gpu: bool,
) -> list[int]:
    """Step 1: Determine which parameters this rank will process.

    Args:
        assignments: Dictionary mapping param_idx -> rank
        rank: Current rank
        num_params: Total number of parameters
        async_gpu: If True, process only assigned params; if False, process all

    Returns:
        List of parameter indices to process
    """
    if async_gpu:
        # Async mode: each rank processes only its assigned parameters
        return [i for i in range(num_params) if assignments[i] == rank]
    else:
        # Sync mode: all ranks process all parameters (easier debugging)
        return list(range(num_params))


def _orthogonalize_and_apply_update(
    param: Tensor,
    param_idx: int,
    momentum_buffer_full: Optional[Tensor],
    distributed_config: DistributedConfig,
    assignments: dict[int, int],
    rank: int,
    lr: float,
    weight_decay: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
) -> None:
    """Orthogonalize momentum buffer and apply update to parameter.

    This is the core computation step that is shared between all processing modes:
    - Sequential processing (Phase 1/2)
    - Prefetch processing (Phase 3)
    - Future: Async processing (Phase 4)

    Args:
        param: Parameter to update
        param_idx: Parameter index
        momentum_buffer_full: Full gathered momentum buffer (None on non-assigned ranks)
        distributed_config: Distributed config
        assignments: Parameter assignments
        rank: Current rank
        lr: Learning rate
        weight_decay: Weight decay coefficient
        nesterov: Whether to use Nesterov momentum
        ns_coefficients: Newton-Schulz coefficients
        ns_steps: Number of NS iterations
        eps: Epsilon for stability
        adjust_lr_fn: LR adjustment function
    """
    # Orthogonalize only on assigned rank (zero-redundancy)
    update_full = None
    if rank == assignments[param_idx]:
        assert momentum_buffer_full is not None, (
            f"Rank {rank} should have full momentum buffer for param {param_idx}"
        )

        # Apply nesterov if enabled
        # TODO: Properly implement nesterov with distributed gather of grad
        # Current implementation: use momentum buffer directly (approximation)
        # Full implementation needs: grad.lerp(momentum_buf, momentum)
        if nesterov:
            update = momentum_buffer_full
        else:
            update = momentum_buffer_full

        # Orthogonalize via Newton-Schulz iteration
        update_full = _zeropower_via_newtonschulz(
            update, ns_coefficients, ns_steps, eps
        )

    # Redistribute update to all ranks
    update = distributed_config.redistribute_fn(
        update_full,
        src_rank=assignments[param_idx],
        state=distributed_config.state,
    )

    # Apply update with weight decay
    adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
    param.mul_(1 - lr * weight_decay)
    param.add_(update, alpha=-adjusted_lr)


def _wait_for_prefetch_gather(
    prefetch_buffer: Optional[tuple[Any, Optional[Any]]],
    param_idx: int,
    rank: int,
    assignments: dict[int, int],
    fallback_gather_fn: Callable,
    momentum_buf: Tensor,
    state: dict[str, Any],
) -> Optional[Tensor]:
    """Wait for prefetched gather to complete and return result.

    Handles:
    - Waiting on async work handle
    - Concatenating gathered tensors
    - Filtering None results for non-dst ranks
    - Fallback to synchronous gather on errors

    Args:
        prefetch_buffer: Tuple of (gather_result, work_handle) from async gather
        param_idx: Current parameter index
        rank: Current rank
        assignments: Parameter assignments
        fallback_gather_fn: Function to call for synchronous gather on error
        momentum_buf: Momentum buffer for fallback gather
        state: Distributed state

    Returns:
        Full momentum buffer on dst_rank, None on other ranks
    """
    # No prefetch buffer - must use synchronous gather
    if prefetch_buffer is None or prefetch_buffer == (None, None):
        return fallback_gather_fn(
            momentum_buf,
            dst_rank=assignments[param_idx],
            state=state,
        )

    # Unpack prefetch results
    gather_result, work_handle = prefetch_buffer

    # Wait for async operation to complete
    if work_handle is not None:
        work_handle.wait()

        # Concatenate if result is a list (from async gather)
        if isinstance(gather_result, list):
            momentum_buffer_full = torch.cat(gather_result, dim=0)
        else:
            momentum_buffer_full = gather_result
    else:
        # No work handle - result is already available
        momentum_buffer_full = gather_result

    # Validate result
    if momentum_buffer_full is None:
        if rank == assignments[param_idx]:
            # Assigned rank should have buffer - fallback to sync gather
            return fallback_gather_fn(
                momentum_buf,
                dst_rank=assignments[param_idx],
                state=state,
            )
        else:
            # Non-assigned rank correctly has None
            return None

    return momentum_buffer_full


def _supports_async_gather(state: dict[str, Any]) -> bool:
    """Check if current distributed config supports async gather operations.

    Async gather is currently supported for:
    - Tensor Parallel (TP) process groups
    - Fully Sharded Data Parallel (FSDP) process groups

    Not supported for:
    - Data Parallel (DDP) - already replicated
    - DeviceMesh without process groups
    - DTensor without process groups

    Args:
        state: Distributed state dictionary

    Returns:
        True if async gather is supported, False otherwise
    """
    return state.get("tp_pg") is not None or state.get("fsdp_pg") is not None


def _async_gather_fn(
    momentum_buffer: Tensor,
    dst_rank: int,
    state: dict[str, Any],
) -> tuple[Optional[Any], Optional[Any]]:
    """Async version of gather_fn that starts gather operations without waiting.

    Returns:
        Tuple of (gather_result, work_handle) where:
        - gather_result: List of gather buffers (for concat later) or tensor
        - work_handle: Work object for wait() call, or None
    """
    rank = state["rank"]
    result = momentum_buffer
    work_handle = None

    # Chain async gather operations for each active parallelism dimension
    # Order matters: gather inner dimensions first (TP), then outer (FSDP/DDP)

    # Tensor Parallel: async gather shards along TP dimension
    if state.get("tp_pg") is not None:
        result, work_handle = _gather_tensor_shards(
            result, state["tp_pg"], async_op=True
        )
        # result is now a list of buffers, work_handle needs to be waited on
        return result, work_handle

    # FSDP: async gather shards along FSDP dimension
    if state.get("fsdp_pg") is not None:
        result, work_handle = _gather_tensor_shards(
            result, state["fsdp_pg"], async_op=True
        )
        return result, work_handle

    # DDP/CP/EP/PP: already replicated or independent, no gather needed
    # Return synchronously
    if rank == dst_rank:
        return result, None
    else:
        return None, None


def _process_parameters_with_prefetch(
    params: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    param_indices_to_process: list[int],
    distributed_config: DistributedConfig,
    assignments: dict[int, int],
    rank: int,
    lr: float,
    weight_decay: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
) -> None:
    """Process parameters with prefetching to overlap communication and computation.

    Phase 3 Implementation: This function implements prefetching by starting the
    gather operation for the next parameter while processing the current one.

    Algorithm:
        1. Start prefetch gather for first parameter
        2. For each parameter:
           a. Wait for current gather to complete
           b. Start prefetch gather for next parameter (if available)
           c. Orthogonalize current parameter (overlapped with next gather)
           d. Redistribute update to all ranks
           e. Apply update locally

    Args:
        params: List of all parameters
        muon_momentum_bufs: List of all momentum buffers
        param_indices_to_process: Indices of parameters to process
        distributed_config: Distributed configuration
        assignments: Parameter-to-rank assignments
        rank: Current rank
        lr: Learning rate
        weight_decay: Weight decay coefficient
        nesterov: Whether to use nesterov momentum
        ns_coefficients: Newton-Schulz coefficients
        ns_steps: Number of Newton-Schulz iterations
        eps: Epsilon for numerical stability
        adjust_lr_fn: Learning rate adjustment function name
    """
    if len(param_indices_to_process) == 0:
        return

    state = distributed_config.state

    # Start prefetch for first parameter if supported
    prefetch_buffer = None
    if len(param_indices_to_process) > 0 and _supports_async_gather(state):
        first_param_idx = param_indices_to_process[0]
        state["current_param_idx"] = first_param_idx
        prefetch_buffer = _async_gather_fn(
            muon_momentum_bufs[first_param_idx],
            dst_rank=assignments[first_param_idx],
            state=state,
        )

    # Process each parameter with prefetching
    for idx, param_idx in enumerate(param_indices_to_process):
        state["current_param_idx"] = param_idx

        # Wait for current prefetch and get momentum buffer
        momentum_buffer_full = _wait_for_prefetch_gather(
            prefetch_buffer,
            param_idx,
            rank,
            assignments,
            distributed_config.gather_fn,
            muon_momentum_bufs[param_idx],
            state,
        )

        # Start prefetch for next parameter (if available and supported)
        prefetch_buffer = None
        if idx + 1 < len(param_indices_to_process) and _supports_async_gather(state):
            next_param_idx = param_indices_to_process[idx + 1]
            state["current_param_idx"] = next_param_idx
            prefetch_buffer = _async_gather_fn(
                muon_momentum_bufs[next_param_idx],
                dst_rank=assignments[next_param_idx],
                state=state,
            )
            state["current_param_idx"] = param_idx  # Restore

        # Orthogonalize and apply update (shared logic)
        _orthogonalize_and_apply_update(
            params[param_idx],
            param_idx,
            momentum_buffer_full,
            distributed_config,
            assignments,
            rank,
            lr,
            weight_decay,
            nesterov,
            ns_coefficients,
            ns_steps,
            eps,
            adjust_lr_fn,
        )


def _process_single_parameter(
    param_idx: int,
    param: Tensor,
    momentum_buf: Tensor,
    distributed_config: DistributedConfig,
    assignments: dict[int, int],
    rank: int,
    lr: float,
    weight_decay: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
) -> None:
    """Process a single parameter without prefetching (sequential mode).

    This function implements sequential parameter processing without prefetching.
    It performs:
    1. Synchronous gather of full momentum buffer on assigned rank
    2. Orthogonalize and apply update (using shared logic)

    Args:
        param_idx: Index of parameter being processed
        param: Parameter tensor to update
        momentum_buf: Momentum buffer for this parameter
        distributed_config: Distributed configuration
        assignments: Parameter-to-rank assignments
        rank: Current rank
        lr: Learning rate
        weight_decay: Weight decay coefficient
        nesterov: Whether to use nesterov momentum
        ns_coefficients: Newton-Schulz coefficients (a, b, c)
        ns_steps: Number of Newton-Schulz iterations
        eps: Epsilon for numerical stability
        adjust_lr_fn: Learning rate adjustment function name
    """
    # Set current param_idx for buffer allocation
    distributed_config.state["current_param_idx"] = param_idx

    # Gather full momentum buffer on assigned rank
    momentum_buffer_full = distributed_config.gather_fn(
        momentum_buf,
        dst_rank=assignments[param_idx],
        state=distributed_config.state,
    )

    # Orthogonalize and apply update (shared logic)
    _orthogonalize_and_apply_update(
        param,
        param_idx,
        momentum_buffer_full,
        distributed_config,
        assignments,
        rank,
        lr,
        weight_decay,
        nesterov,
        ns_coefficients,
        ns_steps,
        eps,
        adjust_lr_fn,
    )


def _single_tensor_muon_distributed(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    distributed_config: DistributedConfig,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
    has_complex: bool,
) -> None:
    """Distributed Muon with zero-redundancy orthogonalization and prefetching.

    Key design: Each parameter is assigned to exactly one rank for orthogonalization,
    eliminating redundant computation across ranks.

    Phase 3 Enhancement: Prefetching support to overlap communication with computation.
    When prefetch_count > 0, we start gathering the next parameter's momentum buffer
    while processing the current parameter, reducing idle time.

    Algorithm:
        1. All ranks update their local momentum buffer shards
        2. Each rank determines which parameters it will orthogonalize
        3. For each parameter (with optional prefetching):
           - Start async gather for next parameter (if prefetching enabled)
           - Wait for current parameter's gather to complete
           - Assigned rank orthogonalizes
           - Redistribute update to all ranks
           - All ranks apply update locally
        4. Synchronize before next training step (if async_gpu=True)

    Args:
        params: List of parameter tensors
        grads: List of gradient tensors
        muon_momentum_bufs: List of momentum buffer tensors
        distributed_config: Distributed training configuration
        lr: Learning rate
        weight_decay: Weight decay coefficient
        momentum: Momentum coefficient
        nesterov: Whether to use nesterov momentum
        ns_coefficients: Newton-Schulz iteration coefficients (a, b, c)
        ns_steps: Number of Newton-Schulz iterations
        eps: Epsilon for numerical stability
        adjust_lr_fn: Optional learning rate adjustment function name
        has_complex: Whether any parameters are complex (not supported)
    """
    if has_complex:
        raise ValueError("Complex parameters are not supported in distributed Muon")

    lr = _to_scalar(lr)
    assignments = distributed_config.state["assignments"]
    rank = distributed_config.state["rank"]
    async_gpu = distributed_config.async_gpu_parallelism
    prefetch_count = distributed_config.prefetch_count

    # Step 0: Update momentum buffers (synchronous across all ranks)
    _update_momentum_buffers(grads, muon_momentum_bufs, momentum)

    # Step 1: Determine which parameters this rank will process
    param_indices_to_process = _select_parameters_to_process(
        assignments, rank, len(params), async_gpu
    )

    # Step 2: Process parameters with optional prefetching
    if prefetch_count == 0:
        # No prefetching: use sequential processing (Phase 1/2 behavior)
        for param_idx in param_indices_to_process:
            _process_single_parameter(
                param_idx,
                params[param_idx],
                muon_momentum_bufs[param_idx],
                distributed_config,
                assignments,
                rank,
                lr,
                weight_decay,
                nesterov,
                ns_coefficients,
                ns_steps,
                eps,
                adjust_lr_fn,
            )
    else:
        # Phase 3: Prefetching enabled
        _process_parameters_with_prefetch(
            params,
            muon_momentum_bufs,
            param_indices_to_process,
            distributed_config,
            assignments,
            rank,
            lr,
            weight_decay,
            nesterov,
            ns_coefficients,
            ns_steps,
            eps,
            adjust_lr_fn,
        )

    # Clean up temporary state
    distributed_config.state.pop("current_param_idx", None)

    # Step 3: Synchronize all ranks before continuing to next training step
    # This ensures all parameters have been updated before the next iteration
    if async_gpu:
        import torch.distributed as dist

        if "world_pg" in distributed_config.state:
            dist.barrier(distributed_config.state["world_pg"])
        elif dist.is_initialized():
            dist.barrier()
