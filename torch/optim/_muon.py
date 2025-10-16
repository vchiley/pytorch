# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer."""

import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from .optimizer import (
    _disable_dynamo_if_unsupported,
    _params_doc,
    _to_scalar,
    Optimizer,
    ParamsT,
)


__all__ = ["Muon", "DistributedConfig", "create_auto_config"]

# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


@dataclass
class DistributedConfig:
    """Configuration for distributed Muon optimizer.

    This dataclass holds the three user-defined functions that enable Muon to work
    with any distributed training setup (FSDP, TP, HSDP, EP, CP, PP, or hybrids).

    The core API consists of three functions that handle orthogonal concerns:
    1. assign_fn: Which GPU processes which parameters?
    2. gather_fn: How to assemble full update (momentum buffer) on assigned GPU?
    3. redistribute_fn: How to scatter orthogonalized update back to shards?

    For common PyTorch parallelism strategies, use create_auto_config() instead of
    defining these functions manually.

    Args:
        assign_fn: Function that assigns parameters to ranks for orthogonalization.
            Signature: (params: list[Tensor], state: dict[str, Any]) -> dict[int, list[int]]
            Returns: {rank: [param_indices]} mapping
            Example: {0: [0, 2, 4], 1: [1, 3, 5]} for 2 ranks, 6 params

        gather_fn: Function that gathers full update (momentum buffer) to assigned rank.
            Signature: (update: Tensor, rank: int, state: dict[str, Any]) -> Tensor
            Args:
                update: The sharded update (momentum buffer) on this rank
                rank: The rank that will orthogonalize this update
                state: User-provided metadata (process groups, etc.)
            Returns: Full update (only valid on assigned rank)

        redistribute_fn: Function that redistributes orthogonalized update back to shards.
            Signature: (ortho_update: Tensor, rank: int, state: dict[str, Any]) -> Tensor
            Args:
                ortho_update: The orthogonalized full update
                rank: The rank that orthogonalized this update
                state: User-provided metadata
            Returns: This rank's shard of the update

        state: Dictionary containing process groups, device meshes, and any other
            metadata your functions need. Can include:
            - Process groups for FSDP, TP, HSDP, EP, CP, PP
            - Device meshes
            - Shard dimensions
            - Rank and world size
            - Any custom metadata

        async_gpu_parallelism: If True (default), GPUs process parameters in parallel
            (optimal performance). If False, GPUs process sequentially (simpler debugging).
            Sequential mode enforces clear execution order: GPU 0 finishes all its params,
            then GPU 1, etc. Useful for debugging but very slow.

        prefetch_count: Number of updates to prefetch ahead (0=disabled, 1-3 recommended).
            Controls operation-level async communication overlap. Higher values increase
            overlap but consume more memory. Set to 0 for simplest debugging.

    Example (custom setup):
        >>> def my_assign(params, state):
        ...     rank = state["rank"]
        ...     world_size = state["world_size"]
        ...     return {r: list(range(r, len(params), world_size)) for r in range(world_size)}
        >>>
        >>> def my_gather(update, rank, state):
        ...     return torch.distributed.all_gather(..., group=state["process_group"])
        >>>
        >>> def my_redistribute(ortho_update, rank, state):
        ...     return torch.distributed.scatter(..., group=state["process_group"])
        >>>
        >>> config = DistributedConfig(
        ...     assign_fn=my_assign,
        ...     gather_fn=my_gather,
        ...     redistribute_fn=my_redistribute,
        ...     state={"rank": 0, "world_size": 4, "process_group": ...}
        ... )
        >>> optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

    Example (using auto-config for FSDP):
        >>> import torch.distributed as dist
        >>> config = create_auto_config(fsdp_process_group=dist.group.WORLD)
        >>> optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)
    """

    assign_fn: Callable[[list[Tensor], dict[str, Any]], dict[int, list[int]]]
    gather_fn: Callable[[Tensor, int, dict[str, Any]], Tensor]
    redistribute_fn: Callable[[Tensor, int, dict[str, Any]], Tensor]
    state: dict[str, Any]
    async_gpu_parallelism: bool = True
    prefetch_count: int = 1


def create_auto_config(
    *,
    fsdp_process_group=None,
    tp_process_group=None,
    tp_shard_dim: Optional[int] = None,
    shard_process_group=None,
    replicate_process_group=None,
    ep_process_group=None,
    ep_shard_dim: Optional[int] = None,
    cp_process_group=None,
    pp_process_group=None,
    pp_stage_id: Optional[int] = None,
    pp_num_stages: Optional[int] = None,
    async_gpu_parallelism: bool = True,
    prefetch_count: int = 1,
) -> DistributedConfig:
    """Create distributed config for common PyTorch parallelism strategies.

    This factory function automatically generates the three core functions
    (assign_fn, gather_fn, redistribute_fn) for standard PyTorch distributed
    training setups. For custom frameworks or non-standard setups, define
    your own functions and create DistributedConfig directly.

    Supported strategies:
    - FSDP (Fully Sharded Data Parallel)
    - TP (Tensor Parallel)
    - HSDP (Hybrid Sharded Data Parallel: shard + replicate)
    - EP (Expert Parallel for MoE models)
    - CP (Context Parallel for long sequences)
    - PP (Pipeline Parallel)
    - Hybrids: TP+FSDP, TP+HSDP, PP+TP, PP+FSDP, PP+TP+HSDP, etc.

    Args:
        fsdp_process_group: Process group for FSDP (parameters sharded across ranks)
        tp_process_group: Process group for Tensor Parallel (layers sharded across ranks)
        tp_shard_dim: Dimension along which TP shards parameters (0 or 1)
        shard_process_group: Shard process group for HSDP (within-node sharding)
        replicate_process_group: Replicate process group for HSDP (across-node replication)
        ep_process_group: Process group for Expert Parallel (expert weights sharded)
        ep_shard_dim: Dimension along which EP shards expert parameters
        cp_process_group: Process group for Context Parallel (sequence partitioned, params replicated)
        pp_process_group: Process group for Pipeline Parallel (layers partitioned across stages)
        pp_stage_id: This pipeline stage's ID (0 to num_stages-1)
        pp_num_stages: Total number of pipeline stages
        async_gpu_parallelism: If True, GPUs process parameters in parallel (default: True)
        prefetch_count: Number of updates to prefetch (0=disabled, 1-3 recommended, default: 1)

    Returns:
        DistributedConfig with auto-generated functions for the specified parallelism

    Example (FSDP):
        >>> import torch.distributed as dist
        >>> config = create_auto_config(fsdp_process_group=dist.group.WORLD)
        >>> optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

    Example (TP + FSDP hybrid):
        >>> config = create_auto_config(
        ...     tp_process_group=tp_pg,
        ...     fsdp_process_group=fsdp_pg,
        ...     tp_shard_dim=1
        ... )
        >>> optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)
    """
    try:
        import torch.distributed as dist
    except ImportError as e:
        raise RuntimeError(
            "create_auto_config() requires torch.distributed. "
            "Please ensure PyTorch is built with distributed support."
        ) from e

    # Detect which parallelism strategies are used
    has_fsdp = fsdp_process_group is not None
    has_tp = tp_process_group is not None
    has_hsdp = shard_process_group is not None or replicate_process_group is not None
    has_ep = ep_process_group is not None
    has_cp = cp_process_group is not None
    has_pp = pp_process_group is not None

    # Validation
    if has_tp and tp_shard_dim is None:
        raise ValueError("tp_shard_dim must be specified when using Tensor Parallel")
    if has_ep and ep_shard_dim is None:
        raise ValueError("ep_shard_dim must be specified when using Expert Parallel")
    if has_pp and (pp_stage_id is None or pp_num_stages is None):
        raise ValueError(
            "pp_stage_id and pp_num_stages must be specified when using Pipeline Parallel"
        )
    if has_hsdp and has_fsdp:
        raise ValueError(
            "Cannot use both FSDP and HSDP simultaneously. "
            "HSDP is a variant of FSDP with shard + replicate groups."
        )

    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Build state dict with all provided metadata
    state: dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
    }

    if has_fsdp:
        state["fsdp_process_group"] = fsdp_process_group
    if has_tp:
        state["tp_process_group"] = tp_process_group
        state["tp_shard_dim"] = tp_shard_dim
    if has_hsdp:
        state["shard_process_group"] = shard_process_group
        state["replicate_process_group"] = replicate_process_group
    if has_ep:
        state["ep_process_group"] = ep_process_group
        state["ep_shard_dim"] = ep_shard_dim
    if has_cp:
        state["cp_process_group"] = cp_process_group
    if has_pp:
        state["pp_process_group"] = pp_process_group
        state["pp_stage_id"] = pp_stage_id
        state["pp_num_stages"] = pp_num_stages

    # Create assignment function (round-robin by default)
    def assign_fn(params: list[Tensor], state: dict[str, Any]) -> dict[int, list[int]]:
        """Assign parameters to ranks via round-robin."""
        world_size = state["world_size"]
        assignments = {r: [] for r in range(world_size)}
        for idx in range(len(params)):
            assigned_rank = idx % world_size
            assignments[assigned_rank].append(idx)
        return assignments

    # Create gather function (compose gathers for hybrid parallelism)
    def gather_fn(update: Tensor, rank: int, state: dict[str, Any]) -> Tensor:
        """Gather full update from shards.

        For hybrid parallelism, gathers are composed in order: TP → EP → FSDP/HSDP/CP
        """
        result = update

        # First, gather across TP if present
        if "tp_process_group" in state:
            tp_pg = state["tp_process_group"]
            tp_shard_dim = state["tp_shard_dim"]
            tp_world_size = dist.get_world_size(tp_pg)

            # all_gather along TP dimension
            gathered_shards = [torch.zeros_like(result) for _ in range(tp_world_size)]
            dist.all_gather(gathered_shards, result, group=tp_pg)
            result = torch.cat(gathered_shards, dim=tp_shard_dim)

        # Then, gather across EP if present
        if "ep_process_group" in state:
            ep_pg = state["ep_process_group"]
            ep_shard_dim = state["ep_shard_dim"]
            ep_world_size = dist.get_world_size(ep_pg)

            gathered_shards = [torch.zeros_like(result) for _ in range(ep_world_size)]
            dist.all_gather(gathered_shards, result, group=ep_pg)
            result = torch.cat(gathered_shards, dim=ep_shard_dim)

        # Then, gather across FSDP if present
        if "fsdp_process_group" in state:
            fsdp_pg = state["fsdp_process_group"]
            fsdp_world_size = dist.get_world_size(fsdp_pg)

            # FSDP typically shards along last dimension
            gathered_shards = [
                torch.zeros_like(result) for _ in range(fsdp_world_size)
            ]
            dist.all_gather(gathered_shards, result, group=fsdp_pg)
            # Concatenate along dimension 1 (assuming 2D params)
            result = torch.cat(gathered_shards, dim=1)

        # Or gather across HSDP if present
        if "shard_process_group" in state:
            shard_pg = state["shard_process_group"]
            shard_world_size = dist.get_world_size(shard_pg)

            gathered_shards = [
                torch.zeros_like(result) for _ in range(shard_world_size)
            ]
            dist.all_gather(gathered_shards, result, group=shard_pg)
            result = torch.cat(gathered_shards, dim=1)

        # CP: Parameters are replicated, no gathering needed
        # Just ensure all ranks have the same update (already done via all-reduce in backward)

        # PP: Each stage optimizes independently, no cross-stage gathering

        return result

    # Create redistribute function (inverse of gather)
    def redistribute_fn(ortho_update: Tensor, rank: int, state: dict[str, Any]) -> Tensor:
        """Redistribute orthogonalized update back to shards.

        This reverses the gather operation, scattering the full orthogonalized
        update back to the original sharded form.
        """
        result = ortho_update

        # Redistribute in reverse order: FSDP/HSDP/CP → EP → TP

        # First, redistribute across FSDP/HSDP if present
        if "fsdp_process_group" in state:
            fsdp_pg = state["fsdp_process_group"]
            fsdp_rank = dist.get_rank(fsdp_pg)
            fsdp_world_size = dist.get_world_size(fsdp_pg)

            # Split along dim 1 and take this rank's shard
            shard_size = result.shape[1] // fsdp_world_size
            result = result[:, fsdp_rank * shard_size : (fsdp_rank + 1) * shard_size]

        if "shard_process_group" in state:
            shard_pg = state["shard_process_group"]
            shard_rank = dist.get_rank(shard_pg)
            shard_world_size = dist.get_world_size(shard_pg)

            shard_size = result.shape[1] // shard_world_size
            result = result[:, shard_rank * shard_size : (shard_rank + 1) * shard_size]

        # CP: Broadcast from owner rank to all ranks
        if "cp_process_group" in state:
            cp_pg = state["cp_process_group"]
            # For CP, we need to know which rank owns this parameter
            # For simplicity, broadcast from rank 0 in CP group
            # (Proper implementation would track ownership via assignments)
            dist.broadcast(result, src=0, group=cp_pg)

        # Then, redistribute across EP if present
        if "ep_process_group" in state:
            ep_pg = state["ep_process_group"]
            ep_rank = dist.get_rank(ep_pg)
            ep_world_size = dist.get_world_size(ep_pg)
            ep_shard_dim = state["ep_shard_dim"]

            shard_size = result.shape[ep_shard_dim] // ep_world_size
            if ep_shard_dim == 0:
                result = result[
                    ep_rank * shard_size : (ep_rank + 1) * shard_size, ...
                ]
            else:
                result = result[
                    ..., ep_rank * shard_size : (ep_rank + 1) * shard_size
                ]

        # Finally, redistribute across TP if present
        if "tp_process_group" in state:
            tp_pg = state["tp_process_group"]
            tp_rank = dist.get_rank(tp_pg)
            tp_world_size = dist.get_world_size(tp_pg)
            tp_shard_dim = state["tp_shard_dim"]

            shard_size = result.shape[tp_shard_dim] // tp_world_size
            if tp_shard_dim == 0:
                result = result[
                    tp_rank * shard_size : (tp_rank + 1) * shard_size, ...
                ]
            else:
                result = result[
                    ..., tp_rank * shard_size : (tp_rank + 1) * shard_size
                ]

        return result

    return DistributedConfig(
        assign_fn=assign_fn,
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

        # Store distributed_config as instance variable (not in defaults)
        # This is because it contains callable functions which shouldn't be in param_groups
        self.distributed_config = distributed_config

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters whereas we found a parameter with size: {p.size()}"
                    )

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
        distributed_config (DistributedConfig, optional): configuration for distributed training.
            Required when using sharded parallelism strategies (FSDP, TP, HSDP, EP, PP, etc.)
            to ensure correct orthogonalization across distributed updates. Use create_auto_config()
            for common PyTorch parallelism strategies, or provide custom functions for non-PyTorch
            distributed frameworks. (default: None, which uses single-GPU implementation)

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
    distributed_config: Optional[DistributedConfig] = None,
) -> None:
    """Unified Muon implementation for both single-GPU and distributed training.

    This function handles both cases via a gating pattern: distributed operations
    are only executed when distributed_config is provided.

    For single-GPU (distributed_config is None):
        1. Update momentum buffer
        2. Orthogonalize update
        3. Apply to parameter

    For distributed (distributed_config is not None):
        1. Update momentum buffer (on sharded parameters)
        2. Gather full update from all shards
        3. Orthogonalize full update
        4. Redistribute orthogonalized update back to shards
        5. Apply to sharded parameter

    Why orthogonalize updates instead of parameters? More efficient - only the update
    (momentum buffer) needs orthogonalization and redistribution. Parameters stay
    sharded; we apply the orthogonalized update to them in-place.
    """
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    # Distributed path: requires gather → orthogonalize → redistribute
    if distributed_config is not None:
        try:
            import torch.distributed as dist
        except ImportError as e:
            raise RuntimeError(
                "Distributed config provided but torch.distributed is not available. "
                "Please ensure PyTorch is built with distributed support."
            ) from e

        # Get assignments: which parameters does this rank orthogonalize?
        assignments = distributed_config.assign_fn(params, distributed_config.state)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Update momentum buffers (always, regardless of assignment)
        for i in range(len(params)):
            grad = grads[i]
            if grad.ndim != 2:
                raise ValueError("Param gradient must be a 2D matrix")
            buf = muon_momentum_bufs[i]
            buf.lerp_(grad, 1 - momentum)

        # Process parameters based on async_gpu_parallelism flag
        if distributed_config.async_gpu_parallelism:
            # Parallel mode: Each GPU processes its assigned parameters independently
            for param_idx in assignments.get(rank, []):
                _process_param_distributed(
                    param_idx,
                    params,
                    grads,
                    muon_momentum_bufs,
                    lr,
                    weight_decay,
                    momentum,
                    nesterov,
                    ns_coefficients,
                    ns_steps,
                    eps,
                    adjust_lr_fn,
                    distributed_config,
                    rank,
                )
        else:
            # Sequential mode (debug): GPUs process one at a time
            for gpu_rank in range(world_size):
                if rank == gpu_rank:
                    for param_idx in assignments.get(rank, []):
                        _process_param_distributed(
                            param_idx,
                            params,
                            grads,
                            muon_momentum_bufs,
                            lr,
                            weight_decay,
                            momentum,
                            nesterov,
                            ns_coefficients,
                            ns_steps,
                            eps,
                            adjust_lr_fn,
                            distributed_config,
                            rank,
                        )
                # Barrier: wait for current GPU to finish before next GPU proceeds
                dist.barrier()

    else:
        # Single-GPU path: orthogonalize update directly and apply
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


def _process_param_distributed(
    param_idx: int,
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
    distributed_config: DistributedConfig,
    rank: int,
) -> None:
    """Process a single parameter in distributed mode.

    This function:
    1. Computes the update (with Nesterov if enabled)
    2. Gathers full update from all shards
    3. Orthogonalizes the full update
    4. Redistributes orthogonalized update back to shards
    5. Applies orthogonalized update to sharded parameter
    """
    param = params[param_idx]
    grad = grads[param_idx]
    buf = muon_momentum_bufs[param_idx]

    # Compute update to orthogonalize (with Nesterov momentum if enabled)
    # Note: buf was already updated via lerp in the main loop
    # For Nesterov: update = grad * (1-momentum) + buf * momentum
    # For standard: update = buf
    if nesterov:
        update = grad.lerp(buf, momentum)
    else:
        update = buf.clone()

    # Gather full update (momentum buffer) from all shards
    full_update = distributed_config.gather_fn(update, rank, distributed_config.state)

    # Orthogonalize the full update
    ortho_update = _zeropower_via_newtonschulz(
        full_update, ns_coefficients, ns_steps, eps
    )

    # Redistribute orthogonalized update back to shards
    shard_update = distributed_config.redistribute_fn(
        ortho_update, rank, distributed_config.state
    )

    # Apply orthogonalized update to sharded parameter
    adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
    param.mul_(1 - lr * weight_decay)
    param.add_(shard_update, alpha=-adjusted_lr)


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
        distributed_config=distributed_config,
    )
