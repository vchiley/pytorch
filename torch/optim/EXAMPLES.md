# CLAUDE_EXAMPLES.md - Muon Distributed Training Examples

Complete end-to-end examples for using Muon optimizer with various distributed training strategies.

## Table of Contents
0. [Understanding the Unified Implementation](#understanding-the-unified-implementation)
1. [Single GPU (Baseline)](#single-gpu-baseline)
2. [DDP (No Special Config)](#ddp-no-special-config)
3. [FSDP](#fsdp)
4. [Tensor Parallel](#tensor-parallel)
5. [HSDP (Hybrid Sharded Data Parallel)](#hsdp-hybrid-sharded-data-parallel)
6. [TP + FSDP Hybrid](#tp--fsdp-hybrid)
7. [TP + HSDP Hybrid](#tp--hsdp-hybrid)
8. [Expert Parallel (EP)](#expert-parallel-ep)
9. [Context Parallel (CP)](#context-parallel-cp)
10. [Pipeline Parallel](#pipeline-parallel)
11. [Full 4D Parallelism (PP + TP + HSDP)](#full-4d-parallelism-pp--tp--hsdp)
12. [Custom Distributed Setup](#custom-distributed-setup)
13. [Debug Mode](#debug-mode)
14. [Migration Guide](#migration-guide)
15. [Debugging Tips](#debugging-tips)

---

## Understanding the Unified Implementation

Before diving into examples, it's important to understand that **Muon uses ONE unified implementation** for both single-GPU and distributed training. There are no separate function variants like `_muon_distributed`, `_muon_sequential`, etc.

### How It Works

The implementation uses a gating pattern:

```python
def _single_tensor_muon(..., distributed_config: Optional[DistributedConfig] = None):
    # Update momentum (always happens, regardless of distributed or not)
    for param, grad, buf in zip(params, grads, muon_momentum_bufs):
        buf.lerp_(grad, 1 - momentum)

    # Distributed operations gated behind distributed_config
    if distributed_config is not None:
        # DISTRIBUTED PATH: gather update → orthogonalize → redistribute → apply
        assignments = distributed_config.assign_fn(params, distributed_config.state)

        for param_idx in assignments[rank]:
            # Gather full update (momentum buffer) from all shards
            full_update = distributed_config.gather_fn(buf[param_idx], rank, state)

            # Orthogonalize update
            ortho_update = _zeropower_via_newtonschulz(full_update, ...)

            # Redistribute orthogonalized update back to shards
            shard_update = distributed_config.redistribute_fn(ortho_update, rank, state)

            # Apply orthogonalized update to sharded parameter
            param[param_idx].add_(shard_update, alpha=-lr)
    else:
        # SINGLE-GPU PATH: orthogonalize update and apply
        for param, buf in zip(params, muon_momentum_bufs):
            ortho_update = _zeropower_via_newtonschulz(buf, ...)
            param.add_(ortho_update, alpha=-lr)
```

### Benefits

1. **No function proliferation**: Single implementation handles all cases
2. **Maintainability**: One code path to test and debug
3. **Flexibility**: `async_gpu_parallelism` and `prefetch_count` naturally live in the config
4. **Simplicity**: Clear separation between core algorithm and distributed logic

### What This Means for Users

- **Single-GPU**: Just use `Muon(params, lr=1e-3)` — no distributed config needed
- **Distributed**: Add `distributed_config=create_auto_config(...)` — same core optimizer
- **Custom setups**: Define your own `gather_fn`/`redistribute_fn` — still the same unified function

The examples below show how to configure the optimizer for different parallelism strategies, but remember: **it's always the same underlying implementation**.

---

## Single GPU (Baseline)

No distributed configuration needed.

```python
import torch
from torch.optim.muon import Muon

# Model and data
model = MyTransformer().cuda()
dataloader = get_dataloader()

# Optimizer (no distributed config)
optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.95)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
```

---

## DDP (No Special Config)

DDP replicates full model on each GPU, so no special gathering needed.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.muon import Muon

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Model wrapped in DDP
model = MyTransformer().to(device)
model = DDP(model, device_ids=[rank])

# Optimizer (no distributed config needed!)
optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.95)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()  # DDP automatically all-reduces gradients
        optimizer.step()  # Each GPU has full model, orthogonalize independently
```

**Key point**: DDP maintains full model copy on each GPU, so Muon works without distributed config. However, you can optionally use `distributed_config` to coordinate work and avoid redundant orthogonalization (similar to CP):

```python
# Optional: Use distributed config with DDP for efficiency
config = create_auto_config(
    cp_process_group=dist.group.WORLD  # Treat DDP like CP
)
optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

# Now each GPU orthogonalizes subset of params, then broadcasts results
# Same workflow as CP: gradients all-reduced, then split orthogonalization work
```

---

## FSDP

Fully Sharded Data Parallel shards model parameters across GPUs.

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.muon import Muon, create_auto_config

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Model wrapped in FSDP
model = MyTransformer().to(device)
model = FSDP(model)

# Create distributed config for FSDP
config = create_auto_config(fsdp_process_group=dist.group.WORLD)

# Optimizer with distributed config
optimizer = Muon(
    model.parameters(),
    lr=1e-3,
    momentum=0.95,
    distributed_config=config
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()  # Automatically gathers/redistributes for orthogonalization
```

### FSDP with Custom Sharding Strategy

```python
from torch.distributed.fsdp import ShardingStrategy

# Use FULL_SHARD for maximum memory savings
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=device
)

# Same distributed config works
config = create_auto_config(fsdp_process_group=dist.group.WORLD)
optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)
```

---

## Tensor Parallel

Tensor Parallel shards individual layers across GPUs (e.g., column or row parallel).

```python
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.optim.muon import Muon, create_auto_config

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Create device mesh for TP
device_mesh = DeviceMesh("cuda", list(range(world_size)))

# Model with tensor parallel layers
model = MyTransformer()

# Parallelize specific layers
tp_plan = {
    "attention.qkv_proj": ColwiseParallel(),  # Shard along dim 1
    "attention.out_proj": RowwiseParallel(),  # Shard along dim 0
    "ffn.w1": ColwiseParallel(),
    "ffn.w2": RowwiseParallel(),
}
model = parallelize_module(model, device_mesh, tp_plan)

# Create distributed config for TP
# Note: tp_shard_dim depends on which dimension is sharded (ColwiseParallel = dim 1)
config = create_auto_config(
    tp_process_group=dist.group.WORLD,
    tp_shard_dim=1  # ColwiseParallel shards dimension 1
)

# Optimizer with distributed config
optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

# Training loop (same as before)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
```

### Mixed Sharding Dimensions

```python
# Some layers shard dim 0, others shard dim 1
# Solution: Group parameters by shard dimension

params_dim0 = [p for name, p in model.named_parameters() if "out_proj" in name or "w2" in name]
params_dim1 = [p for name, p in model.named_parameters() if "qkv_proj" in name or "w1" in name]

config_dim0 = create_auto_config(tp_process_group=dist.group.WORLD, tp_shard_dim=0)
config_dim1 = create_auto_config(tp_process_group=dist.group.WORLD, tp_shard_dim=1)

optimizer_dim0 = Muon(params_dim0, lr=1e-3, distributed_config=config_dim0)
optimizer_dim1 = Muon(params_dim1, lr=1e-3, distributed_config=config_dim1)

# In training loop, call both
optimizer_dim0.step()
optimizer_dim1.step()
```

---

## HSDP (Hybrid Sharded Data Parallel)

HSDP combines sharding within nodes and replication across nodes for better network utilization.

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.optim.muon import Muon, create_auto_config

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = rank % 8  # Assuming 8 GPUs per node

# Create process groups for HSDP
# Shard group: GPUs within the same node
# Replicate group: Corresponding GPUs across nodes
num_gpus_per_node = 8
shard_group_ranks = list(range(rank // num_gpus_per_node * num_gpus_per_node,
                                (rank // num_gpus_per_node + 1) * num_gpus_per_node))
replicate_group_ranks = list(range(local_rank, world_size, num_gpus_per_node))

shard_pg = dist.new_group(shard_group_ranks)
replicate_pg = dist.new_group(replicate_group_ranks)

# Model wrapped in FSDP with HSDP
model = MyTransformer().cuda()
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_group=(shard_pg, replicate_pg),  # Tuple for HSDP
)

# Create distributed config for HSDP
config = create_auto_config(
    shard_process_group=shard_pg,
    replicate_process_group=replicate_pg
)

# Optimizer
optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

# Training loop (same as before)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
```

---

## TP + FSDP Hybrid

Combine Tensor Parallel (within node) with FSDP (across nodes) for 2D parallelism.

```python
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.muon import Muon, create_auto_config

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Assuming: 2 nodes, 8 GPUs per node = 16 GPUs total
# TP within node (8 GPUs), FSDP across nodes (2 groups)
tp_size = 8
fsdp_size = world_size // tp_size

# Create process groups
tp_rank = rank % tp_size
fsdp_rank = rank // tp_size

tp_group_ranks = list(range(fsdp_rank * tp_size, (fsdp_rank + 1) * tp_size))
fsdp_group_ranks = list(range(tp_rank, world_size, tp_size))

tp_pg = dist.new_group(tp_group_ranks)
fsdp_pg = dist.new_group(fsdp_group_ranks)

# Create device mesh for TP
device_mesh = DeviceMesh("cuda", list(range(tp_size)))

# Model: First apply TP, then wrap in FSDP
model = MyTransformer()
model = parallelize_module(model, device_mesh, {
    "attention.qkv_proj": ColwiseParallel(),
    "ffn.w1": ColwiseParallel(),
})
model = FSDP(model, process_group=fsdp_pg)

# Create distributed config for TP + FSDP
config = create_auto_config(
    tp_process_group=tp_pg,
    fsdp_process_group=fsdp_pg,
    tp_shard_dim=1
)

# Optimizer
optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()  # Gathers across TP, then FSDP, then redistributes
```

---

## TP + HSDP Hybrid

Three-way parallelism: Tensor Parallel + Shard + Replicate.

```python
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.optim.muon import Muon, create_auto_config

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Topology: 4 nodes × 8 GPUs = 32 GPUs
# TP size: 2 (pairs of GPUs)
# Shard size: 4 (groups within node)
# Replicate size: 4 (across nodes)

tp_size = 2
shard_size = 4
replicate_size = world_size // (tp_size * shard_size)

# Calculate ranks for each dimension
tp_rank = rank % tp_size
shard_rank = (rank // tp_size) % shard_size
replicate_rank = rank // (tp_size * shard_size)

# Create process groups
# TP: Adjacent GPUs (0-1, 2-3, 4-5, ...)
tp_group_ranks = [(rank // tp_size) * tp_size + i for i in range(tp_size)]
tp_pg = dist.new_group(tp_group_ranks)

# Shard: Within node (0,2,4,6, 1,3,5,7, ...)
base = (rank // 8) * 8
shard_group_ranks = [base + tp_rank + i * tp_size for i in range(shard_size)]
shard_pg = dist.new_group(shard_group_ranks)

# Replicate: Across nodes (0,8,16,24, 1,9,17,25, ...)
local_id = rank % 8
replicate_group_ranks = [local_id + i * 8 for i in range(replicate_size)]
replicate_pg = dist.new_group(replicate_group_ranks)

# Model: TP first, then FSDP with HSDP
device_mesh = DeviceMesh("cuda", list(range(tp_size)))
model = MyTransformer()
model = parallelize_module(model, device_mesh, {
    "attention.qkv_proj": ColwiseParallel(),
})
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_group=(shard_pg, replicate_pg)
)

# Create distributed config for TP + HSDP
config = create_auto_config(
    tp_process_group=tp_pg,
    shard_process_group=shard_pg,
    replicate_process_group=replicate_pg,
    tp_shard_dim=1
)

# Optimizer
optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
```

---

## Expert Parallel (EP)

Expert Parallel distributes **different experts to different GPUs** in Mixture-of-Experts (MoE) models. Unlike TP/FSDP where a single layer is sharded, in EP each expert is a complete, independent unit.

```python
import torch
import torch.distributed as dist
from torch.optim.muon import Muon

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# MoE model with expert parallelism
# Each GPU holds COMPLETE different experts (not sharded)
class MoEModel(torch.nn.Module):
    def __init__(self, num_experts=16, expert_dim=1024):
        super().__init__()
        # Each GPU holds a subset of COMPLETE experts
        experts_per_gpu = num_experts // world_size
        start_expert = rank * experts_per_gpu
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(expert_dim, expert_dim)
            for _ in range(experts_per_gpu)
        ])
        self.gate = torch.nn.Linear(expert_dim, num_experts)

    def forward(self, x):
        # Gate determines which experts to use
        gate_logits = self.gate(x)
        # ... MoE routing logic ...
        return output

model = MoEModel().cuda()

# Expert Parallel: Each expert is independent, NO distributed_config needed
# Each GPU optimizes its own experts independently
optimizer = Muon(model.parameters(), lr=1e-3)  # No distributed_config!

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()  # Each GPU orthogonalizes its own experts independently
```

**Key insight**: In EP, each expert is a **complete, independent** logical unit (similar to PP stages). Experts should NOT be gathered together - each GPU orthogonalizes its own experts independently. This is fundamentally different from TP/FSDP where a single logical layer is sharded and must be gathered.

### EP + TP Hybrid (TP-Sharded Experts)

If individual experts are **TP-sharded** (expert weights distributed across multiple devices), then you need `distributed_config`:

```python
# Topology: 4 experts, each expert TP-sharded across 2 GPUs = 8 GPUs total
# GPU 0-1: Expert 0 (TP-sharded)
# GPU 2-3: Expert 1 (TP-sharded)
# GPU 4-5: Expert 2 (TP-sharded)
# GPU 6-7: Expert 3 (TP-sharded)

tp_size = 2
expert_id = rank // tp_size
tp_rank = rank % tp_size

# Create TP process group for this expert
tp_group_ranks = [expert_id * tp_size + i for i in range(tp_size)]
tp_pg = dist.new_group(tp_group_ranks)

# Apply TP to expert
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel

device_mesh = DeviceMesh("cuda", list(range(tp_size)))
model = parallelize_module(model, device_mesh, {
    "experts": ColwiseParallel(),  # TP-shard expert weights
})

# Use distributed_config for TP (NOT EP!)
from torch.optim.muon import create_auto_config
config = create_auto_config(
    tp_process_group=tp_pg,
    tp_shard_dim=1  # TP sharding within each expert
)

optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()  # Gather across TP within each expert, but NOT across experts
```

**Key difference**:
- **EP alone**: Experts are independent complete units, no gathering needed
- **EP + TP**: Each expert is TP-sharded, so gather **within** each expert (across TP), but NOT **across** experts (across EP)

---

## Context Parallel (CP)

Context Parallel partitions long sequences across GPUs while keeping model parameters replicated.

```python
import torch
import torch.distributed as dist
from torch.optim.muon import Muon, create_auto_config

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Model with context parallelism for long sequences
model = LongContextTransformer(
    seq_length=128000,  # Very long sequence
    hidden_dim=2048
).cuda()

# With CP, sequence is split but model params are replicated
# Each GPU processes different portion of sequence
seq_per_gpu = 128000 // world_size

# Create CP process group
cp_pg = dist.group.WORLD

# Create distributed config for CP
config = create_auto_config(cp_process_group=cp_pg)

# Optimizer
optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

# Training loop with sequence splitting
for epoch in range(num_epochs):
    for batch in dataloader:
        # Split sequence across GPUs
        seq_start = rank * seq_per_gpu
        seq_end = (rank + 1) * seq_per_gpu
        batch_chunk = batch[:, seq_start:seq_end, :]

        optimizer.zero_grad()
        loss = model(batch_chunk).loss
        loss.backward()

        # All-reduce gradients across CP group (standard DDP behavior)
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, group=cp_pg)

        # Optimizer handles work coordination and broadcast
        optimizer.step()  # Only subset of updates orthogonalized per GPU, then broadcast results
```

**Key insight**: CP is different from other parallelism strategies - model parameters are replicated, not sharded. The workflow is:

1. **Gradient all-reduce**: Standard DDP behavior - each GPU has identical gradients after backward pass
2. **Momentum update**: Each GPU computes same momentum state using the all-reduced gradients
3. **Work assignment**: Each GPU orthogonalizes a subset of updates (e.g., GPU 0 gets even indices, GPU 1 gets odd)
4. **Exchange results**: After orthogonalization, broadcast (NOT all-reduce) to exchange orthogonalized updates
   - Each GPU broadcasts its orthogonalized updates to all others
   - Result: all GPUs have all orthogonalized updates, which they apply to replicated params

**Difference from DDP**: DDP all-reduces gradients and each GPU updates all parameters. CP all-reduces gradients but splits orthogonalization work, then broadcasts results to avoid redundant computation.

---

## Pipeline Parallel

Pipeline Parallel partitions model layers across GPUs. Each stage runs optimizer independently.

```python
import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage
from torch.optim.muon import Muon, create_auto_config

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Define pipeline stages (4 stages, 100 layers total)
num_stages = 4
stage_id = rank  # Assuming 1 GPU per stage

def get_stage_model(full_model, stage_id, num_stages):
    """Extract layers for this pipeline stage"""
    layers_per_stage = len(full_model.layers) // num_stages
    start_idx = stage_id * layers_per_stage
    end_idx = start_idx + layers_per_stage if stage_id < num_stages - 1 else len(full_model.layers)

    stage_model = StageModel(full_model.layers[start_idx:end_idx])
    return stage_model

# Create stage-specific model
full_model = MyTransformer()
stage_model = get_stage_model(full_model, stage_id, num_stages).cuda()

# Create distributed config for PP
config = create_auto_config(
    pp_process_group=dist.group.WORLD,
    pp_stage_id=stage_id,
    pp_num_stages=num_stages
)

# Each stage creates its own optimizer
optimizer = Muon(stage_model.parameters(), lr=1e-3, distributed_config=config)

# Training loop with 1F1B schedule
def train_step_1f1b(stage_model, optimizer, microbatches):
    num_microbatches = len(microbatches)

    # Warmup: forward passes
    for mb_id in range(min(num_microbatches, num_stages)):
        if stage_id == 0:
            input_mb = microbatches[mb_id]
        else:
            input_mb = receive_from_prev_stage()

        output_mb = stage_model(input_mb)

        if stage_id < num_stages - 1:
            send_to_next_stage(output_mb)

    # 1F1B: Steady state
    for mb_id in range(num_stages, num_microbatches):
        # Backward for previous microbatch
        if stage_id == num_stages - 1:
            loss = compute_loss(output_mb)
            grad_mb = torch.autograd.grad(loss, output_mb)[0]
        else:
            grad_mb = receive_grad_from_next_stage()

        compute_backward(grad_mb)

        if stage_id > 0:
            send_grad_to_prev_stage()

        # Forward for new microbatch
        if stage_id == 0:
            input_mb = microbatches[mb_id]
        else:
            input_mb = receive_from_prev_stage()

        output_mb = stage_model(input_mb)

        if stage_id < num_stages - 1:
            send_to_next_stage(output_mb)

    # Cooldown: remaining backward passes
    for mb_id in range(num_microbatches - num_stages, num_microbatches):
        if stage_id == num_stages - 1:
            grad_mb = torch.autograd.grad(loss, output_mb)[0]
        else:
            grad_mb = receive_grad_from_next_stage()

        compute_backward(grad_mb)

        if stage_id > 0:
            send_grad_to_prev_stage()

    # Optimizer step (runs independently per stage)
    optimizer.step()
    optimizer.zero_grad()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        microbatches = split_into_microbatches(batch)
        train_step_1f1b(stage_model, optimizer, microbatches)
```

**Key insight**: Each pipeline stage optimizes independently. No cross-stage coordination for optimizer!

---

## Full 4D Parallelism (PP + TP + HSDP)

Ultimate parallelism: Pipeline + Tensor + Shard + Replicate.

```python
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.optim.muon import Muon, create_auto_config

# Initialize process group
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Topology: 4 pipeline stages × 2 TP × 2 shard × 2 replicate = 32 GPUs
pp_size = 4
tp_size = 2
shard_size = 2
replicate_size = 2

# Calculate position in 4D grid
pp_rank = rank // (tp_size * shard_size * replicate_size)
tp_rank = (rank // (shard_size * replicate_size)) % tp_size
shard_rank = (rank // replicate_size) % shard_size
replicate_rank = rank % replicate_size

# Create process groups for each dimension
# PP: Groups across pipeline stages (not used for optimizer)
pp_group_ranks = [pp_rank * (tp_size * shard_size * replicate_size) + i
                  for i in range(tp_size * shard_size * replicate_size)]
pp_pg = dist.new_group(pp_group_ranks) if len(set(pp_group_ranks)) > 1 else None

# TP: Adjacent pairs
tp_base = rank // tp_size * tp_size
tp_group_ranks = [tp_base + i for i in range(tp_size)]
tp_pg = dist.new_group(tp_group_ranks)

# Shard: Within local group
shard_base = rank // (shard_size * replicate_size) * (shard_size * replicate_size)
shard_group_ranks = [shard_base + replicate_rank + i * replicate_size for i in range(shard_size)]
shard_pg = dist.new_group(shard_group_ranks)

# Replicate: Across replicate dimension
replicate_base = rank // replicate_size * replicate_size
replicate_group_ranks = [replicate_base + i for i in range(replicate_size)]
replicate_pg = dist.new_group(replicate_group_ranks)

# Get stage-specific model
full_model = MyTransformer()
stage_model = get_stage_model(full_model, pp_rank, pp_size)

# Apply TP
device_mesh = DeviceMesh("cuda", list(range(tp_size)))
stage_model = parallelize_module(stage_model, device_mesh, {
    "attention.qkv_proj": ColwiseParallel(),
})

# Apply FSDP with HSDP
stage_model = FSDP(
    stage_model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_group=(shard_pg, replicate_pg)
)

# Create distributed config for PP + TP + HSDP
config = create_auto_config(
    pp_process_group=pp_pg,
    pp_stage_id=pp_rank,
    pp_num_stages=pp_size,
    tp_process_group=tp_pg,
    tp_shard_dim=1,
    shard_process_group=shard_pg,
    replicate_process_group=replicate_pg
)

# Optimizer
optimizer = Muon(stage_model.parameters(), lr=1e-3, distributed_config=config)

# Training loop (same as PP example with 1F1B)
for epoch in range(num_epochs):
    for batch in dataloader:
        microbatches = split_into_microbatches(batch)
        train_step_1f1b(stage_model, optimizer, microbatches)
```

---

## Custom Distributed Setup

### Writing Custom Functions for Non-PyTorch Frameworks

For custom distributed frameworks or proprietary parallelism strategies, define your own functions instead of using `create_auto_config()`:

```python
import torch
import torch.distributed as dist
from torch.optim.muon import Muon, DistributedConfig

# Example: Custom distributed framework (e.g., Horovod, custom MPI, proprietary)
class CustomDistributedFramework:
    """Example custom distributed framework"""
    def __init__(self):
        self.rank = 0  # Your framework's rank
        self.world_size = 4  # Your framework's world size
        self.process_group = None  # Your framework's process group handle

    def all_gather(self, tensor):
        """Your framework's all_gather implementation"""
        # Custom all_gather logic
        pass

    def scatter(self, tensor, src):
        """Your framework's scatter implementation"""
        # Custom scatter logic
        pass

# Initialize your framework
my_framework = CustomDistributedFramework()

# Define custom assign function
def custom_assign_fn(params, state):
    """Assign parameters to ranks - simple round-robin"""
    rank = state["rank"]
    world_size = state["world_size"]

    # Round-robin assignment: rank i gets params [i, i+N, i+2N, ...]
    assignments = {r: [] for r in range(world_size)}
    for idx, param in enumerate(params):
        assigned_rank = idx % world_size
        assignments[assigned_rank].append(idx)

    return assignments

# Define custom gather function
def custom_gather_fn(update, rank, state):
    """Gather sharded update using custom framework"""
    framework = state["framework"]
    shard_dim = state["shard_dim"]

    # Use your framework's all_gather
    gathered_shards = framework.all_gather(update)

    # Concatenate along shard dimension
    full_update = torch.cat(gathered_shards, dim=shard_dim)

    return full_update

# Define custom redistribute function
def custom_redistribute_fn(ortho_update, rank, state):
    """Redistribute orthogonalized update using custom framework"""
    framework = state["framework"]
    shard_dim = state["shard_dim"]
    world_size = state["world_size"]

    # Split update into shards
    shard_size = ortho_update.shape[shard_dim] // world_size
    shards = torch.split(ortho_update, shard_size, dim=shard_dim)

    # Use your framework's scatter to get this rank's shard
    my_shard = framework.scatter(shards, src=rank)

    return my_shard

# Create DistributedConfig with custom functions
config = DistributedConfig(
    assign_fn=custom_assign_fn,
    gather_fn=custom_gather_fn,
    redistribute_fn=custom_redistribute_fn,
    state={
        "rank": my_framework.rank,
        "world_size": my_framework.world_size,
        "framework": my_framework,
        "shard_dim": 0,  # Your sharding dimension
        # Add any other metadata your functions need
    }
)

# Initialize model and optimizer
model = MyTransformer().cuda()
optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

# Training loop (framework-agnostic)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()

        # Your framework's gradient synchronization
        for param in model.parameters():
            if param.grad is not None:
                param.grad = my_framework.all_reduce(param.grad)

        # Muon optimizer uses your custom functions
        optimizer.step()
```

### Custom Assignment Logic Based on Parameter Size

```python
def size_aware_assign_fn(params, state):
    """Assign parameters based on size for load balancing"""
    world_size = state["world_size"]

    # Sort parameters by size (largest first)
    param_sizes = [(idx, p.numel()) for idx, p in enumerate(params)]
    param_sizes.sort(key=lambda x: x[1], reverse=True)

    # Greedy assignment: assign each param to least-loaded rank
    rank_loads = [0] * world_size
    assignments = {r: [] for r in range(world_size)}

    for param_idx, size in param_sizes:
        # Assign to rank with smallest current load
        min_rank = min(range(world_size), key=lambda r: rank_loads[r])
        assignments[min_rank].append(param_idx)
        rank_loads[min_rank] += size

    # Log assignment for debugging
    rank = state["rank"]
    if rank == 0:
        for r in range(world_size):
            num_params = len(assignments[r])
            total_size = rank_loads[r]
            print(f"Rank {r}: {num_params} params, {total_size:,} elements")

    return assignments

# Use with create_auto_config by overriding assign_fn
config = create_auto_config(fsdp_process_group=dist.group.WORLD)
config.assign_fn = size_aware_assign_fn
config.state["rank"] = dist.get_rank()
config.state["world_size"] = dist.get_world_size()
```

---

## Debug Mode

### Using Sequential Execution for Debugging

When debugging distributed training issues, use `async_gpu_parallelism=False` for sequential execution:

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.muon import Muon, create_auto_config, DistributedConfig

# Initialize
dist.init_process_group(backend="nccl")
rank = dist.get_rank()

model = MyTransformer().cuda()
model = FSDP(model)

# Create config with debug mode enabled
config = create_auto_config(fsdp_process_group=dist.group.WORLD)

# Override for debugging (simplest execution)
config.async_gpu_parallelism = False  # Sequential execution
config.prefetch_count = 0  # Disable prefetching

optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

# Training loop - GPUs will process parameters sequentially
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()

        # In debug mode: GPU 0 processes all params, then GPU 1, etc.
        # This makes it easy to see which parameter causes issues
        if rank == 0:
            print(f"Rank {rank} starting optimizer.step()")

        optimizer.step()

        if rank == 0:
            print(f"Rank {rank} finished optimizer.step()")
```

**Benefits of debug mode:**
- Clear execution order - no race conditions or timing-dependent bugs
- Easy to add print statements without interleaving
- Simple to isolate which parameter causes issues
- Can match execution against single-GPU reference step-by-step

**Tradeoff:** Very slow performance (all GPUs idle except one), but that's acceptable for debugging.

**When to use:**
- Tracking down NaN/Inf in gradients or parameters
- Comparing distributed vs single-GPU results step-by-step
- Debugging process group configuration issues
- Investigating gather/redistribute correctness

---

## Migration Guide

### From Non-Distributed to FSDP

**Before**:
```python
optimizer = Muon(model.parameters(), lr=1e-3)
```

**After**:
```python
model = FSDP(model)
config = create_auto_config(fsdp_process_group=dist.group.WORLD)
optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)
```

### From FSDP to TP + FSDP

**Before**:
```python
config = create_auto_config(fsdp_process_group=fsdp_pg)
```

**After**:
```python
# Add TP process group
config = create_auto_config(
    tp_process_group=tp_pg,
    fsdp_process_group=fsdp_pg,
    tp_shard_dim=1
)
```

### From Custom Gather to Auto Config

**Before**:
```python
def my_gather(param, rank, state):
    # Custom gathering logic
    ...

config = DistributedConfig(
    assign_fn=my_assign,
    gather_fn=my_gather,
    redistribute_fn=my_redistribute,
    state={}
)
```

**After**:
```python
# Use auto config (covers common PyTorch strategies)
config = create_auto_config(
    fsdp_process_group=fsdp_pg,
    tp_process_group=tp_pg,
    tp_shard_dim=1
)

# Keep custom functions for non-PyTorch frameworks or custom requirements
```

---

## Debugging Tips

### 1. Verify Process Groups

```python
import torch.distributed as dist

def debug_process_groups(tp_pg, fsdp_pg):
    rank = dist.get_rank()

    tp_ranks = dist.get_process_group_ranks(tp_pg)
    fsdp_ranks = dist.get_process_group_ranks(fsdp_pg)

    print(f"Rank {rank}: TP group = {tp_ranks}, FSDP group = {fsdp_ranks}")

    # Verify orthogonality: rank appears exactly once in each group
    assert tp_ranks.count(rank) == 1
    assert fsdp_ranks.count(rank) == 1
```

### 2. Check Gather/Redistribute Correctness

```python
def test_gather_redistribute(update, config):
    """Verify gather/redistribute is identity operation"""
    original_shard = update.data.clone()

    # Simulate gather + redistribute
    full_update = config.gather_fn(update, dist.get_rank(), config.state)
    shard_update = config.redistribute_fn(full_update, dist.get_rank(), config.state)

    # Verify shard unchanged
    assert torch.allclose(shard_update, original_shard), "Gather/redistribute changed shard!"
```

### 3. Compare Against Single-GPU

```python
def verify_correctness_single_gpu():
    """Run both distributed and single-GPU, compare results"""
    torch.manual_seed(42)

    # Single-GPU version
    model_single = MyTransformer().cuda()
    optimizer_single = Muon(model_single.parameters(), lr=1e-3)

    # Distributed version
    model_dist = MyTransformer().cuda()
    model_dist = FSDP(model_dist)
    config = create_auto_config(fsdp_process_group=dist.group.WORLD)
    optimizer_dist = Muon(model_dist.parameters(), lr=1e-3, distributed_config=config)

    # Run one step
    loss_single = model_single(batch).loss
    loss_single.backward()
    optimizer_single.step()

    loss_dist = model_dist(batch).loss
    loss_dist.backward()
    optimizer_dist.step()

    # Compare parameters after step
    # Note: Need to gather sharded params to compare
    for p_single, p_dist in zip(model_single.parameters(), model_dist.parameters()):
        # Use FSDP's all_gather for parameter comparison
        full_p_dist = torch.zeros_like(p_single)
        dist.all_gather_into_tensor(full_p_dist, p_dist, group=dist.group.WORLD)
        if dist.get_rank() == 0:
            assert torch.allclose(p_single, full_p_dist, atol=1e-5), "Mismatch!"
```

### 4. Profile Communication Overhead

```python
import time

def profile_optimizer_step(optimizer, num_steps=10):
    """Profile gather/orthogonalize/redistribute time"""
    times = []

    for _ in range(num_steps):
        torch.cuda.synchronize()
        start = time.time()

        optimizer.step()

        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    print(f"Avg step time: {sum(times)/len(times):.3f}s")
    print(f"Min: {min(times):.3f}s, Max: {max(times):.3f}s")
```

### 5. Debug Hanging

```python
# Add timeout to detect hangs
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    optimizer.step()
    signal.alarm(0)  # Cancel timeout
except TimeoutError:
    print(f"Rank {dist.get_rank()} timed out during optimizer.step()!")
    # Check which collective is hanging
```

### 6. Enable Distributed Debug Logging

```python
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["NCCL_DEBUG"] = "INFO"

# Now distributed collectives print detailed logs
```

---

**Document Status**: Living document with practical examples.
**Last Updated**: 2025-10-15
