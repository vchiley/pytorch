# Muon Distributed Training: Usage Examples

This document provides concrete implementations and usage patterns for distributed Muon training.

**Related Documents**:
- `PROJECT.md` - High-level overview and problem statement
- `DESIGN.md` - Complete API specification and implementation details
- `TESTING.md` - Testing strategy and validation approach

## Configuration Functions

This document shows usage examples for both DTensor and process group configurations.

**For detailed implementation and API specification**, see `DESIGN.md`.

### DTensor Configuration: `create_dtensor_config()`

Simple configuration for models using `torch.distributed.tensor.DTensor`:
- Automatically extracts device mesh and placement from DTensor parameters
- No process groups needed
- See [API Specification in DESIGN.md](DESIGN.md#api-specification) for implementation details

### Process Group Configuration: `create_auto_config()`

Advanced configuration for models using process groups directly:
- Supports FSDP, TP, DDP, EP, CP and combinations
- Requires explicit process group specification
- Supports per-parameter TP dimensions
- See [API Specification in DESIGN.md](DESIGN.md#api-specification) for details and [Auto-Configuration Implementation in DESIGN.md](DESIGN.md#auto-configuration-implementation) for complete implementation

## Usage Examples

> **Note on Example Code**: Throughout these examples, `MyModel`, `MyModelWithDTensorParams`, `LargeModel`, and similar class names represent placeholder `torch.nn.Module` instances. Replace these with your actual model classes.

Examples are organized by priority to help you get started quickly:

### Priority Guide

**P0: Start Here** - Essential examples for getting started:
- Example 0a: DTensor with Automatic Configuration
- Example 0b: DTensor with Sharding
- Example 1: FSDP Only
- Example 2: DDP Only

**P1: Common Patterns** - Frequently used distributed strategies:
- Example 0c: DTensor with 2D Mesh (FSDP-like + TP-like)
- Example 0d: DTensor with Replicate (DDP-like)
- Example 3: Tensor Parallel with Different Dimensions per Layer
- Example 5: FSDP + Tensor Parallel (2D Parallelism)
- Example 6: HSDP (FSDP + DDP)

**P2: Advanced Configurations** - Specialized patterns and large-scale setups:
- Example 4: Tensor Parallel with Single Dimension (Shorthand)
- Example 7: Expert Parallel (MoE)
- Example 7b: Expert Parallel + FSDP
- Example 8: Complex 3D Parallelism (FSDP + TP + DDP)
- Custom Configurations section

**Recommendation**: Start with P0 examples and verify correctness before moving to P1/P2.

### DTensor-Based Examples

For models using `torch.distributed.tensor.DTensor`, use `create_dtensor_config()` for simpler configuration:

> **Important: DTensor Parameter Handling**
>
> When working with DTensor parameters in PyTorch, note that `nn.Parameter` wraps the actual tensor.
> The DTensor is stored in `param.data`, not in `param` directly.
>
> - **Correct**: `isinstance(param.data, DTensor)` - checks the underlying tensor
> - **Incorrect**: `isinstance(param, DTensor)` - checks the Parameter wrapper
>
> This affects how you verify DTensor parameters and how the configuration functions extract metadata.
> See [DTensor Configuration Implementation in DESIGN.md](DESIGN.md#dtensor-configuration-implementation) for `gather_fn` implementation details.

#### Example 0a: DTensor with Automatic Configuration

```python
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate
from torch.optim import Muon
from torch.optim.muon import create_dtensor_config

# Model with DTensor parameters
# (DTensor parameters already contain device mesh and placement info)
model = MyModelWithDTensorParams()

# Create config - no process groups needed!
config = create_dtensor_config()

# Create optimizer
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

# Training loop
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### Example 0b: DTensor with Sharding

```python
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, distribute_tensor
from torch.optim import Muon
from torch.optim.muon import create_dtensor_config

# Initialize distributed
dist.init_process_group(backend="nccl")

# Create device mesh (e.g., 1D mesh for simple sharding)
device_mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))

# Create model with regular parameters
model = torch.nn.Linear(1024, 1024)

# Convert parameters to DTensor with sharding
for name, param in model.named_parameters():
    # Shard weight along dimension 0 (row-parallel)
    if 'weight' in name:
        param_dtensor = distribute_tensor(
            param.data,
            device_mesh=device_mesh,
            placements=[Shard(0)],
        )
        param.data = param_dtensor

# Create optimizer with DTensor config
config = create_dtensor_config()
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

# Training: DTensor handles all distributed operations automatically
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()  # Orthogonalization uses full_tensor() internally
    optimizer.zero_grad()
```

#### Example 0c: DTensor with 2D Mesh (FSDP-like + TP-like)

```python
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, distribute_tensor
from torch.optim import Muon
from torch.optim.muon import create_dtensor_config

# Initialize distributed
dist.init_process_group(backend="nccl")

# Create 2D device mesh: (FSDP dimension, TP dimension)
# For 8 GPUs: 2x4 mesh (2 FSDP groups, 4 TP groups)
device_mesh = DeviceMesh(
    "cuda",
    torch.arange(8).reshape(2, 4),
    mesh_dim_names=("fsdp", "tp"),
)

# Distribute model parameters with 2D sharding
model = torch.nn.Linear(2048, 2048)
for name, param in model.named_parameters():
    if 'weight' in name:
        # Shard along both dimensions: FSDP (dim 0) and TP (dim 1)
        param_dtensor = distribute_tensor(
            param.data,
            device_mesh=device_mesh,
            placements=[Shard(0), Shard(1)],  # 2D sharding
        )
        param.data = param_dtensor

# Create optimizer - DTensor handles the complex 2D communication
config = create_dtensor_config(
    async_gpu_parallelism=True,
    prefetch_count=1,
)
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

# Training: full_tensor() automatically gathers from 2D mesh
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### Example 0d: DTensor with Replicate (DDP-like)

```python
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh, Replicate, distribute_tensor
from torch.optim import Muon
from torch.optim.muon import create_dtensor_config

# Initialize distributed
dist.init_process_group(backend="nccl")

# Create device mesh
device_mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))

# Model with replicated parameters (like DDP)
model = torch.nn.Linear(512, 512)
for name, param in model.named_parameters():
    # Replicate parameters across all devices
    param_dtensor = distribute_tensor(
        param.data,
        device_mesh=device_mesh,
        placements=[Replicate()],  # Full replication
    )
    param.data = param_dtensor

# Create optimizer
config = create_dtensor_config()
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

# Training: full_tensor() is a no-op for replicated tensors,
# but zero redundancy is maintained through assignment
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()  # Only assigned rank orthogonalizes, then broadcasts
    optimizer.zero_grad()
```

### Process Group-Based Examples

For models using traditional process groups (FSDP, DDP, TP), use `create_auto_config()`:

### Example 1: FSDP Only

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Muon
from torch.optim.muon import create_auto_config

# Setup model with FSDP
model = FSDP(MyModel())

# Get FSDP process group from model
fsdp_pg = model.process_group

# Create config with just FSDP
config = create_auto_config(fsdp_process_group=fsdp_pg)

# Create optimizer
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

### Example 2: DDP Only

```python
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Muon
from torch.optim.muon import create_auto_config

# Setup model with DDP
model = DDP(MyModel())

# Use world process group for DDP
ddp_pg = dist.group.WORLD

# Create config with just DDP
config = create_auto_config(ddp_process_group=ddp_pg)

# Create optimizer
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

### Example 3: Tensor Parallel with Different Dimensions per Layer

```python
from torch.optim import Muon
from torch.optim.muon import create_auto_config

# TP setup where different layers are sharded along different dimensions
tp_pg = ...  # TP process group

# Map each parameter to its TP dimension
# Example: transformer model with column-parallel and row-parallel layers
tp_dim_per_param = {
    0: 0,  # Layer 0 weight: column-parallel (shard dim 0)
    1: 1,  # Layer 1 weight: row-parallel (shard dim 1)
    2: 0,  # Layer 2 weight: column-parallel (shard dim 0)
    3: 1,  # Layer 3 weight: row-parallel (shard dim 1)
    4: 0,  # Embedding: shard vocab dimension (dim 0)
    # ... etc
}

# Create config with per-parameter TP dimensions
config = create_auto_config(
    tp_process_group=tp_pg,
    tp_dim_per_param=tp_dim_per_param,
)

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

### Example 4: Tensor Parallel with Single Dimension (Shorthand)

```python
from torch.optim import Muon
from torch.optim.muon import create_auto_config

# If all parameters use the same TP dimension
tp_pg = ...

# Use single int for all parameters
config = create_auto_config(
    tp_process_group=tp_pg,
    tp_dim_per_param=1,  # All params sharded along dim 1
)

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

### Example 5: FSDP + Tensor Parallel (2D Parallelism)

```python
from torch.optim import Muon
from torch.optim.muon import create_auto_config

# 2D parallelism with different TP dimensions per layer
fsdp_pg = ...  # FSDP process group
tp_pg = ...    # TP process group

# Different layers have different TP sharding patterns
tp_dim_per_param = {
    0: 0,  # QKV projection: column-parallel
    1: 1,  # Output projection: row-parallel
    2: 0,  # FFN up projection: column-parallel
    3: 1,  # FFN down projection: row-parallel
    # ... pattern repeats for each transformer block
}

# Create config with both FSDP and TP
config = create_auto_config(
    fsdp_process_group=fsdp_pg,
    tp_process_group=tp_pg,
    tp_dim_per_param=tp_dim_per_param,
)

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

### Example 6: HSDP (FSDP + DDP)

```python
from torch.optim import Muon
from torch.optim.muon import create_auto_config

# HSDP: FSDP within nodes, DDP across nodes
fsdp_pg = ...  # Intra-node sharding
ddp_pg = ...   # Inter-node replication

# Create config with FSDP and DDP
config = create_auto_config(
    fsdp_process_group=fsdp_pg,
    ddp_process_group=ddp_pg,
)

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

### Example 7: Expert Parallel (MoE)

```python
from torch.optim import Muon
from torch.optim.muon import create_auto_config

# Setup for MoE model with expert parallelism
ep_pg = ...  # Expert parallel process group

# Map each parameter to its expert ID
expert_assignments = {
    0: 0,  # param 0 belongs to expert 0
    1: 0,  # param 1 belongs to expert 0
    2: 1,  # param 2 belongs to expert 1
    3: 1,  # param 3 belongs to expert 1
    # ... etc
}

# Create config with Expert Parallel
config = create_auto_config(
    ep_process_group=ep_pg,
    expert_assignments=expert_assignments,
)

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

### Example 7b: Expert Parallel + FSDP (MoE with Sharded Experts)

```python
from torch.optim import Muon
from torch.optim.muon import create_auto_config

# MoE model where experts are distributed AND each expert is FSDP-sharded
# This is useful for very large experts that don't fit on a single GPU
ep_pg = ...    # Expert parallel process group (across experts)
fsdp_pg = ...  # FSDP within each expert (shard expert params)

# Map each parameter to its expert ID
expert_assignments = {
    0: 0,  # Expert 0 params
    1: 0,
    2: 1,  # Expert 1 params
    3: 1,
    # ... etc
}

# Combine EP (no gather across experts) with FSDP (gather within each expert)
config = create_auto_config(
    ep_process_group=ep_pg,
    fsdp_process_group=fsdp_pg,      # FSDP shards each expert
    expert_assignments=expert_assignments,
)

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

**How it works**: Each expert's parameters are orthogonalized independently (EP), but the gather/redistribute operations still apply FSDP within each expert to handle the sharding.

### Example 8: Complex 3D Parallelism (FSDP + TP + DDP)

```python
from torch.optim import Muon
from torch.optim.muon import create_auto_config

# 3D parallelism setup
fsdp_pg = ...  # Sharding
tp_pg = ...    # Tensor parallel
ddp_pg = ...   # Data parallel (outer replication)

# Per-parameter TP dimensions
tp_dim_per_param = {
    0: 0, 1: 1, 2: 0, 3: 1,  # Pattern for different layer types
    # ... etc
}

# Single function call handles all three!
config = create_auto_config(
    fsdp_process_group=fsdp_pg,
    tp_process_group=tp_pg,
    ddp_process_group=ddp_pg,
    tp_dim_per_param=tp_dim_per_param,
    prefetch_count=2,  # More overlap for complex setup
)

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

## Helper: Automatically Detect TP Dimensions

**When to use this**: Many parallelism frameworks (Megatron-LM, PyTorch DTensor, custom TP implementations) attach metadata to parameters indicating how they're sharded. Instead of manually building `tp_dim_per_param` by inspecting your model architecture, you can extract this information automatically from parameter attributes.

**Why this matters**: For large models with hundreds of parameters, manually specifying TP dimensions for each parameter is error-prone. Auto-detection ensures your `tp_dim_per_param` dict stays in sync with your model's actual sharding configuration.

For models using standard parallelism libraries, you can automatically detect TP dimensions:

```python
def detect_tp_dims_from_model(model):
    """
    Automatically detect TP dimension for each parameter.

    This example assumes parameters have a 'tp_dim' attribute set by
    the parallelism framework. Adjust based on your framework.
    """
    tp_dim_per_param = {}

    for param_idx, param in enumerate(model.parameters()):
        # Check if parameter has TP metadata
        if hasattr(param, 'tp_dim'):
            tp_dim_per_param[param_idx] = param.tp_dim
        elif hasattr(param, 'partition_dim'):
            tp_dim_per_param[param_idx] = param.partition_dim
        else:
            # Default to dim 0 if no metadata found
            tp_dim_per_param[param_idx] = 0

    return tp_dim_per_param

# Usage
tp_dims = detect_tp_dims_from_model(model)
config = create_auto_config(
    tp_process_group=tp_pg,
    tp_dim_per_param=tp_dims,
)
```

## Performance Tuning

### Debugging Configuration

Start with conservative settings for debugging:

```python
config = create_auto_config(
    fsdp_process_group=fsdp_pg,
    async_gpu_parallelism=False,  # Sequential processing
    prefetch_count=0,              # No prefetching
)
```

### Production Configuration

Once correctness is verified, enable optimizations:

```python
config = create_auto_config(
    fsdp_process_group=fsdp_pg,
    tp_process_group=tp_pg,
    tp_dim_per_param=tp_dims,
    async_gpu_parallelism=True,   # Parallel processing
    prefetch_count=1,              # Overlap communication
)
```

### Aggressive Optimization

For models with many parameters and fast GPUs:

```python
config = create_auto_config(
    fsdp_process_group=fsdp_pg,
    tp_process_group=tp_pg,
    tp_dim_per_param=tp_dims,
    async_gpu_parallelism=True,
    prefetch_count=2,  # More prefetching (higher memory usage)
)
```

## ⚠️ Important: Operation Order for Custom Configurations

When implementing custom `gather_fn` and `redistribute_fn`, operation order is **critical for correctness**:

- **gather_fn**: Apply operations inner to outer (TP first, then FSDP)
- **redistribute_fn**: Apply operations outer to inner (FSDP first, then TP) - **reversed order**

**Why reversed?** You must undo operations in reverse order when unpacking a nested structure, like unwrapping a gift - remove the outer wrapping (FSDP) before the inner wrapping (TP).

See [Auto-Configuration Implementation in DESIGN.md](DESIGN.md#auto-configuration-implementation) for detailed examples and explanation.

## Custom Configurations

For unusual distributed setups not covered by `create_auto_config()`, you can define custom functions:

```python
from torch.optim.muon import DistributedConfig

def custom_assign_fn(params, state):
    # Custom assignment logic
    return {0: [0, 1], 1: [2, 3]}

def custom_gather_fn(tensor, rank, state):
    # Custom gather logic
    return tensor

def custom_redistribute_fn(update, rank, state):
    # Custom redistribute logic
    return update

config = DistributedConfig(
    assign_fn=custom_assign_fn,
    gather_fn=custom_gather_fn,
    redistribute_fn=custom_redistribute_fn,
    state={'rank': dist.get_rank(), 'custom_metadata': ...},
    async_gpu_parallelism=True,
    prefetch_count=1,
)

optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

## Choosing Between DTensor and Process Group Approaches

### Use `create_dtensor_config()` when:

1. **Your model already uses DTensor**: Parameters are `torch.distributed.tensor.DTensor` instances
2. **You want simpler configuration**: DTensor metadata is extracted automatically
3. **You're using PyTorch's native DTensor APIs**: DeviceMesh, Shard, Replicate placements
4. **Multi-dimensional sharding**: DTensor naturally handles N-D device meshes

**Example scenario**: Modern PyTorch models using DTensor for distributed training, or when converting from single-device to distributed with minimal code changes.

### Use `create_auto_config()` when:

1. **Your model uses process groups directly**: FSDP, DDP, custom distributed wrappers
2. **You need explicit control**: Fine-grained control over gather/redistribute operations
3. **Legacy distributed code**: Existing code base with process group infrastructure
4. **Expert Parallel or custom patterns**: Non-standard sharding that requires custom logic

**Example scenario**: Large-scale training frameworks (Megatron-LM style), MoE models with expert parallelism, or existing codebases with established process group patterns.

### Comparison Table

| Feature | `create_dtensor_config()` | `create_auto_config()` |
|---------|---------------------------|------------------------|
| **API Complexity** | Simple (no args needed) | Moderate (process groups required) |
| **Setup Overhead** | Low (automatic detection) | Higher (manual PG setup) |
| **Gather Function** | `DTensor.full_tensor()` | Manual `all_gather` + concat |
| **Redistribute Function** | `distribute_tensor()` | Manual slice + broadcast/scatter |
| **Flexibility** | Works with any DTensor placement | Supports custom distributed strategies |
| **Use Case** | Native DTensor models | Process group-based distributed training |
| **Best For** | Modern PyTorch, simpler setups | Large-scale training, custom frameworks |

## Migration Paths

### Path 1: Single-Device → Distributed with DTensor

Migrating from single-device training to distributed training with minimal code changes:

```python
# Before: Single-device training
from torch.optim import Muon

model = MyModel()
optimizer = Muon(model.parameters(), lr=0.01)

# Train normally
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```python
# After: Distributed training with DTensor
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DeviceMesh, Shard
from torch.optim import Muon
from torch.optim.muon import create_dtensor_config

# Initialize distributed (add this)
dist.init_process_group(backend="nccl")

# Create model and shard parameters (modify this)
model = MyModel()
device_mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))
for name, param in model.named_parameters():
    param.data = distribute_tensor(param.data, device_mesh, [Shard(0)])

# Create optimizer with distributed config (modify this)
config = create_dtensor_config()
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

# Training loop stays the same!
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Key changes**:
1. Initialize distributed process group
2. Convert model parameters to DTensor with desired placement
3. Create distributed config and pass to optimizer
4. Training loop remains unchanged

### Path 2: Single-Device → Distributed with FSDP

Migrating to FSDP for larger models:

```python
# Before: Single-device training
model = LargeModel()
optimizer = Muon(model.parameters(), lr=0.01)
```

```python
# After: FSDP distributed training
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.muon import create_auto_config

# Initialize distributed
dist.init_process_group(backend="nccl")

# Wrap model with FSDP
model = FSDP(LargeModel())

# Create distributed config from FSDP process group
config = create_auto_config(fsdp_process_group=model.process_group)
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

### Path 3: Adding Muon to Existing Distributed Training

Adding Muon optimizer to an existing distributed training setup:

```python
# Before: Existing distributed training with AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(MyModel())
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Training loop
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```python
# After: Replace with Muon
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Muon
from torch.optim.muon import create_auto_config

model = FSDP(MyModel())

# Add distributed config for Muon
config = create_auto_config(fsdp_process_group=model.process_group)
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)  # Changed

# Training loop stays the same
for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Note**: You may need to tune the learning rate when switching optimizers. Muon typically uses higher learning rates than AdamW.

### Path 4: Process Groups → DTensor

Simplifying process group configuration by migrating to DTensor:

```python
# Before: Process group approach with FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.muon import create_auto_config

model = FSDP(MyModel())
config = create_auto_config(fsdp_process_group=model.process_group)
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

```python
# After: DTensor approach (simpler)
from torch.distributed.tensor import distribute_tensor, DeviceMesh, Shard
from torch.optim.muon import create_dtensor_config

# Convert model parameters to DTensor with Shard placement
model = MyModel()
device_mesh = DeviceMesh("cuda", list(range(world_size)))
for name, param in model.named_parameters():
    param.data = distribute_tensor(param.data, device_mesh, [Shard(0)])

# Use simpler config - no process groups needed
config = create_dtensor_config()
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

**Benefits of DTensor migration**:
- Simpler configuration code (no manual process group management)
- Automatic handling of complex placement strategies
- Better composability with other DTensor-based features
- Future-proof as PyTorch moves toward DTensor as the standard distributed tensor API

### Path 5: DDP → DTensor with Replicate

Migrating from DDP to DTensor while maintaining replica semantics:

```python
# Before: DDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.muon import create_auto_config

model = DDP(MyModel())
config = create_auto_config(ddp_process_group=dist.group.WORLD)
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

```python
# After: DTensor with Replicate placement
from torch.distributed.tensor import distribute_tensor, DeviceMesh, Replicate
from torch.optim.muon import create_dtensor_config

model = MyModel()
device_mesh = DeviceMesh("cuda", list(range(dist.get_world_size())))
for name, param in model.named_parameters():
    param.data = distribute_tensor(param.data, device_mesh, [Replicate()])

config = create_dtensor_config()
optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)
```

**Note**: With DTensor Replicate placement, zero redundancy is maintained through the assignment mechanism - only one rank orthogonalizes each parameter, then broadcasts the result.

## Best Practices

1. **Start simple**: Begin with a single parallelism strategy (FSDP or DDP)
2. **Debug first**: Use `async_gpu_parallelism=False` and `prefetch_count=0` initially
3. **Validate correctness**: Compare outputs with single-device training
4. **Track TP dimensions**: Use per-parameter TP dimensions for complex models
5. **Automate detection**: Write helpers to detect TP dimensions from model metadata
6. **Enable gradually**: Turn on optimizations after verifying correctness
7. **Monitor memory**: Higher `prefetch_count` uses more memory
8. **Profile**: Use PyTorch profiler to identify bottlenecks

## Common Pitfalls

1. **Wrong process group**: Ensure process groups match your model's parallelism setup
2. **Wrong TP dimension**: Each parameter must have correct TP dimension specified
3. **Missing expert_assignments**: Required for Expert Parallel to work correctly
4. **Mixing TP patterns**: Ensure tp_dim_per_param matches actual model sharding
5. **Deadlocks**: All ranks must participate in collectives (handled automatically by `create_auto_config`)
6. **Memory errors**: Reduce `prefetch_count` if running out of memory during training
7. **Shape mismatches**: Verify gather/redistribute produces correct tensor shapes per parameter
