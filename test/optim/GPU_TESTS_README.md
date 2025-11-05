# GPU Tests for Muon Distributed Optimizer

## Overview

The GPU tests in `/data/users/vchiley/pytorch/test/optim/test_muon_distributed_real.py` provide comprehensive testing of Muon's distributed features with actual GPU hardware, FSDP, DDP, and NCCL communication.

## GPU Test Functions

### 1. `_test_muon_with_fsdp_gpu`
**Tests**: FSDP with NO_SHARD strategy on GPU

**What it does**:
- Creates model on GPU device
- Wraps with FSDP using `ShardingStrategy.NO_SHARD`
- Uses Muon optimizer with FSDP process group
- Runs training steps with actual NCCL communication
- Verifies 2D parameters are available for Muon

**Why NO_SHARD**: Muon requires 2D parameters. NO_SHARD keeps full parameters on each rank, ensuring compatibility.

### 2. `_test_muon_with_fsdp_sharded_gpu`
**Tests**: FSDP with FULL_SHARD strategy on GPU

**What it does**:
- Creates larger model for meaningful sharding
- Wraps with FSDP using `ShardingStrategy.FULL_SHARD`
- Tests with async mode and prefetching enabled
- Handles case where FSDP flattens parameters

**Important**: FULL_SHARD may flatten parameters, making them incompatible with Muon. This test verifies graceful handling of this case.

### 3. `_test_muon_with_ddp_gpu`
**Tests**: DDP on GPU with NCCL backend

**What it does**:
- Creates model on GPU
- Wraps with DDP using device_ids
- Tests Muon with DDP process group
- Verifies actual gradient synchronization
- Ensures training progresses correctly

### 4. `_test_muon_async_mode_gpu`
**Tests**: Async zero-redundancy mode on GPU

**What it does**:
- Tests `async_gpu_parallelism=True` with real GPUs
- Enables prefetching (`prefetch_count=1`)
- Verifies each rank processes only assigned parameters
- Tests actual parallel GPU computation

**Key feature**: Each rank processes different parameters in parallel - true zero-redundancy optimizer state.

### 5. `_test_muon_mixed_precision_gpu`
**Tests**: Mixed precision training (AMP) with Muon

**What it does**:
- Uses `torch.cuda.amp.autocast()` for forward pass
- Uses `GradScaler` for loss scaling
- Verifies Muon works with mixed precision training
- Tests DDP + AMP + Muon integration

**Important**: Tests that Muon's orthogonalization works correctly with scaled gradients.

## GPU Test Cases

### TestMuonRealDistributedGPU Class

#### `test_muon_with_fsdp_no_shard_gpu`
- Runs 2 GPUs
- Tests FSDP NO_SHARD with Muon
- Verifies parameters and training

#### `test_muon_with_fsdp_full_shard_gpu`
- Runs 2 GPUs
- Tests FSDP FULL_SHARD with Muon
- Handles parameter flattening gracefully

#### `test_muon_with_ddp_gpu`
- Runs 2 GPUs
- Tests DDP with Muon
- Verifies gradient synchronization
- Checks loss progression

#### `test_muon_async_mode_gpu`
- Runs 2 GPUs
- Tests async zero-redundancy mode
- Verifies no parameter overlap between ranks
- Confirms parallel processing

#### `test_muon_mixed_precision_gpu`
- Runs 2 GPUs
- Tests automatic mixed precision
- Verifies GradScaler integration
- Ensures numerical stability

#### `test_muon_4_gpus`
- Runs 4 GPUs (if available)
- Tests with larger world size
- Verifies balanced parameter distribution
- Tests scaling to more GPUs

#### `test_muon_8_gpus`
- Runs 8 GPUs (if available)
- Tests async mode with 8-way parallelism
- Verifies no parameter overlap across all ranks
- Confirms balanced distribution

#### `test_muon_fsdp_8_gpus`
- Runs 8 GPUs (if available)
- Tests FSDP NO_SHARD with 8 ranks
- Verifies all ranks have parameters
- Tests FSDP at scale

#### `test_muon_ddp_8_gpus`
- Runs 8 GPUs (if available)
- Tests DDP with 8 ranks
- Verifies gradient synchronization at scale
- Ensures training works on all GPUs

#### `test_baseline_vs_ddp_vs_fsdp_equivalence` ⭐ **CRITICAL**
- **THE MOST IMPORTANT TEST**
- Compares baseline (no dist) vs DDP vs FSDP
- Verifies numerical equivalence
- Uses same seeds for reproducibility
- Compares losses at each step
- Compares final parameters
- Ensures distributed training produces identical results to non-distributed
- **This test validates correctness of the entire distributed implementation**

## Requirements

### Hardware
- **Minimum**: 2 NVIDIA GPUs with CUDA support
- **Recommended**: 4+ GPUs for full test coverage
- **GPU Memory**: 8GB+ per GPU recommended

### Software
- CUDA toolkit installed
- NCCL backend available (`torch.distributed.is_nccl_available()`)
- PyTorch with CUDA support
- FSDP available (PyTorch 1.11+)

### Environment
```bash
# Check requirements
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
print(f'FSDP available: {hasattr(torch.distributed, \"fsdp\")}')
"
```

## Running GPU Tests

### Run All GPU Tests
```bash
# Requires 2+ GPUs
python test/optim/test_muon_distributed_real.py TestMuonRealDistributedGPU
```

### Run Specific GPU Test
```bash
# Test FSDP NO_SHARD
python test/optim/test_muon_distributed_real.py \
    TestMuonRealDistributedGPU.test_muon_with_fsdp_no_shard_gpu

# Test DDP on GPU
python test/optim/test_muon_distributed_real.py \
    TestMuonRealDistributedGPU.test_muon_with_ddp_gpu

# Test async mode
python test/optim/test_muon_distributed_real.py \
    TestMuonRealDistributedGPU.test_muon_async_mode_gpu
```

### Run with 4 GPUs
```bash
# Only if 4+ GPUs available
python test/optim/test_muon_distributed_real.py \
    TestMuonRealDistributedGPU.test_muon_4_gpus
```

### Run with 8 GPUs
```bash
# Only if 8+ GPUs available
python test/optim/test_muon_distributed_real.py \
    TestMuonRealDistributedGPU.test_muon_8_gpus

# Test FSDP with 8 GPUs
python test/optim/test_muon_distributed_real.py \
    TestMuonRealDistributedGPU.test_muon_fsdp_8_gpus

# Test DDP with 8 GPUs
python test/optim/test_muon_distributed_real.py \
    TestMuonRealDistributedGPU.test_muon_ddp_8_gpus
```

## Test Execution Flow

```
1. Test starts
   ↓
2. Check GPU availability (skip if < 2 GPUs)
   ↓
3. Spawn process per GPU (using mp.spawn)
   ↓
4. Each process:
   - Sets torch.cuda.set_device(rank)
   - Initializes NCCL process group
   - Creates model on its GPU
   - Wraps model (FSDP/DDP)
   - Creates Muon optimizer
   - Runs training steps
   - Collects results
   ↓
5. Main process verifies results
```

## FSDP Sharding Strategies

### NO_SHARD
```python
model = FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD)
```
- **Behavior**: Full parameters on each rank (like DDP)
- **Muon compatibility**: ✅ Excellent - keeps 2D parameters
- **Memory**: Higher (full model per rank)
- **Use case**: Testing, smaller models

### FULL_SHARD
```python
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```
- **Behavior**: Parameters sharded across ranks
- **Muon compatibility**: ⚠️ May flatten parameters
- **Memory**: Lower (sharded model)
- **Use case**: Large models, production

### HYBRID_SHARD
```python
model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)
```
- **Behavior**: Combination of FULL_SHARD and NO_SHARD
- **Muon compatibility**: ⚠️ Depends on configuration
- **Memory**: Balanced
- **Use case**: Multi-node training

## Expected Behavior

### Successful Test Output
```
test_muon_with_fsdp_no_shard_gpu ... ok
test_muon_with_fsdp_full_shard_gpu ... ok
test_muon_with_ddp_gpu ... ok
test_muon_async_mode_gpu ... ok
test_muon_mixed_precision_gpu ... ok
test_muon_4_gpus ... ok (or skipped if < 4 GPUs)

----------------------------------------------------------------------
Ran 6 tests in 45.123s

OK
```

### Tests Skipped (No GPUs)
```
test_muon_with_fsdp_no_shard_gpu ... skipped 'CUDA not available'
test_muon_with_ddp_gpu ... skipped 'CUDA not available'
...
```

## Troubleshooting

### Issue: "NCCL error: unhandled system error"
**Cause**: NCCL communication failure
**Solutions**:
- Check GPU connectivity
- Verify NCCL installation: `python -c "import torch; print(torch.cuda.nccl.version())"`
- Set environment: `export NCCL_DEBUG=INFO`
- Try different NCCL backend: `export NCCL_IB_DISABLE=1`

### Issue: "RuntimeError: CUDA out of memory"
**Cause**: Insufficient GPU memory
**Solutions**:
- Reduce batch size in tests
- Use smaller models
- Clear GPU cache: `torch.cuda.empty_cache()`
- Run fewer processes

### Issue: Tests hang indefinitely
**Cause**: Process synchronization deadlock
**Solutions**:
- Check barrier usage
- Verify all ranks reach collectives
- Enable debug: `export NCCL_DEBUG=INFO`
- Check network connectivity between GPUs

### Issue: "FSDP needs a non-CPU accelerator device"
**Cause**: Trying to run FSDP on CPU
**Solution**: This is expected - FSDP requires GPU. Use GPU tests only.

## Performance Considerations

### Communication Patterns

**DDP (All-Reduce)**:
```
GPU0: ──┐      ┌──
        ├─(All)─┤
GPU1: ──┘      └──
```
- Synchronous gradient sync
- All ranks wait for all

**FSDP (Sharding)**:
```
GPU0: [Shard 0] ──(Gather)──> [Full Model]
GPU1: [Shard 1] ──(Gather)──> [Full Model]
```
- Gather parameters on demand
- More complex communication

**Muon Async**:
```
GPU0: [Param 0, 2] (independent)
GPU1: [Param 1, 3] (independent)
```
- Zero-redundancy
- Minimal communication
- Broadcast updates only

### Expected Timings (2 GPUs)

| Test | Time (sec) | Notes |
|------|-----------|-------|
| FSDP NO_SHARD | 5-10 | Similar to DDP |
| FSDP FULL_SHARD | 10-20 | More communication |
| DDP | 3-8 | Baseline |
| Async mode | 5-12 | Reduced communication |
| Mixed precision | 3-8 | Faster forward pass |
| 4 GPUs | 15-30 | More coordination |

## Integration with FSDP

### Recommended Usage Pattern

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import Muon
from torch.optim._muon import create_processgroup_config

# Setup distributed
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Create model
model = MyModel().cuda(rank)

# Wrap with FSDP (use NO_SHARD for Muon compatibility)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.NO_SHARD,
    device_id=rank,
)

# Create Muon optimizer with FSDP process group
config = create_processgroup_config(
    fsdp_pg=dist.group.WORLD,
    async_gpu_parallelism=True,
    prefetch_count=1,
)

# Only use 2D parameters
params_2d = [p for p in model.parameters() if p.dim() == 2]
optimizer = Muon(params_2d, lr=0.02, distributed_config=config)

# Training loop
for batch in dataloader:
    output = model(batch)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: GPU Tests

on: [push, pull_request]

jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu, multi-gpu]

    steps:
    - uses: actions/checkout@v2

    - name: Check GPU availability
      run: nvidia-smi

    - name: Run GPU tests
      run: |
        python test/optim/test_muon_distributed_real.py \
          TestMuonRealDistributedGPU
      timeout-minutes: 30
```

### Manual Testing Script
```bash
#!/bin/bash
# test_gpu.sh

echo "=== GPU Test Suite for Muon Optimizer ==="
echo

# Check requirements
echo "Checking GPU availability..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.device_count() >= 2, 'Need 2+ GPUs'
print(f'✓ Found {torch.cuda.device_count()} GPUs')
"

echo
echo "Running GPU tests..."
python test/optim/test_muon_distributed_real.py TestMuonRealDistributedGPU -v

echo
echo "✓ GPU tests complete!"
```

## Summary

The GPU tests provide:
- ✅ Real FSDP integration (NO_SHARD and FULL_SHARD)
- ✅ Real DDP integration with NCCL
- ✅ Async zero-redundancy mode on GPU
- ✅ Mixed precision training support
- ✅ Multi-GPU scaling (2, 4+ GPUs)
- ✅ Actual NCCL communication patterns
- ✅ Production-like scenarios

These tests ensure Muon works correctly in real-world distributed GPU training environments.
