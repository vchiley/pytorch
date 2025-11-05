# Distributed Testing for Muon Optimizer

This directory contains three types of tests for the Muon optimizer's distributed capabilities:

## Test Files

### 1. `test_muon_distributed.py` - Unit Tests (Mocked)
**Purpose**: Fast unit tests with mocked distributed components

**Characteristics**:
- Uses `unittest.mock` to mock `torch.distributed` functions
- Tests individual functions in isolation
- Very fast execution (no process spawning)
- Easy to debug (single process)
- Run in CI on every commit

**What it tests**:
- Assignment validation logic
- Configuration helper functions
- Gather/redistribute function signatures
- Parameter assignment algorithms
- Edge cases and error conditions

**How to run**:
```bash
python test/optim/test_muon_distributed.py
# or
python -m pytest test/optim/test_muon_distributed.py -v
```

### 2. `test_muon_e2e.py` - End-to-End Tests (Simulated)
**Purpose**: End-to-end testing with simulated distributed behavior

**Characteristics**:
- Uses mock functions but tests the full optimizer workflow
- Single process execution
- Tests backward compatibility
- Tests complete training loops

**What it tests**:
- Non-distributed Muon (baseline)
- Distributed Muon with simulated single rank
- Async mode simulation
- Assignment validation in context
- 2D parameter requirements

**How to run**:
```bash
python test/optim/test_muon_e2e.py
```

### 3. `test_muon_distributed_real.py` - Real Distributed Tests (Multi-Process)
**Purpose**: Integration tests with actual distributed communication

**Characteristics**:
- Spawns multiple processes using `torch.multiprocessing`
- Uses real `torch.distributed` with Gloo/NCCL backends
- Tests actual FSDP and DDP wrappers
- Requires working distributed setup
- Automatically skipped if requirements not met

**What it tests**:
- Real FSDP integration
- Real DDP integration
- Actual parameter assignment across ranks
- Gradient synchronization
- Async mode with multiple processes
- Prefetching with real communication
- Multiple world sizes (1, 2, 4 ranks)

**How to run**:
```bash
# Run all real distributed tests (CPU-based, uses Gloo)
python test/optim/test_muon_distributed_real.py

# Run with pytest
python -m pytest test/optim/test_muon_distributed_real.py -v

# Run specific test
python -m pytest test/optim/test_muon_distributed_real.py::TestMuonRealDistributed::test_muon_with_fsdp -v
```

**GPU tests** (requires 2+ GPUs):
```bash
# GPU tests are automatically skipped if:
# - CUDA not available
# - Less than 2 GPUs
# - NCCL not available

python -m pytest test/optim/test_muon_distributed_real.py::TestMuonRealDistributedGPU -v
```

## Test Coverage Comparison

| Test Type | Mocked Unit | E2E Simulated | Real Distributed |
|-----------|-------------|---------------|------------------|
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Medium | ⚡ Slow |
| **CI Friendly** | ✅ Yes | ✅ Yes | ⚠️ Limited |
| **Debugging** | ✅ Easy | ✅ Easy | ⚠️ Harder |
| **Real Communication** | ❌ No | ❌ No | ✅ Yes |
| **FSDP/DDP** | ❌ Mocked | ⚠️ Simulated | ✅ Real |
| **Multi-Process** | ❌ No | ❌ No | ✅ Yes |
| **GPU Testing** | ❌ No | ❌ No | ✅ Optional |
| **Edge Cases** | ✅ Excellent | ✅ Good | ⚠️ Limited |

## Recommended Testing Strategy

### During Development
```bash
# Fast feedback loop - run unit tests frequently
python test/optim/test_muon_distributed.py
python test/optim/test_muon_e2e.py
```

### Before Committing
```bash
# Run all tests including real distributed
python test/optim/test_muon_distributed.py
python test/optim/test_muon_e2e.py
python test/optim/test_muon_distributed_real.py
```

### In CI Pipeline
```bash
# Fast tests on every commit
python test/optim/test_muon_distributed.py
python test/optim/test_muon_e2e.py

# Real distributed tests (if CI has multi-core machines)
python test/optim/test_muon_distributed_real.py
```

### Manual Validation (with GPUs)
```bash
# Run GPU tests on multi-GPU machine
python test/optim/test_muon_distributed_real.py
```

## Troubleshooting

### Tests hang or timeout
**Problem**: Real distributed tests hang during process spawning

**Solutions**:
- Check that port 12355 is not in use
- Ensure `torch.distributed` is properly installed
- Verify Gloo backend is available: `torch.distributed.is_gloo_available()`

### Import errors
**Problem**: Cannot import `create_processgroup_config`

**Solution**: Ensure you're running from the PyTorch root directory:
```bash
cd /data/users/vchiley/pytorch
python test/optim/test_muon_distributed_real.py
```

### GPU tests always skipped
**Problem**: GPU tests are being skipped

**Check**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"NCCL available: {torch.distributed.is_nccl_available()}")
```

### Process cleanup issues
**Problem**: Processes not cleaning up properly

**Solution**: The tests use `cleanup_process_group()` to ensure proper cleanup. If issues persist:
```bash
# Kill any hanging Python processes
pkill -9 python
```

## Writing New Tests

### Adding a mocked unit test
Add to `/data/users/vchiley/pytorch/test/optim/test_muon_distributed.py`:
```python
class TestNewFeature(unittest.TestCase):
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_new_feature(self, mock_rank, mock_is_init):
        mock_is_init.return_value = True
        mock_rank.return_value = 0
        # Your test here
```

### Adding a real distributed test
Add to `/data/users/vchiley/pytorch/test/optim/test_muon_distributed_real.py`:

1. Create test function that runs on each rank:
```python
def _test_new_feature(rank, world_size, init_method, results_queue):
    try:
        setup_process_group(rank, world_size, backend="gloo", init_method=init_method)
        # Your test code here
        cleanup_process_group()
        results_queue.put((rank, "success", result_data))
    except Exception as e:
        cleanup_process_group()
        results_queue.put((rank, "error", str(e)))
```

2. Add test method to `TestMuonRealDistributed`:
```python
def test_new_feature(self):
    """Test new feature with real distributed."""
    results = run_distributed_test(_test_new_feature, world_size=2)
    for rank, status, result in results:
        self.assertEqual(status, "success", f"Rank {rank} failed: {result}")
```

## Why We Have Both Mocked and Real Tests

**Mocked tests** are essential for:
- Fast development iterations
- Testing error conditions easily
- Running on any machine (no special requirements)
- CI/CD pipelines
- Debugging individual components

**Real tests** are essential for:
- Validating actual distributed behavior
- Catching integration issues
- Testing real communication patterns
- Verifying performance characteristics
- Ensuring FSDP/DDP compatibility

**The hybrid approach gives you the best of both worlds!**
