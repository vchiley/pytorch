# CLAUDE_TESTING.md - Muon Distributed Training Testing Strategy

Comprehensive testing strategy for Muon distributed optimizer implementation.

## Table of Contents
1. [Testing Philosophy](#testing-philosophy)
2. [Test Matrix](#test-matrix)
3. [Correctness Tests](#correctness-tests)
4. [Unit Tests](#unit-tests)
5. [Integration Tests](#integration-tests)
6. [Performance Tests](#performance-tests)
7. [Regression Tests](#regression-tests)
8. [CI/CD Strategy](#cicd-strategy)

---

## Testing Philosophy

### Core Principles

1. **Correctness First**: Distributed version must match single-GPU results exactly (within numerical precision)
2. **Exhaustive Coverage**: Test all parallelism combinations, not just common ones
3. **Performance Validation**: Ensure distributed version provides speedup, not slowdown
4. **Regression Prevention**: Lock in correct behavior to catch future breakages
5. **Realistic Scenarios**: Test with production-like model sizes and configurations

### Test Pyramid

```
           ┌─────────────────┐
           │  Performance    │  (Few, expensive, run weekly)
           │  Benchmarks     │
           └─────────────────┘
          ┌───────────────────┐
          │  Integration      │  (Moderate, run on PR)
          │  Tests            │
          └───────────────────┘
        ┌───────────────────────┐
        │  Unit Tests           │  (Many, fast, run on commit)
        │  (gather/redistribute)│
        └───────────────────────┘
```

---

## Test Matrix

### Parallelism Strategies to Test

| Strategy | Process Groups | Shard Dim | Priority |
|----------|---------------|-----------|----------|
| None (Single GPU) | - | - | P0 |
| DDP | WORLD | - | P0 |
| FSDP | WORLD | - | P0 |
| TP (dim 0) | WORLD | 0 | P0 |
| TP (dim 1) | WORLD | 1 | P0 |
| HSDP | shard + replicate | - | P1 |
| TP + FSDP | TP + FSDP | 0/1 | P1 |
| TP + HSDP | TP + shard + replicate | 0/1 | P1 |
| EP (Expert Parallel) | - | - | P1 |
| EP + TP (TP-sharded experts) | TP | 0/1 | P2 |
| CP (Context Parallel) | CP | - | P1 |
| PP (2 stages) | PP | - | P1 |
| PP (4 stages) | PP | - | P2 |
| PP + TP | PP + TP | 0/1 | P2 |
| PP + FSDP | PP + FSDP | - | P2 |
| EP + TP | EP + TP | 0 | P2 |
| CP + TP | CP + TP | 0/1 | P2 |
| PP + TP + HSDP | PP + TP + shard + replicate | 0/1 | P2 |

**Priority levels**:
- **P0**: Must pass for every commit (blocking CI)
- **P1**: Must pass for every PR (pre-merge CI)
- **P2**: Must pass weekly (scheduled CI)

### GPU Configurations to Test

| GPUs | Topology | Use Cases |
|------|----------|-----------|
| 1 | Single GPU | Baseline, backward compatibility |
| 2 | Single node | TP, minimal distributed |
| 4 | Single node | FSDP, TP, small HSDP |
| 8 | Single node | FSDP, TP, HSDP, small PP |
| 16 | 2 nodes × 8 | HSDP, TP+FSDP, PP |
| 32 | 4 nodes × 8 | Full 4D parallelism |

### Model Sizes to Test

| Size | Parameters | Layers | Use Case |
|------|-----------|--------|----------|
| Tiny | 10M | 4 | Fast unit tests |
| Small | 100M | 12 | Integration tests |
| Medium | 1B | 24 | Performance tests |
| Large | 7B | 32 | Weekly benchmarks |

---

## Correctness Tests

### Test 1: Distributed vs Single-GPU Match

**Goal**: Verify distributed optimization produces identical results to single-GPU.

```python
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.muon import Muon, create_auto_config


class TestMuonCorrectness(TestCase):
    def test_fsdp_matches_single_gpu(self):
        """FSDP distributed should match single-GPU results exactly"""
        # Seed for reproducibility
        torch.manual_seed(42)

        # Single-GPU baseline
        model_single = TinyTransformer(dim=256, layers=4).cuda()
        optimizer_single = Muon(model_single.parameters(), lr=1e-3, momentum=0.95)

        # FSDP distributed
        dist.init_process_group(backend="nccl")
        model_dist = TinyTransformer(dim=256, layers=4).cuda()
        model_dist = FSDP(model_dist)
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)
        optimizer_dist = Muon(
            model_dist.parameters(),
            lr=1e-3,
            momentum=0.95,
            distributed_config=config
        )

        # Run 10 steps with same data
        for step in range(10):
            # Create batch
            batch = torch.randn(8, 128, 256).cuda()

            # Single-GPU step
            optimizer_single.zero_grad()
            loss_single = model_single(batch).sum()
            loss_single.backward()
            optimizer_single.step()

            # Distributed step
            optimizer_dist.zero_grad()
            loss_dist = model_dist(batch).sum()
            loss_dist.backward()
            optimizer_dist.step()

            # Compare losses
            self.assertEqual(loss_single, loss_dist, atol=1e-5, rtol=1e-5)

        # Compare final parameters
        rank = dist.get_rank()
        for p_single, p_dist in zip(
            model_single.parameters(),
            model_dist.parameters()
        ):
            # Gather distributed parameter
            gathered = torch.zeros_like(p_single)
            dist.all_gather_into_tensor(gathered, p_dist, group=dist.group.WORLD)

            if rank == 0:
                self.assertEqual(p_single, gathered, atol=1e-4, rtol=1e-4)

        dist.destroy_process_group()


    def test_tp_matches_single_gpu(self):
        """TP distributed should match single-GPU results"""
        # Similar structure, but with TP instead of FSDP
        ...


    def test_hsdp_matches_single_gpu(self):
        """HSDP distributed should match single-GPU results"""
        ...


if __name__ == "__main__":
    run_tests()
```

### Test 2: Gradient Accumulation Correctness

**Goal**: Verify orthogonalization happens after gradient accumulation, not per microbatch.

```python
class TestGradientAccumulation(TestCase):
    def test_grad_accumulation_matches_single_batch(self):
        """Accumulated microbatches should match single large batch"""
        dist.init_process_group(backend="nccl")

        model = TinyTransformer(dim=256, layers=4).cuda()
        model = FSDP(model)
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)
        optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

        # Single large batch
        torch.manual_seed(42)
        large_batch = torch.randn(32, 128, 256).cuda()
        optimizer.zero_grad()
        loss_large = model(large_batch).sum()
        loss_large.backward()
        optimizer.step()

        params_large = [p.clone() for p in model.parameters()]

        # Reset model
        model = TinyTransformer(dim=256, layers=4).cuda()
        model = FSDP(model)
        optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

        # Accumulated microbatches (4 × 8)
        torch.manual_seed(42)
        large_batch = torch.randn(32, 128, 256).cuda()
        microbatches = large_batch.chunk(4)

        optimizer.zero_grad()
        for mb in microbatches:
            loss_mb = model(mb).sum()
            loss_mb.backward()  # Gradients accumulate

        optimizer.step()  # Orthogonalization happens once here

        params_accum = [p.clone() for p in model.parameters()]

        # Compare
        for p_large, p_accum in zip(params_large, params_accum):
            self.assertEqual(p_large, p_accum, atol=1e-5, rtol=1e-5)

        dist.destroy_process_group()
```

### Test 3: Debug Mode (Sequential Execution)

**Goal**: Verify `async_gpu_parallelism=False` enforces sequential execution.

```python
class TestDebugMode(TestCase):
    def test_sequential_execution(self):
        """Verify debug mode executes sequentially across GPUs"""
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        model = TinyTransformer(dim=256, layers=4).cuda()
        model = FSDP(model)
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)

        # Enable debug mode
        config.async_gpu_parallelism = False

        optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

        # Track execution order
        execution_log = []

        # Hook to log when each rank starts/finishes
        def log_execution(stage):
            execution_log.append((rank, stage, time.time()))

        batch = torch.randn(8, 128, 256).cuda()
        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()

        log_execution("start")
        optimizer.step()
        log_execution("end")

        # Gather logs from all ranks
        all_logs = [None] * world_size
        dist.all_gather_object(all_logs, execution_log)

        # Verify sequential execution: rank i finishes before rank i+1 starts
        if rank == 0:
            for i in range(world_size - 1):
                rank_i_end = [log for log in all_logs[i] if log[1] == "end"][0][2]
                rank_i1_start = [log for log in all_logs[i+1] if log[1] == "start"][0][2]
                self.assertLess(rank_i_end, rank_i1_start,
                                f"Rank {i} should finish before rank {i+1} starts")

        dist.destroy_process_group()


    def test_parallel_vs_sequential_same_result(self):
        """Verify parallel and sequential modes produce same results"""
        dist.init_process_group(backend="nccl")

        # Run with parallel mode
        torch.manual_seed(42)
        model_parallel = TinyTransformer(dim=256, layers=4).cuda()
        model_parallel = FSDP(model_parallel)
        config_parallel = create_auto_config(fsdp_process_group=dist.group.WORLD)
        config_parallel.async_gpu_parallelism = True
        optimizer_parallel = Muon(model_parallel.parameters(), lr=1e-3, distributed_config=config_parallel)

        batch = torch.randn(8, 128, 256).cuda()
        optimizer_parallel.zero_grad()
        loss_parallel = model_parallel(batch).sum()
        loss_parallel.backward()
        optimizer_parallel.step()

        params_parallel = [p.clone() for p in model_parallel.parameters()]

        # Run with sequential mode (same seed)
        torch.manual_seed(42)
        model_sequential = TinyTransformer(dim=256, layers=4).cuda()
        model_sequential = FSDP(model_sequential)
        config_sequential = create_auto_config(fsdp_process_group=dist.group.WORLD)
        config_sequential.async_gpu_parallelism = False
        optimizer_sequential = Muon(model_sequential.parameters(), lr=1e-3, distributed_config=config_sequential)

        optimizer_sequential.zero_grad()
        loss_sequential = model_sequential(batch).sum()
        loss_sequential.backward()
        optimizer_sequential.step()

        params_sequential = [p.clone() for p in model_sequential.parameters()]

        # Compare results
        for p_par, p_seq in zip(params_parallel, params_sequential):
            self.assertEqual(p_par, p_seq, atol=1e-6, rtol=1e-6)

        dist.destroy_process_group()
```

### Test 4: Process Group Validation

**Goal**: Verify `create_auto_config()` rejects invalid process group configurations.

```python
class TestProcessGroupValidation(TestCase):
    def test_reject_overlapping_fsdp_tp(self):
        """Should reject FSDP and TP groups that aren't orthogonal"""
        dist.init_process_group(backend="nccl")

        # Create overlapping groups (invalid)
        fsdp_pg = dist.new_group([0, 1, 2, 3])
        tp_pg = dist.new_group([0, 1, 2, 3])  # Same as FSDP, not orthogonal!

        with self.assertRaises(ValueError):
            config = create_auto_config(
                fsdp_process_group=fsdp_pg,
                tp_process_group=tp_pg
            )

        dist.destroy_process_group()


    def test_reject_invalid_pp_stage_id(self):
        """Should reject PP stage_id >= num_stages"""
        dist.init_process_group(backend="nccl")

        with self.assertRaises(ValueError):
            config = create_auto_config(
                pp_process_group=dist.group.WORLD,
                pp_stage_id=4,  # Invalid if num_stages=4
                pp_num_stages=4
            )

        dist.destroy_process_group()
```

---

## Unit Tests

### Test 1: Gather Function

```python
class TestGatherFunction(TestCase):
    def test_fsdp_gather(self):
        """Test FSDP gather function"""
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Create sharded update (momentum buffer)
        full_size = (256, 1024)
        shard_size = (256, 1024 // world_size)
        update_shard = torch.randn(shard_size).cuda()

        # Create gather function
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)

        # Gather update
        gathered_update = config.gather_fn(update_shard, rank, config.state)

        # Verify shape
        self.assertEqual(gathered_update.shape, full_size)

        # Verify all ranks get same result
        gathered_rank0 = gathered_update.clone()
        dist.broadcast(gathered_rank0, src=0)
        self.assertEqual(gathered_update, gathered_rank0)

        dist.destroy_process_group()


    def test_tp_gather_dim0(self):
        """Test TP gather along dimension 0"""
        ...


    def test_tp_gather_dim1(self):
        """Test TP gather along dimension 1"""
        ...


    def test_hsdp_gather(self):
        """Test HSDP gather (shard + replicate)"""
        ...
```

### Test 2: Redistribute Function

```python
class TestRedistributeFunction(TestCase):
    def test_fsdp_redistribute(self):
        """Test FSDP redistribute function"""
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Create full update
        full_update = torch.randn(256, 1024).cuda()

        # Create redistribute function
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)

        # Redistribute
        update_shard = config.redistribute_fn(full_update, rank, config.state)

        # Verify shape
        expected_shard_size = (256, 1024 // world_size)
        self.assertEqual(update_shard.shape, expected_shard_size)

        # Verify gather + redistribute is identity
        regathered = config.gather_fn(update_shard, rank, config.state)
        self.assertEqual(full_update, regathered, atol=1e-6)

        dist.destroy_process_group()
```

### Test 3: Assignment Function

```python
class TestAssignmentFunction(TestCase):
    def test_balanced_assignment(self):
        """Test parameters are assigned evenly across ranks"""
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()

        # Create dummy parameters
        params = [torch.randn(100, 100) for _ in range(20)]

        config = create_auto_config(fsdp_process_group=dist.group.WORLD)

        # Get assignment
        assignments = config.assign_fn(params, config.state)

        # Verify all params assigned
        all_assigned = []
        for rank_params in assignments.values():
            all_assigned.extend(rank_params)
        self.assertEqual(sorted(all_assigned), list(range(20)))

        # Verify roughly balanced
        counts = [len(assignments[r]) for r in range(world_size)]
        self.assertLessEqual(max(counts) - min(counts), 1)  # Diff at most 1

        dist.destroy_process_group()
```

---

## Integration Tests

### Test 1: End-to-End Training

```python
class TestEndToEndTraining(TestCase):
    def test_fsdp_training_converges(self):
        """Test FSDP training converges on toy problem"""
        dist.init_process_group(backend="nccl")

        # Create model and optimizer
        model = TinyTransformer(dim=128, layers=4).cuda()
        model = FSDP(model)
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)
        optimizer = Muon(model.parameters(), lr=1e-2, distributed_config=config)

        # Create toy dataset (memorize 10 sequences)
        torch.manual_seed(42)
        dataset = [torch.randn(16, 64, 128).cuda() for _ in range(10)]

        # Train for 100 steps
        losses = []
        for epoch in range(10):
            for batch in dataset:
                optimizer.zero_grad()
                output = model(batch)
                loss = (output - batch).pow(2).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Verify loss decreased significantly
        initial_loss = sum(losses[:10]) / 10
        final_loss = sum(losses[-10:]) / 10
        self.assertLess(final_loss, 0.1 * initial_loss)

        dist.destroy_process_group()
```

### Test 2: Multi-Node Training

```python
class TestMultiNode(TestCase):
    def test_hsdp_across_nodes(self):
        """Test HSDP works correctly across multiple nodes"""
        # This test requires multi-node setup
        # Run with: torchrun --nproc_per_node=8 --nnodes=2 test_muon.py
        ...
```

---

## Performance Tests

### Test 1: Speedup vs Sequential

```python
import time


class TestPerformance(TestCase):
    def test_parallel_speedup(self):
        """Verify parallel version is faster than sequential"""
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()

        model = MediumTransformer(dim=1024, layers=24).cuda()
        model = FSDP(model)
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)
        optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

        # Warmup
        batch = torch.randn(4, 128, 1024).cuda()
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()

        # Measure time for 10 steps
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(10):
            optimizer.zero_grad()
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        end = time.time()

        time_per_step = (end - start) / 10

        # Expected: ~linear speedup (allowing for communication overhead)
        # On 8 GPUs, expect at least 4x speedup (50% efficiency)
        if world_size == 8:
            self.assertLess(time_per_step, 0.5)  # Baseline ~2s, expect <1s

        dist.destroy_process_group()


    def test_prefetch_speedup(self):
        """Verify prefetching provides speedup"""
        # Compare with/without prefetching (requires implementation flag)
        ...
```

### Test 2: Memory Overhead

```python
class TestMemory(TestCase):
    def test_memory_bounded(self):
        """Verify memory usage stays bounded during optimization"""
        dist.init_process_group(backend="nccl")

        model = MediumTransformer(dim=1024, layers=24).cuda()
        model = FSDP(model)
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)
        optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

        batch = torch.randn(4, 128, 1024).cuda()

        # Measure baseline memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()

        memory_before = torch.cuda.max_memory_allocated()

        # Run optimizer step
        optimizer.step()

        memory_after = torch.cuda.max_memory_allocated()

        # Memory increase should be bounded (< 2GB for prefetch buffer)
        memory_increase = memory_after - memory_before
        self.assertLess(memory_increase, 2 * 1024**3)  # 2GB

        dist.destroy_process_group()
```

---

## Regression Tests

### Test 1: Backward Compatibility

```python
class TestBackwardCompatibility(TestCase):
    def test_no_distributed_config_works(self):
        """Verify Muon works without distributed_config (backward compatible)"""
        model = TinyTransformer(dim=256, layers=4).cuda()
        optimizer = Muon(model.parameters(), lr=1e-3)  # No distributed_config

        batch = torch.randn(8, 128, 256).cuda()

        # Should work without error
        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()

        # No exception = success
```

### Test 2: Numerical Precision

```python
class TestNumericalPrecision(TestCase):
    def test_orthogonalization_preserves_precision(self):
        """Verify orthogonalization doesn't accumulate numerical errors"""
        dist.init_process_group(backend="nccl")

        model = TinyTransformer(dim=256, layers=4).cuda()
        model = FSDP(model)
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)
        optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

        batch = torch.randn(8, 128, 256).cuda()

        # Run 100 steps
        for _ in range(100):
            optimizer.zero_grad()
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()

        # Check parameters are still finite
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p).all())

        dist.destroy_process_group()
```

---

## CI/CD Strategy

### Commit-Level CI (P0 Tests)

**Runtime**: < 5 minutes
**Priority**: P0 (blocking, runs on every commit)

```yaml
# .github/workflows/test-muon-unit.yml
name: Muon Unit Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    container: pytorch/pytorch:latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: |
          python -m pytest test/optim/test_muon_unit.py -v
```

**Tests**:
- Unit tests (gather, redistribute, assign functions)
- Single-GPU correctness
- Backward compatibility

### PR-Level CI (P0 + P1 Tests)

**Runtime**: < 30 minutes
**Priority**: P0 + P1 (blocking, runs on every PR before merge)

```yaml
# .github/workflows/test-muon-integration.yml
name: Muon Integration Tests

on: [pull_request]

jobs:
  test-fsdp:
    runs-on: [self-hosted, gpu, 4x]
    steps:
      - uses: actions/checkout@v3
      - name: Test FSDP
        run: |
          torchrun --nproc_per_node=4 test/optim/test_muon_fsdp.py

  test-tp:
    runs-on: [self-hosted, gpu, 4x]
    steps:
      - uses: actions/checkout@v3
      - name: Test TP
        run: |
          torchrun --nproc_per_node=4 test/optim/test_muon_tp.py

  test-hsdp:
    runs-on: [self-hosted, gpu, 8x]
    steps:
      - uses: actions/checkout@v3
      - name: Test HSDP
        run: |
          torchrun --nproc_per_node=8 test/optim/test_muon_hsdp.py
```

**Tests**:
- FSDP correctness (4 GPUs)
- TP correctness (4 GPUs)
- HSDP correctness (8 GPUs)
- Hybrid TP+FSDP (8 GPUs)

### Weekly CI (P2 Tests)

**Runtime**: < 2 hours
**Priority**: P2 (non-blocking, runs weekly on schedule)

```yaml
# .github/workflows/test-muon-weekly.yml
name: Muon Weekly Tests

on:
  schedule:
    - cron: '0 0 * * 0'  # Sunday midnight

jobs:
  test-pp:
    runs-on: [self-hosted, gpu, 8x]
    steps:
      - uses: actions/checkout@v3
      - name: Test Pipeline Parallel
        run: |
          torchrun --nproc_per_node=8 test/optim/test_muon_pp.py

  test-4d-parallelism:
    runs-on: [self-hosted, gpu, 32x, multi-node]
    steps:
      - uses: actions/checkout@v3
      - name: Test PP + TP + HSDP
        run: |
          torchrun --nproc_per_node=8 --nnodes=4 test/optim/test_muon_4d.py

  benchmark:
    runs-on: [self-hosted, gpu, 8x]
    steps:
      - uses: actions/checkout@v3
      - name: Run performance benchmarks
        run: |
          python test/optim/benchmark_muon.py --output results.json
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results.json
```

**Tests**:
- Pipeline Parallel (8 GPUs)
- Full 4D parallelism (32 GPUs, multi-node)
- Performance benchmarks
- Large model tests (7B params)

---

## Test File Organization

```
pytorch/
├── test/
│   └── optim/
│       ├── test_muon_unit.py              # Unit tests (gather, redistribute, assign)
│       ├── test_muon_correctness.py        # Correctness vs single-GPU
│       ├── test_muon_fsdp.py              # FSDP integration tests
│       ├── test_muon_tp.py                # TP integration tests
│       ├── test_muon_hsdp.py              # HSDP integration tests
│       ├── test_muon_pp.py                # Pipeline parallel tests
│       ├── test_muon_4d.py                # Full 4D parallelism tests
│       ├── test_muon_performance.py        # Performance benchmarks
│       └── benchmark_muon.py              # Performance regression tracking
```

---

## Manual Testing Checklist

Before merging:

- [ ] Run correctness tests on 1, 2, 4, 8 GPUs
- [ ] Verify FSDP matches single-GPU on toy model
- [ ] Verify TP matches single-GPU on toy model
- [ ] Verify HSDP matches single-GPU on toy model
- [ ] Test gradient accumulation correctness
- [ ] Profile memory overhead (should be < 2GB per GPU)
- [ ] Profile performance speedup (should be > 4x on 8 GPUs)
- [ ] Test with different model sizes (10M, 100M, 1B)
- [ ] Test process group validation rejects invalid configs
- [ ] Test backward compatibility (no distributed_config)

---

**Document Status**: Living document, updated as testing evolves.
**Last Updated**: 2025-10-15
