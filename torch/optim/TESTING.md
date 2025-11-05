# Testing Strategy for Distributed Muon Optimizer

This document outlines the comprehensive testing strategy for the distributed Muon optimizer implementation, covering unit tests, integration tests, performance benchmarks, and correctness validation.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Environment Setup](#test-environment-setup)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [Performance Tests](#performance-tests)
6. [Edge Case Tests](#edge-case-tests)
7. [Correctness Validation](#correctness-validation)
8. [CI/CD Integration](#cicd-integration)
9. [Debugging Failed Tests](#debugging-failed-tests)

---

## Testing Philosophy

**Goals:**
1. **Correctness**: Distributed training produces numerically identical results to single-GPU training
2. **Performance**: Optimizations (prefetch, async) provide measurable speedups
3. **Robustness**: Handles edge cases, invalid inputs, and communication failures gracefully
4. **Compatibility**: Works across all supported parallelism strategies (FSDP, TP, DDP, etc.)

**Testing Pyramid:**
```
                   /\
                  /  \     E2E Tests (Few)
                 /----\
                /      \   Integration Tests (Some)
               /--------\
              /          \ Unit Tests (Many)
             /____________\
```

---

## Test Environment Setup

### Required Infrastructure

```python
# test_utils.py - Common testing utilities

import torch
import torch.distributed as dist
from torch.testing._internal.distributed.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from typing import Callable, Optional
import os

class MuonDistributedTestCase(MultiProcessTestCase):
    """Base class for distributed Muon tests."""

    @property
    def world_size(self) -> int:
        """Default world size for tests."""
        return 4

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def init_process_group(self, backend='nccl'):
        """Initialize distributed process group."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size
        )
        torch.cuda.set_device(self.rank)

    def destroy_process_group(self):
        """Clean up process group."""
        dist.destroy_process_group()


def create_dummy_model(num_params: int = 4, param_size: tuple = (128, 64)):
    """Create a simple model with 2D parameters for testing."""
    import torch.nn as nn

    layers = []
    for i in range(num_params):
        layers.append(nn.Linear(param_size[1], param_size[0], bias=False))

    model = nn.Sequential(*layers)
    return model


def seed_all(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def assert_tensors_equal(t1: torch.Tensor, t2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5):
    """Assert two tensors are equal within tolerance."""
    torch.testing.assert_close(t1, t2, rtol=rtol, atol=atol)
```

### Test Fixtures

```python
# test_fixtures.py - Common test fixtures

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

@pytest.fixture
def fsdp_process_group():
    """Fixture for FSDP process group."""
    if not dist.is_initialized():
        raise RuntimeError("Process group not initialized")

    pg = dist.new_group()  # World process group for FSDP
    yield pg

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group(pg)


@pytest.fixture
def tp_process_group():
    """Fixture for Tensor Parallel process group."""
    if not dist.is_initialized():
        raise RuntimeError("Process group not initialized")

    # Create TP group with first 2 ranks
    ranks = list(range(min(2, dist.get_world_size())))
    pg = dist.new_group(ranks=ranks)
    yield pg

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group(pg)


@pytest.fixture
def device_mesh_2d():
    """Fixture for 2D device mesh (DP x TP)."""
    from torch.distributed.device_mesh import init_device_mesh

    # Assumes 4 GPUs: 2x2 mesh
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])
    yield mesh
```

---

## Unit Tests

### 1. Test `assign_fn` - Parameter Assignment

```python
# test_assign_fn.py

import unittest
from torch.optim.muon import _default_assign_fn

class TestAssignFunction(unittest.TestCase):
    """Test parameter assignment logic."""

    def test_round_robin_assignment(self):
        """Test round-robin assignment distributes params evenly."""
        # Create 8 params, 4 ranks
        params = [torch.randn(10, 10) for _ in range(8)]
        state = {'world_size': 4}

        assignments = _default_assign_fn(params, state)

        # Verify all params assigned
        self.assertEqual(len(assignments), 8)

        # Verify assignments are in valid range
        for param_idx, rank in assignments.items():
            self.assertGreaterEqual(rank, 0)
            self.assertLess(rank, 4)

        # Verify round-robin pattern
        expected = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3}
        self.assertEqual(assignments, expected)

    def test_assignment_with_one_rank(self):
        """Test assignment with single rank (all assigned to rank 0)."""
        params = [torch.randn(10, 10) for _ in range(4)]
        state = {'world_size': 1}

        assignments = _default_assign_fn(params, state)

        for param_idx, rank in assignments.items():
            self.assertEqual(rank, 0)

    def test_assignment_more_ranks_than_params(self):
        """Test assignment when world_size > num_params."""
        params = [torch.randn(10, 10) for _ in range(2)]
        state = {'world_size': 8}

        assignments = _default_assign_fn(params, state)

        # Each param assigned to different rank
        self.assertEqual(assignments[0], 0)
        self.assertEqual(assignments[1], 1)

    def test_load_balanced_assignment(self):
        """Test load-balanced assignment by parameter size."""
        # Create params with varying sizes
        params = [
            torch.randn(100, 100),  # 10K params
            torch.randn(10, 10),    # 100 params
            torch.randn(50, 50),    # 2.5K params
            torch.randn(200, 200),  # 40K params
        ]
        state = {'world_size': 2}

        # Load-balanced assignment should assign largest to different ranks
        from torch.optim.muon import _load_balanced_assign_fn
        assignments = _load_balanced_assign_fn(params, state)

        # Largest (idx=3, 40K) and second largest (idx=0, 10K) to different ranks
        self.assertNotEqual(assignments[3], assignments[0])


### 2. Test `gather_fn` - Gathering Shards

```python
# test_gather_fn.py

class TestGatherFunction(MuonDistributedTestCase):
    """Test gather function for different parallelism strategies."""

    @skip_if_lt_x_gpu(4)
    def test_fsdp_gather(self):
        """Test gathering FSDP sharded tensor."""
        self.init_process_group()

        # Create sharded tensor (each rank has a shard)
        shard_size = 64
        full_size = shard_size * self.world_size
        local_shard = torch.randn(shard_size, 32).cuda(self.rank)

        # Setup gather_fn for FSDP
        from torch.optim.muon import create_processgroup_config
        config = create_processgroup_config(fsdp_pg=dist.group.WORLD)

        # Test gathering to rank 0
        dst_rank = 0
        result = config.gather_fn(local_shard, dst_rank, config.state)

        if self.rank == dst_rank:
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (full_size, 32))
        else:
            self.assertIsNone(result)

        self.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_ddp_gather(self):
        """Test gathering DDP replicated tensor (no actual gather needed)."""
        self.init_process_group()

        # Create replicated tensor (same on all ranks)
        tensor = torch.randn(128, 64).cuda(self.rank)

        from torch.optim.muon import create_processgroup_config
        config = create_processgroup_config(dp_pg=dist.group.WORLD)

        # Gather on rank 0
        dst_rank = 0
        result = config.gather_fn(tensor, dst_rank, config.state)

        if self.rank == dst_rank:
            self.assertIsNotNone(result)
            assert_tensors_equal(result, tensor)
        else:
            self.assertIsNone(result)

        self.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_gather_async(self):
        """Test async gather with prefetching."""
        self.init_process_group()

        shard_size = 64
        local_shard = torch.randn(shard_size, 32).cuda(self.rank)

        from torch.optim.muon import _gather_fn_async

        # Start async gather
        result, work = _gather_fn_async(
            local_shard,
            dst_rank=0,
            state={'rank': self.rank, 'world_size': self.world_size, 'fsdp_pg': dist.group.WORLD},
            async_op=True
        )

        # Work should be in progress
        self.assertIsNotNone(work)

        # Wait for completion
        work.wait()

        # Now result should be ready
        if self.rank == 0:
            self.assertIsNotNone(result)

        self.destroy_process_group()


### 3. Test `redistribute_fn` - Scattering Updates

```python
# test_redistribute_fn.py

class TestRedistributeFunction(MuonDistributedTestCase):
    """Test redistribute function for different parallelism strategies."""

    @skip_if_lt_x_gpu(4)
    def test_fsdp_redistribute(self):
        """Test redistributing (scattering) tensor to FSDP shards."""
        self.init_process_group()

        full_size = 256
        shard_size = full_size // self.world_size

        from torch.optim.muon import create_processgroup_config
        config = create_processgroup_config(fsdp_pg=dist.group.WORLD)

        # Rank 0 has full tensor, others have None
        if self.rank == 0:
            full_tensor = torch.randn(full_size, 32).cuda(self.rank)
        else:
            full_tensor = None

        # Redistribute from rank 0 to all ranks
        result = config.redistribute_fn(full_tensor, src_rank=0, state=config.state)

        # All ranks should receive their shard
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (shard_size, 32))

        self.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_ddp_redistribute(self):
        """Test redistributing (broadcasting) tensor to DDP replicas."""
        self.init_process_group()

        from torch.optim.muon import create_processgroup_config
        config = create_processgroup_config(dp_pg=dist.group.WORLD)

        # Rank 0 has tensor, others have None
        if self.rank == 0:
            tensor = torch.randn(128, 64).cuda(self.rank)
        else:
            tensor = None

        # Broadcast from rank 0
        result = config.redistribute_fn(tensor, src_rank=0, state=config.state)

        # All ranks should have full replica
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (128, 64))

        # Verify all ranks have same values
        # Gather all results to rank 0 for comparison
        if self.rank == 0:
            gathered = [torch.empty_like(result) for _ in range(self.world_size)]
            dist.gather(result, gathered, dst=0)

            for i in range(1, self.world_size):
                assert_tensors_equal(gathered[0], gathered[i])
        else:
            dist.gather(result, dst=0)

        self.destroy_process_group()


### 4. Test Helper Functions

```python
# test_helpers.py

class TestHelperFunctions(unittest.TestCase):
    """Test helper functions for creating DistributedConfig."""

    def test_create_processgroup_config_fsdp(self):
        """Test creating config for FSDP."""
        from torch.optim.muon import create_processgroup_config

        # Mock process group
        pg = dist.new_group()

        config = create_processgroup_config(fsdp_pg=pg, prefetch_count=2, async_gpu_parallelism=False)

        self.assertIsNotNone(config)
        self.assertEqual(config.prefetch_count, 2)
        self.assertEqual(config.async_gpu_parallelism, False)
        self.assertIn('fsdp_pg', config.state)
        self.assertIsNotNone(config.assign_fn)
        self.assertIsNotNone(config.gather_fn)
        self.assertIsNotNone(config.redistribute_fn)

    def test_create_devicemesh_config(self):
        """Test creating config from DeviceMesh."""
        from torch.optim.muon import create_devicemesh_config
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])
        config = create_devicemesh_config(mesh, ["dp", "tp"])

        self.assertIsNotNone(config)
        self.assertIn('device_mesh', config.state)

    def test_create_dtensor_config(self):
        """Test creating config for DTensor."""
        from torch.optim.muon import create_dtensor_config

        config = create_dtensor_config(prefetch_count=1, async_gpu_parallelism=True)

        self.assertIsNotNone(config)
        self.assertEqual(config.prefetch_count, 1)
        self.assertEqual(config.async_gpu_parallelism, True)
```

---

## Integration Tests

### 1. Distributed vs Non-Distributed Equivalence

```python
# test_equivalence.py

class TestDistributedEquivalence(MuonDistributedTestCase):
    """Test that distributed and non-distributed training produce identical results."""

    @skip_if_lt_x_gpu(4)
    def test_fsdp_equivalence(self):
        """Test FSDP distributed matches single-GPU."""
        self.init_process_group()
        seed_all(42)

        # Create model
        model = create_dummy_model(num_params=4, param_size=(128, 64))
        model = model.cuda(self.rank)

        # Wrap with FSDP
        model = FSDP(model)

        # Create optimizer with distributed config
        from torch.optim import Muon
        from torch.optim.muon import create_processgroup_config

        optimizer = Muon(
            model.parameters(),
            lr=0.02,
            momentum=0.95,
            distributed_config=create_processgroup_config(fsdp_pg=model.process_group)
        )

        # Training loop
        num_steps = 10
        for step in range(num_steps):
            # Forward pass with dummy data
            input_data = torch.randn(32, 64).cuda(self.rank)
            output = model(input_data)
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

        # Gather final parameters to rank 0
        final_params = []
        for param in model.parameters():
            # Gather FSDP shards
            with FSDP.summon_full_params(model):
                if self.rank == 0:
                    final_params.append(param.detach().cpu().clone())

        # Compare with single-GPU baseline (run separately and save)
        if self.rank == 0:
            # Load baseline params from file
            baseline_params = torch.load('baseline_params.pt')

            for i, (p_dist, p_baseline) in enumerate(zip(final_params, baseline_params)):
                assert_tensors_equal(p_dist, p_baseline, rtol=1e-5, atol=1e-5)

        self.destroy_process_group()

    @skip_if_lt_x_gpu(1)
    def test_single_gpu_baseline(self):
        """Generate single-GPU baseline for comparison."""
        seed_all(42)

        # Create model (same as distributed test)
        model = create_dummy_model(num_params=4, param_size=(128, 64))
        model = model.cuda(0)

        # Create optimizer WITHOUT distributed config
        from torch.optim import Muon
        optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)

        # Training loop (same as distributed test)
        num_steps = 10
        for step in range(num_steps):
            input_data = torch.randn(32, 64).cuda(0)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save final parameters
        final_params = [p.detach().cpu().clone() for p in model.parameters()]
        torch.save(final_params, 'baseline_params.pt')


### 2. Test Combined Parallelism Strategies

```python
# test_combined_parallelism.py

class TestCombinedParallelism(MuonDistributedTestCase):
    """Test combinations of parallelism strategies."""

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_combined(self):
        """Test FSDP + TP combination."""
        self.init_process_group()

        from torch.distributed.device_mesh import init_device_mesh
        from torch.optim.muon import create_devicemesh_config

        # Create 2x2 mesh (DP=2, TP=2)
        device_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=["dp", "tp"])

        # Create model and apply parallelism
        model = create_dummy_model(num_params=4, param_size=(128, 64))
        # TODO: Apply TP and FSDP to model

        # Create optimizer
        from torch.optim import Muon
        optimizer = Muon(
            model.parameters(),
            lr=0.02,
            distributed_config=create_devicemesh_config(device_mesh, ["dp", "tp"])
        )

        # Run training step
        input_data = torch.randn(32, 64).cuda(self.rank)
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Test passes if no errors
        self.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_hsdp(self):
        """Test Hybrid Sharded Data Parallel."""
        # HSDP = FSDP with both replicate and shard dimensions
        self.init_process_group()

        # Create model with HSDP
        model = create_dummy_model()
        # TODO: Apply HSDP wrapping

        from torch.optim import Muon
        from torch.optim.muon import create_processgroup_config

        optimizer = Muon(
            model.parameters(),
            lr=0.02,
            distributed_config=create_processgroup_config(fsdp_pg=model.process_group)
        )

        # Run training
        input_data = torch.randn(32, 64).cuda(self.rank)
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        self.destroy_process_group()
```

---

## Performance Tests

### 1. Prefetch Speedup

```python
# test_performance.py

class TestPerformance(MuonDistributedTestCase):
    """Test performance optimizations."""

    @skip_if_lt_x_gpu(4)
    def test_prefetch_speedup(self):
        """Verify prefetching reduces wall-clock time."""
        self.init_process_group()

        model = create_dummy_model(num_params=20, param_size=(512, 512))
        model = FSDP(model)

        from torch.optim import Muon
        from torch.optim.muon import create_processgroup_config

        # Benchmark WITHOUT prefetch
        optimizer_no_prefetch = Muon(
            model.parameters(),
            lr=0.02,
            distributed_config=create_processgroup_config(
                fsdp_pg=model.process_group,
                prefetch_count=0,
                async_gpu_parallelism=False
            )
        )

        time_no_prefetch = self._benchmark_optimizer(model, optimizer_no_prefetch, steps=50)

        # Benchmark WITH prefetch
        optimizer_prefetch = Muon(
            model.parameters(),
            lr=0.02,
            distributed_config=create_processgroup_config(
                fsdp_pg=model.process_group,
                prefetch_count=1,
                async_gpu_parallelism=False
            )
        )

        time_prefetch = self._benchmark_optimizer(model, optimizer_prefetch, steps=50)

        # Verify speedup (should be at least 10% faster in bandwidth-limited scenarios)
        if self.rank == 0:
            speedup = time_no_prefetch / time_prefetch
            print(f"Prefetch speedup: {speedup:.2f}x")
            self.assertGreater(speedup, 1.1)  # At least 10% speedup

        self.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_async_parallelism_speedup(self):
        """Verify async GPU parallelism reduces wall-clock time."""
        self.init_process_group()

        model = create_dummy_model(num_params=20, param_size=(512, 512))
        model = FSDP(model)

        from torch.optim import Muon
        from torch.optim.muon import create_processgroup_config

        # Benchmark synchronous
        optimizer_sync = Muon(
            model.parameters(),
            lr=0.02,
            distributed_config=create_processgroup_config(
                fsdp_pg=model.process_group,
                prefetch_count=0,
                async_gpu_parallelism=False
            )
        )

        time_sync = self._benchmark_optimizer(model, optimizer_sync, steps=50)

        # Benchmark async
        optimizer_async = Muon(
            model.parameters(),
            lr=0.02,
            distributed_config=create_processgroup_config(
                fsdp_pg=model.process_group,
                prefetch_count=0,
                async_gpu_parallelism=True
            )
        )

        time_async = self._benchmark_optimizer(model, optimizer_async, steps=50)

        # Verify speedup
        if self.rank == 0:
            speedup = time_sync / time_async
            print(f"Async speedup: {speedup:.2f}x")
            self.assertGreater(speedup, 1.2)  # At least 20% speedup

        self.destroy_process_group()

    def _benchmark_optimizer(self, model, optimizer, steps: int) -> float:
        """Helper to benchmark optimizer performance."""
        import time

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(steps):
            input_data = torch.randn(32, 512).cuda(self.rank)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        end = time.time()

        return end - start
```

---

## Edge Case Tests

### 1. Error Handling

```python
# test_edge_cases.py

class TestEdgeCases(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_invalid_rank_assignment(self):
        """Test that invalid rank assignments raise error."""
        from torch.optim.muon import DistributedConfig

        params = [torch.randn(10, 10) for _ in range(4)]

        # Create assign_fn that returns invalid rank
        def bad_assign_fn(params, state):
            # Assign to rank outside world_size
            return {i: 999 for i in range(len(params))}

        def dummy_fn(*args, **kwargs):
            return None

        config = DistributedConfig(
            assign_fn=bad_assign_fn,
            gather_fn=dummy_fn,
            redistribute_fn=dummy_fn,
            state={'world_size': 4}
        )

        # Should raise ValueError when validating
        from torch.optim.muon import _validate_assignments
        with self.assertRaises(ValueError):
            _validate_assignments(bad_assign_fn(params, config.state), params, 4)

    def test_missing_parameter_assignment(self):
        """Test that missing parameter assignments raise error."""
        from torch.optim.muon import DistributedConfig

        params = [torch.randn(10, 10) for _ in range(4)]

        # assign_fn that skips some parameters
        def incomplete_assign_fn(params, state):
            return {0: 0, 1: 1}  # Missing params 2 and 3

        def dummy_fn(*args, **kwargs):
            return None

        config = DistributedConfig(
            assign_fn=incomplete_assign_fn,
            gather_fn=dummy_fn,
            redistribute_fn=dummy_fn,
            state={'world_size': 4}
        )

        from torch.optim.muon import _validate_assignments
        with self.assertRaises(AssertionError):
            _validate_assignments(incomplete_assign_fn(params, config.state), params, 4)

    def test_shape_mismatch_in_gather(self):
        """Test that shape mismatches in gather are caught."""
        # TODO: Implement shape validation in gather_fn
        pass

    def test_non_2d_parameter_rejected(self):
        """Test that non-2D parameters are rejected."""
        from torch.optim import Muon

        # Create model with 1D parameter (bias)
        model = torch.nn.Linear(10, 10, bias=True)

        # Should raise ValueError
        with self.assertRaises(ValueError):
            optimizer = Muon(model.parameters(), lr=0.02)

    def test_zero_parameters(self):
        """Test optimizer with zero parameters."""
        from torch.optim import Muon

        # Empty parameter list
        optimizer = Muon([], lr=0.02)

        # Should not crash on step
        optimizer.step()
```

---

## Correctness Validation

### User Validation Guide

```python
# validate_distributed_setup.py - Script for users to validate their setup

"""
Script to validate that distributed Muon produces correct results.

Usage:
    # Single-GPU baseline
    python validate_distributed_setup.py --mode baseline --model_name my_model

    # Multi-GPU distributed
    torchrun --nproc_per_node=4 validate_distributed_setup.py --mode distributed --model_name my_model

    # Compare results
    python validate_distributed_setup.py --mode compare --model_name my_model
"""

import torch
from torch.optim import Muon
from torch.optim.muon import create_processgroup_config
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import argparse

def create_model(model_name: str):
    """Create model based on name."""
    # User implements their model creation logic
    pass

def train_single_gpu(model, num_steps: int = 100, lr: float = 0.02):
    """Train model on single GPU."""
    device = torch.device("cuda:0")
    model = model.to(device)

    optimizer = Muon(model.parameters(), lr=lr, momentum=0.95)

    for step in range(num_steps):
        # Generate dummy data
        input_data = torch.randn(32, model.input_dim).to(device)

        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save final parameters
    final_params = {name: param.detach().cpu() for name, param in model.named_parameters()}
    torch.save(final_params, f'{model_name}_baseline_params.pt')
    print(f"Baseline parameters saved to {model_name}_baseline_params.pt")

def train_distributed(model, num_steps: int = 100, lr: float = 0.02):
    """Train model with distributed FSDP."""
    import torch.distributed as dist

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    device = torch.device(f"cuda:{rank}")
    model = model.to(device)

    # Wrap with FSDP
    model = FSDP(model)

    # Create optimizer with distributed config
    optimizer = Muon(
        model.parameters(),
        lr=lr,
        momentum=0.95,
        distributed_config=create_processgroup_config(fsdp_pg=model.process_group)
    )

    for step in range(num_steps):
        input_data = torch.randn(32, model.input_dim).to(device)

        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Gather final parameters
    final_params = {}
    with FSDP.summon_full_params(model):
        if rank == 0:
            final_params = {name: param.detach().cpu() for name, param in model.named_parameters()}
            torch.save(final_params, f'{model_name}_distributed_params.pt')
            print(f"Distributed parameters saved to {model_name}_distributed_params.pt")

    dist.destroy_process_group()

def compare_results(model_name: str):
    """Compare baseline and distributed results."""
    baseline = torch.load(f'{model_name}_baseline_params.pt')
    distributed = torch.load(f'{model_name}_distributed_params.pt')

    print(f"\nComparing {model_name} results:")
    print("=" * 60)

    all_close = True
    for name in baseline.keys():
        p_baseline = baseline[name]
        p_distributed = distributed[name]

        try:
            torch.testing.assert_close(p_baseline, p_distributed, rtol=1e-5, atol=1e-5)
            print(f"✓ {name}: PASS")
        except AssertionError as e:
            print(f"✗ {name}: FAIL")
            print(f"  Error: {e}")
            all_close = False

    print("=" * 60)
    if all_close:
        print("✓ All parameters match! Distributed setup is correct.")
    else:
        print("✗ Some parameters don't match. Check your distributed configuration.")

    return all_close

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'distributed', 'compare'], required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.02)
    args = parser.parse_args()

    if args.mode == 'baseline':
        model = create_model(args.model_name)
        train_single_gpu(model, args.num_steps, args.lr)

    elif args.mode == 'distributed':
        model = create_model(args.model_name)
        train_distributed(model, args.num_steps, args.lr)

    elif args.mode == 'compare':
        compare_results(args.model_name)
```

### Expected Tolerances

When comparing results, use these tolerances:

| **Precision** | **rtol** | **atol** | **Notes** |
|---------------|----------|----------|-----------|
| FP32 | 1e-5 | 1e-5 | Standard precision, tight tolerance |
| BF16 | 1e-3 | 1e-3 | Lower precision, more accumulated error |
| FP16 | 1e-3 | 1e-3 | Lower precision, more accumulated error |
| Mixed Precision | 1e-4 | 1e-4 | Balance between FP32 and FP16/BF16 |

**Why tolerances matter:**
- Distributed reductions (all_gather, reduce_scatter) may use different algorithms
- Floating-point operations are not associative: `(a + b) + c ≠ a + (b + c)`
- Different GPU architectures may have slightly different results
- Communication overhead can lead to minor numerical differences

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test_muon_distributed.yml

name: Test Muon Distributed

on:
  push:
    branches: [main]
    paths:
      - 'torch/optim/_muon.py'
      - 'torch/optim/muon/**'
      - 'test/distributed/optim/test_muon_distributed.py'
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install torch
          pip install -e .

      - name: Run unit tests
        run: |
          pytest test/optim/test_muon_assign_fn.py
          pytest test/optim/test_muon_helpers.py

  distributed-tests:
    runs-on: self-hosted  # Requires multi-GPU machine
    strategy:
      matrix:
        world_size: [2, 4]
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install torch
          pip install -e .

      - name: Run distributed tests
        run: |
          torchrun --nproc_per_node=${{ matrix.world_size }} \
            test/distributed/optim/test_muon_distributed.py

  performance-tests:
    runs-on: self-hosted  # Requires multi-GPU machine
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Run performance benchmarks
        run: |
          torchrun --nproc_per_node=4 \
            test/distributed/optim/benchmark_muon.py \
            --output=benchmark_results.json

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results.json
```

### Test Organization

```
pytorch/
├── test/
│   ├── optim/
│   │   ├── test_muon_assign_fn.py          # Unit tests (no GPU needed)
│   │   ├── test_muon_helpers.py            # Unit tests (no GPU needed)
│   │   └── test_muon_edge_cases.py         # Unit tests (no GPU needed)
│   │
│   └── distributed/
│       └── optim/
│           ├── test_muon_gather_fn.py      # Distributed tests (multi-GPU)
│           ├── test_muon_redistribute_fn.py # Distributed tests (multi-GPU)
│           ├── test_muon_equivalence.py    # Integration tests (multi-GPU)
│           ├── test_muon_combined.py       # Combined parallelism tests
│           ├── test_muon_performance.py    # Performance tests
│           └── benchmark_muon.py           # Benchmarking script
```

---

## Debugging Failed Tests

### Common Failure Patterns

#### 1. Numerical Mismatch

**Symptom:**
```
AssertionError: Tensor-likes are not close!
Mismatched elements: 128 / 8192 (1.6%)
Greatest absolute difference: 0.0012 at index (5, 3)
Greatest relative difference: 0.002 at index (7, 1)
```

**Debugging Steps:**
1. Check precision: Are you using FP16/BF16? Adjust tolerances
2. Verify seeding: Are all ranks using the same random seed?
3. Check communication: Is gather/redistribute working correctly?
4. Compare step-by-step: Add logging to compare intermediate values

```python
# Add debugging in optimizer step
def step(self):
    for i, param in enumerate(self.params):
        if self.rank == 0:
            print(f"Param {i} before update: {param.norm().item()}")

        # ... optimizer logic ...

        if self.rank == 0:
            print(f"Param {i} after update: {param.norm().item()}")
```

#### 2. Deadlock

**Symptom:** Test hangs indefinitely, no error message

**Common Causes:**
- Not all ranks calling collective operations
- Mismatched collective operations (e.g., one rank calls gather, another calls scatter)
- Missing barrier synchronization

**Debugging Steps:**
1. Add timeouts:
```python
import torch.distributed as dist
dist.monitored_barrier()  # Will error if any rank fails
```

2. Add logging:
```python
if self.rank == 0:
    print(f"[Rank {self.rank}] About to call gather")
dist.all_gather(...)
if self.rank == 0:
    print(f"[Rank {self.rank}] Finished gather")
```

3. Use environment variable:
```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun test.py
```

#### 3. Shape Mismatch

**Symptom:**
```
RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 256 but got size 64 for dimension 1.
```

**Debugging Steps:**
1. Print shapes at every step:
```python
print(f"[Rank {self.rank}] Momentum buffer shape: {momentum_buf.shape}")
print(f"[Rank {self.rank}] Gathered shape: {momentum_buf_full.shape if momentum_buf_full is not None else 'None'}")
```

2. Verify metadata in state:
```python
print(f"State metadata: {self.distributed_config.state}")
```

3. Check shard dimensions:
```python
# For FSDP, verify shard size
expected_shard_size = full_size // world_size
assert local_shard.size(0) == expected_shard_size
```

#### 4. Communication Failure

**Symptom:**
```
RuntimeError: NCCL error in: collective.cc:123
[Rank 2] Watchdog caught collective operation timeout
```

**Debugging Steps:**
1. Check network connectivity between ranks
2. Verify NCCL is properly installed
3. Use alternative backend for debugging:
```python
dist.init_process_group(backend='gloo')  # CPU backend, slower but more stable
```

4. Reduce problem size to isolate issue

### Test Debugging Checklist

Before filing a bug report, verify:

- [ ] Single-GPU (non-distributed) version works correctly
- [ ] All ranks complete initialization
- [ ] Process groups are created correctly
- [ ] Shapes match at gather/redistribute boundaries
- [ ] Random seeds are set consistently
- [ ] Tolerances are appropriate for precision
- [ ] No CUDA errors (check `torch.cuda.is_available()`)
- [ ] Sufficient GPU memory available
- [ ] All collective operations are called by all ranks
- [ ] Logging shows expected execution flow

### Useful Debugging Commands

```bash
# Enable verbose distributed logging
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Monitor GPU memory
nvidia-smi -l 1

# Check distributed setup
python -c "import torch; import torch.distributed as dist; dist.init_process_group('nccl'); print(f'Rank {dist.get_rank()} of {dist.get_world_size()}')"

# Profile distributed training
python -m torch.profiler test_distributed.py
```

---

## Summary

This testing strategy provides:

1. **Comprehensive Coverage**: Unit tests (components), integration tests (full system), performance tests (optimizations)
2. **Correctness Validation**: Tools for users to verify their distributed setup
3. **CI/CD Integration**: Automated testing on every commit
4. **Debugging Guide**: Common failures and how to fix them

**Test Priorities:**
- **P0 (Must Have)**: Unit tests for assign/gather/redistribute, single equivalence test
- **P1 (Should Have)**: Integration tests for FSDP/TP/DDP, performance benchmarks
- **P2 (Nice to Have)**: Combined parallelism tests, edge cases, profiling

**Next Steps:**
1. Implement P0 tests during Phase 1 of development
2. Add P1 tests during Phase 2-3
3. Continuously expand test coverage as new features are added
