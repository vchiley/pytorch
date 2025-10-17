# Muon Distributed Training: Testing Strategy

This document outlines the testing approach for distributed Muon training.

**Related Documents**:
- `PROJECT.md` - High-level overview and problem statement
- `DESIGN.md` - Complete API specification and implementation details
- `EXAMPLES.md` - Concrete usage patterns and examples

## Testing Framework

> **Note on Test Code**: Throughout these test examples, `SimpleModel`, `SimpleMoE`, `LargeModel`, and similar class names represent placeholder `torch.nn.Module` instances for testing. Replace these with your actual model classes or test fixtures.

Use PyTorch's testing infrastructure:

```python
from torch.testing._internal.common_utils import run_tests, TestCase

class TestMuonDistributed(TestCase):
    def test_feature(self):
        # Test implementation
        ...

if __name__ == "__main__":
    run_tests()
```

## Test Categories

### 1. Unit Tests (Single-Device)

Test core functionality without distribution:

```python
class TestMuonCore(TestCase):
    def test_backward_compatibility(self):
        """Verify distributed_config=None works correctly."""
        model = SimpleModel()
        optimizer = Muon(model.parameters(), lr=0.01, distributed_config=None)

        # Run optimizer step
        loss = model(input_data).sum()
        loss.backward()
        optimizer.step()

        # Should complete without error
        self.assertTrue(True)

    def test_orthogonalization_correctness(self):
        """Test Newton-Schulz orthogonalization produces orthogonal matrices."""
        from torch.optim._muon import _zeropower_via_newtonschulz

        # Create test matrix
        matrix = torch.randn(64, 64)

        # Orthogonalize
        result = _zeropower_via_newtonschulz(matrix, steps=5)

        # Verify orthogonality: result @ result.T ≈ I
        product = result @ result.T
        identity = torch.eye(64)
        self.assertTrue(torch.allclose(product, identity, atol=1e-5))
```

### 2. DTensor Configuration Tests

Test DTensor-based distributed configuration:

```python
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor
from torch.optim.muon import create_dtensor_config

class TestDTensorConfig(TestCase):
    def test_dtensor_config_creation(self):
        """Verify create_dtensor_config() produces valid config."""
        config = create_dtensor_config()

        # Config should have required functions
        self.assertTrue(callable(config.assign_fn))
        self.assertTrue(callable(config.gather_fn))
        self.assertTrue(callable(config.redistribute_fn))

        # State should indicate DTensor mode
        self.assertTrue(config.state.get('is_dtensor', False))

    def test_dtensor_gather_full_tensor(self):
        """Test gather_fn with DTensor.full_tensor()."""
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

        # Create simple 1D device mesh
        device_mesh = DeviceMesh("cpu", [0])

        # Create DTensor with Shard placement
        tensor = torch.randn(10, 10)
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])

        # Create config and test gather
        config = create_dtensor_config()
        full_tensor = config.gather_fn(dtensor, 0, config.state)

        # Should return full tensor (regular Tensor, not DTensor)
        self.assertEqual(full_tensor.shape, (10, 10))
        self.assertFalse(isinstance(full_tensor, DTensor))
        self.assertTrue(isinstance(full_tensor, torch.Tensor))

    def test_dtensor_redistribute(self):
        """Test redistribute_fn recreates DTensor with original placement."""
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

        device_mesh = DeviceMesh("cpu", [0])

        # Create DTensor parameters
        params = [
            distribute_tensor(torch.randn(10, 10), device_mesh, [Shard(0)])
            for _ in range(2)
        ]

        # Create config and run assign_fn to populate metadata
        config = create_dtensor_config()
        assignments = config.assign_fn(params, config.state)

        # Test redistribute for param 0
        config.state['current_param_idx'] = 0
        full_update = torch.randn(10, 10)
        redistributed = config.redistribute_fn(full_update, 0, config.state)

        # Should be DTensor with same placement
        self.assertTrue(isinstance(redistributed, DTensor))
        self.assertEqual(redistributed.device_mesh, device_mesh)
        self.assertEqual(redistributed.placements, [Shard(0)])

    def test_dtensor_replicate_placement(self):
        """Test DTensor with Replicate placement (DDP-like)."""
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

        device_mesh = DeviceMesh("cpu", [0])

        # Create replicated DTensor
        tensor = torch.randn(10, 10)
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])

        # Create config
        config = create_dtensor_config()

        # Gather should return tensor as-is (no-op for replicated)
        full_tensor = config.gather_fn(dtensor, 0, config.state)
        self.assertEqual(full_tensor.shape, tensor.shape)

    def test_dtensor_mixed_regular_params(self):
        """Test handling of mixed DTensor and regular tensor parameters."""
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

        device_mesh = DeviceMesh("cpu", [0])

        # Mix of DTensor and regular tensors
        params = [
            distribute_tensor(torch.randn(10, 10), device_mesh, [Shard(0)]),
            torch.randn(5, 5),  # Regular tensor
            distribute_tensor(torch.randn(8, 8), device_mesh, [Replicate()]),
        ]

        # Config should handle both
        config = create_dtensor_config()
        assignments = config.assign_fn(params, config.state)

        # All params should be assigned
        all_assigned = set()
        for rank_assignments in assignments.values():
            all_assigned.update(rank_assignments)
        self.assertEqual(all_assigned, {0, 1, 2})

        # DTensor params should have metadata stored
        # (see DTensor-Based Examples in EXAMPLES.md for details on param.data checking)
        self.assertIn(0, config.state['param_metadata'])
        self.assertNotIn(1, config.state['param_metadata'])  # Regular tensor
        self.assertIn(2, config.state['param_metadata'])

        # Verify metadata contains device_mesh and placements
        self.assertIn('device_mesh', config.state['param_metadata'][0])
        self.assertIn('placements', config.state['param_metadata'][0])

### 3. Configuration Validation Tests

Test DistributedConfig validation:

```python
class TestDistributedConfig(TestCase):
    def test_assign_fn_coverage(self):
        """Verify all parameters are assigned exactly once."""
        params = [torch.randn(10, 10) for _ in range(8)]
        state = {'rank': 0, 'world_size': 4}

        def assign_fn(params, state):
            assignments = {i: [] for i in range(4)}
            for idx in range(len(params)):
                assignments[idx % 4].append(idx)
            return assignments

        assignments = assign_fn(params, state)

        # Check all params assigned
        all_assigned = set()
        for rank_assignments in assignments.values():
            all_assigned.update(rank_assignments)

        self.assertEqual(all_assigned, set(range(len(params))))

        # Check no overlaps
        total_assigned = sum(len(v) for v in assignments.values())
        self.assertEqual(total_assigned, len(params))

    def test_tp_dim_per_param_validation(self):
        """Verify tp_dim_per_param is correctly handled."""
        from torch.optim.muon import create_auto_config

        # Should accept dict
        tp_dims = {0: 0, 1: 1, 2: 0}
        config = create_auto_config(
            tp_process_group=None,
            tp_dim_per_param=tp_dims
        )
        self.assertTrue('tp_dim_per_param' in config.state)

        # Should accept single int
        config = create_auto_config(
            tp_process_group=None,
            tp_dim_per_param=1
        )
        self.assertTrue('tp_dim_default' in config.state)
```

### 3. DTensor Distributed Integration Tests

Test DTensor configurations with actual distributed setup:

```python
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, distribute_tensor
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.optim.muon import create_dtensor_config

class TestMuonDTensor(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_dtensor_shard_correctness(self):
        """Test DTensor with Shard placement produces correct results."""
        # Setup
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size
        )

        # Create device mesh
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        # Create simple model
        model = torch.nn.Linear(64, 64).cuda()

        # Convert parameters to DTensor with sharding
        for name, param in model.named_parameters():
            if 'weight' in name:
                param_dtensor = distribute_tensor(
                    param.data,
                    device_mesh=device_mesh,
                    placements=[Shard(0)],  # Shard along dim 0
                )
                param.data = param_dtensor

        # Create optimizer with DTensor config
        config = create_dtensor_config()
        optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

        # Run training step
        input_data = torch.randn(8, 64).cuda()
        loss = model(input_data).sum()
        loss.backward()
        optimizer.step()

        # Verify step completed
        self.assertTrue(True)

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_dtensor_vs_single_device(self):
        """Compare DTensor sharded results with single-device baseline."""
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size
        )

        # Seed for reproducibility
        torch.manual_seed(42)

        # Create identical models
        model_single = torch.nn.Linear(32, 32).cuda()

        # Model with DTensor sharding
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        model_dtensor = torch.nn.Linear(32, 32).cuda()

        # Sync initial weights
        with torch.no_grad():
            model_dtensor.weight.copy_(model_single.weight)
            model_dtensor.bias.copy_(model_single.bias)

        # Convert to DTensor
        for name, param in model_dtensor.named_parameters():
            if 'weight' in name:
                param.data = distribute_tensor(
                    param.data,
                    device_mesh=device_mesh,
                    placements=[Shard(0)],
                )

        # Create optimizers
        config_dtensor = create_dtensor_config()
        optimizer_dtensor = Muon(
            model_dtensor.parameters(),
            lr=0.01,
            distributed_config=config_dtensor
        )
        optimizer_single = Muon(model_single.parameters(), lr=0.01)

        # Run multiple steps
        for step in range(5):
            torch.manual_seed(42 + step)
            input_data = torch.randn(4, 32).cuda()

            # DTensor forward/backward/step
            loss_dtensor = model_dtensor(input_data).sum()
            loss_dtensor.backward()
            optimizer_dtensor.step()
            optimizer_dtensor.zero_grad()

            # Single-device forward/backward/step (rank 0 only)
            if self.rank == 0:
                loss_single = model_single(input_data).sum()
                loss_single.backward()
                optimizer_single.step()
                optimizer_single.zero_grad()

        # Compare final parameters (rank 0)
        if self.rank == 0:
            # Get full DTensor parameters
            # (see DTensor-Based Examples in EXAMPLES.md for details on param.data)
            dtensor_weight = model_dtensor.weight.data
            if isinstance(dtensor_weight, DTensor):
                full_weight = dtensor_weight.full_tensor()
            else:
                full_weight = dtensor_weight

            single_weight = model_single.weight

            # Compare
            self.assertTrue(
                torch.allclose(full_weight, single_weight, atol=1e-4),
                "DTensor and single-device weights should match"
            )

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_dtensor_2d_mesh(self):
        """Test DTensor with 2D device mesh (FSDP-like + TP-like)."""
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=4
        )

        # Create 2D mesh: 2x2
        device_mesh = DeviceMesh(
            "cuda",
            torch.arange(4).reshape(2, 2),
            mesh_dim_names=("fsdp", "tp"),
        )

        # Create model
        model = torch.nn.Linear(128, 128).cuda()

        # Shard along both dimensions
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.data = distribute_tensor(
                    param.data,
                    device_mesh=device_mesh,
                    placements=[Shard(0), Shard(1)],  # 2D sharding
                )

        # Create optimizer
        config = create_dtensor_config()
        optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

        # Run training step
        input_data = torch.randn(8, 128).cuda()
        loss = model(input_data).sum()
        loss.backward()
        optimizer.step()

        # Verify completion
        self.assertTrue(True)

        dist.destroy_process_group()

### 4. Distributed Integration Tests (Process Groups)

Test with actual distributed setup. Use PyTorch's distributed testing utilities:

```python
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)

class TestMuonFSDP(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_fsdp_correctness(self):
        """Test FSDP produces correct results."""
        # Setup
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size
        )

        # Create model with FSDP
        model = FSDP(SimpleModel().cuda())
        fsdp_pg = model.process_group

        # Create distributed config
        config = create_auto_config(fsdp_process_group=fsdp_pg)
        optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

        # Run training step
        input_data = torch.randn(4, 10).cuda()
        loss = model(input_data).sum()
        loss.backward()
        optimizer.step()

        # Verify step completed
        self.assertTrue(True)

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_vs_single_device(self):
        """Compare FSDP results with single-device baseline."""
        # Setup distributed
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size
        )

        # Seed for reproducibility
        torch.manual_seed(42 + self.rank)

        # Create identical models
        model_fsdp = FSDP(SimpleModel().cuda())
        model_single = SimpleModel().cuda()

        # Sync initial weights
        if self.rank == 0:
            initial_state = model_single.state_dict()
        else:
            initial_state = None

        # Broadcast initial state
        # (Implementation details omitted for brevity)

        # Create optimizers
        config = create_auto_config(fsdp_process_group=model_fsdp.process_group)
        optimizer_fsdp = Muon(
            model_fsdp.parameters(),
            lr=0.01,
            distributed_config=config
        )
        optimizer_single = Muon(model_single.parameters(), lr=0.01)

        # Run multiple steps
        for step in range(10):
            # Same input on all ranks
            torch.manual_seed(42 + step)
            input_data = torch.randn(4, 10).cuda()

            # FSDP forward/backward/step
            loss_fsdp = model_fsdp(input_data).sum()
            loss_fsdp.backward()
            optimizer_fsdp.step()
            optimizer_fsdp.zero_grad()

            # Single-device forward/backward/step (rank 0 only)
            if self.rank == 0:
                loss_single = model_single(input_data).sum()
                loss_single.backward()
                optimizer_single.step()
                optimizer_single.zero_grad()

        # Compare final parameters
        if self.rank == 0:
            # Gather FSDP parameters
            with FSDP.summon_full_params(model_fsdp):
                fsdp_params = [p.clone().cpu() for p in model_fsdp.parameters()]

            single_params = [p.cpu() for p in model_single.parameters()]

            # Compare each parameter
            for i, (p_fsdp, p_single) in enumerate(zip(fsdp_params, single_params)):
                self.assertTrue(
                    torch.allclose(p_fsdp, p_single, atol=1e-4),
                    f"Parameter {i} mismatch between FSDP and single-device"
                )

        dist.destroy_process_group()
```

### 4. Multi-Dimensional Parallelism Tests

Test combinations of parallelism strategies:

```python
class TestMuon2DParallelism(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4  # 2x2 mesh

    @skip_if_lt_x_gpu(4)
    def test_fsdp_plus_tp(self):
        """Test FSDP + TP (2D parallelism)."""
        # Setup process groups for 2x2 mesh
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=4)

        # Create FSDP group (0,1) and (2,3)
        fsdp_ranks = [[0, 1], [2, 3]]
        fsdp_pg = dist.new_group(fsdp_ranks[self.rank // 2])

        # Create TP group (0,2) and (1,3)
        tp_ranks = [[0, 2], [1, 3]]
        tp_pg = dist.new_group(tp_ranks[self.rank % 2])

        # Create model with both parallelism strategies
        # (Implementation depends on how TP is applied)

        # TP dimensions per parameter
        tp_dim_per_param = {0: 0, 1: 1, 2: 0, 3: 1}

        # Create config
        config = create_auto_config(
            fsdp_process_group=fsdp_pg,
            tp_process_group=tp_pg,
            tp_dim_per_param=tp_dim_per_param,
        )

        optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

        # Run training step
        input_data = torch.randn(4, 10).cuda()
        loss = model(input_data).sum()
        loss.backward()
        optimizer.step()

        # Verify completion
        self.assertTrue(True)

        dist.destroy_process_group()
```

### 5. Expert Parallel Tests

Test MoE models with expert parallelism:

```python
class TestMuonExpertParallel(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4  # 4 experts

    @skip_if_lt_x_gpu(4)
    def test_expert_parallel_independence(self):
        """Verify experts are orthogonalized independently."""
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=4)

        # Create MoE model where each rank has one expert
        model = SimpleMoE(num_experts=4, expert_rank=self.rank).cuda()

        # Map parameters to experts
        expert_assignments = {
            idx: self.rank for idx in range(len(list(model.parameters())))
        }

        # Create config
        ep_pg = dist.group.WORLD
        config = create_auto_config(
            ep_process_group=ep_pg,
            expert_assignments=expert_assignments,
        )

        optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

        # Each rank processes different data (expert routing)
        torch.manual_seed(42 + self.rank)
        input_data = torch.randn(4, 10).cuda()
        loss = model(input_data).sum()
        loss.backward()
        optimizer.step()

        # Verify no cross-expert communication occurred
        # (Check that only local expert was updated)

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_expert_parallel_with_fsdp(self):
        """Test EP + FSDP combination (sharded experts)."""
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=4)

        # Setup: 2 experts, each FSDP-sharded across 2 GPUs
        # Ranks 0,1 handle expert 0; Ranks 2,3 handle expert 1

        # Create expert-specific FSDP groups
        fsdp_groups = {
            0: dist.new_group([0, 1]),  # Expert 0 FSDP group
            1: dist.new_group([2, 3]),  # Expert 1 FSDP group
        }
        expert_id = self.rank // 2
        fsdp_pg = fsdp_groups[expert_id]

        # Create EP group (not used for gather/redistribute, just for metadata)
        ep_pg = dist.group.WORLD

        # Create MoE model with FSDP-sharded experts
        model = SimpleMoE(num_experts=2, expert_rank=expert_id)
        model = FSDP(model, process_group=fsdp_pg)

        # Map parameters to experts
        expert_assignments = {
            idx: expert_id for idx in range(len(list(model.parameters())))
        }

        # Config combines EP + FSDP
        config = create_auto_config(
            ep_process_group=ep_pg,
            fsdp_process_group=fsdp_pg,
            expert_assignments=expert_assignments,
        )

        optimizer = Muon(model.parameters(), lr=0.01, distributed_config=config)

        # Run training step
        torch.manual_seed(42 + self.rank)
        input_data = torch.randn(4, 10).cuda()
        loss = model(input_data).sum()
        loss.backward()
        optimizer.step()

        # Verify: Each expert was processed independently, but FSDP still applied
        self.assertTrue(True)

        dist.destroy_process_group()
```

### 6. Performance Tests

Benchmark distributed training:

```python
class TestMuonPerformance(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4

    @skip_if_lt_x_gpu(4)
    def test_prefetch_speedup(self):
        """Measure speedup from prefetching."""
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=4)

        model = FSDP(LargeModel().cuda())
        fsdp_pg = model.process_group

        # Test without prefetch
        config_no_prefetch = create_auto_config(
            fsdp_process_group=fsdp_pg,
            prefetch_count=0,
        )
        optimizer_no_prefetch = Muon(
            model.parameters(), lr=0.01, distributed_config=config_no_prefetch
        )

        import time
        start = time.time()
        for _ in range(10):
            input_data = torch.randn(8, 128).cuda()
            loss = model(input_data).sum()
            loss.backward()
            optimizer_no_prefetch.step()
            optimizer_no_prefetch.zero_grad()
        time_no_prefetch = time.time() - start

        # Test with prefetch
        config_prefetch = create_auto_config(
            fsdp_process_group=fsdp_pg,
            prefetch_count=2,
        )
        optimizer_prefetch = Muon(
            model.parameters(), lr=0.01, distributed_config=config_prefetch
        )

        start = time.time()
        for _ in range(10):
            input_data = torch.randn(8, 128).cuda()
            loss = model(input_data).sum()
            loss.backward()
            optimizer_prefetch.step()
            optimizer_prefetch.zero_grad()
        time_prefetch = time.time() - start

        # Expect speedup with prefetch
        speedup = time_no_prefetch / time_prefetch
        if self.rank == 0:
            print(f"Prefetch speedup: {speedup:.2f}x")

        # Should see at least 1.2x speedup
        self.assertGreater(speedup, 1.2)

        dist.destroy_process_group()
```

### 7. Error Handling Tests

Test error conditions:

```python
class TestMuonErrors(TestCase):
    def test_incomplete_assignment_raises(self):
        """Verify error if not all params are assigned."""
        params = [torch.randn(10, 10) for _ in range(4)]

        def bad_assign_fn(params, state):
            # Only assign 2 of 4 params
            return {0: [0, 1], 1: []}

        state = {'rank': 0, 'world_size': 2}
        config = DistributedConfig(
            assign_fn=bad_assign_fn,
            gather_fn=lambda t, r, s: t,
            redistribute_fn=lambda t, r, s: t,
            state=state,
        )

        # Should raise during initialization
        with self.assertRaises(AssertionError):
            optimizer = Muon(params, lr=0.01, distributed_config=config)

    def test_overlapping_assignment_raises(self):
        """Verify error if params assigned to multiple ranks."""
        params = [torch.randn(10, 10) for _ in range(4)]

        def bad_assign_fn(params, state):
            # Assign param 0 to both ranks
            return {0: [0, 1], 1: [0, 2]}

        state = {'rank': 0, 'world_size': 2}
        config = DistributedConfig(
            assign_fn=bad_assign_fn,
            gather_fn=lambda t, r, s: t,
            redistribute_fn=lambda t, r, s: t,
            state=state,
        )

        # Should raise during initialization
        with self.assertRaises(AssertionError):
            optimizer = Muon(params, lr=0.01, distributed_config=config)

    def test_wrong_tp_dim_raises(self):
        """Verify error if TP dimension is out of bounds."""
        # Create 2D tensor
        param = torch.randn(10, 20)

        # TP dimension 3 is invalid for 2D tensor
        tp_dim_per_param = {0: 3}

        # Should raise during gather/redistribute
        with self.assertRaises((IndexError, RuntimeError)):
            config = create_auto_config(
                tp_process_group=None,
                tp_dim_per_param=tp_dim_per_param,
            )
            # Trigger gather
            config.gather_fn(param, 0, config.state)
```

## Test Helpers

Useful utilities for testing:

```python
def create_simple_model(hidden_size=64, num_layers=2):
    """Create simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        *[
            layer
            for _ in range(num_layers - 1)
            for layer in (torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU())
        ],
    )

def sync_model_params(model, rank, world_size):
    """Synchronize model parameters across all ranks."""
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def compare_params(params1, params2, atol=1e-5):
    """Compare two sets of parameters."""
    for p1, p2 in zip(params1, params2):
        if not torch.allclose(p1, p2, atol=atol):
            return False
    return True
```

## Running Tests

### Run all tests:
```bash
python -m pytest test_muon_distributed.py -v
```

### Run specific test class:
```bash
python -m pytest test_muon_distributed.py::TestMuonFSDP -v
```

### Run distributed tests with 4 GPUs:
```bash
torchrun --nproc_per_node=4 test_muon_distributed.py
```

### Run with coverage:
```bash
pytest test_muon_distributed.py --cov=torch.optim.muon --cov-report=html
```

## Continuous Integration

### CI Test Matrix:

```yaml
# .github/workflows/test_muon.yml
test_matrix:
  strategy:
    matrix:
      parallelism: [fsdp, ddp, tp, fsdp_tp, hsdp, ep]
      num_gpus: [2, 4, 8]
      prefetch: [0, 1, 2]
      async_parallel: [true, false]
```

### Test Stages:

1. **Stage 1: Unit tests** (CPU only, fast)
   - Core functionality
   - Configuration validation
   - Error handling

2. **Stage 2: Single-GPU tests**
   - Backward compatibility
   - Orthogonalization correctness

3. **Stage 3: Multi-GPU tests** (2-4 GPUs)
   - FSDP, DDP, TP individually
   - Basic correctness checks

4. **Stage 4: Large-scale tests** (8+ GPUs)
   - 2D and 3D parallelism
   - Performance benchmarks
   - Stress tests

## Debugging Tips

### Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check process group setup:
```python
def debug_process_groups(fsdp_pg, tp_pg):
    print(f"Rank {dist.get_rank()}")
    print(f"  FSDP group size: {dist.get_world_size(fsdp_pg)}")
    print(f"  FSDP rank: {dist.get_rank(fsdp_pg)}")
    print(f"  TP group size: {dist.get_world_size(tp_pg)}")
    print(f"  TP rank: {dist.get_rank(tp_pg)}")
```

### Verify assignments:
```python
def debug_assignments(assignments, num_params):
    all_assigned = set(chain.from_iterable(assignments.values()))
    print(f"Assignments: {assignments}")
    print(f"All params assigned: {all_assigned == set(range(num_params))}")
    print(f"Total assigned: {sum(len(v) for v in assignments.values())}")
```

### Test with small models first:
- Start with 2-layer MLPs
- Verify correctness before scaling up
- Use deterministic seeds for reproducibility

## Test Priority Order

Tests should be implemented in priority order to ensure the most critical functionality is validated first.

### P0: Must Have Before Merge

These tests are **required** for initial merge and prevent regressions:

- **Backward compatibility**: Verify `distributed_config=None` works correctly
- **DTensor config creation**: Verify `create_dtensor_config()` produces valid config
- **DTensor basic correctness**: Test DTensor gather/redistribute with simple Shard placement
- **FSDP correctness**: Compare FSDP vs single-device results
- **DDP correctness**: Verify DDP produces correct results with zero redundancy
- **Configuration validation**:
  - All parameters assigned exactly once
  - No overlapping assignments
  - `tp_dim_per_param` validation
  - DTensor metadata extraction and storage
- **Error handling**:
  - Incomplete assignments raise errors
  - Overlapping assignments raise errors
  - Invalid TP dimensions raise errors

**Exit criteria**: All P0 tests pass with 100% success rate.

### P1: Required Before Production

These tests should be completed before production deployment:

- **DTensor vs single-device**: Compare DTensor sharded training with single-device baseline
- **DTensor 2D mesh**: Test DTensor with 2D device mesh (FSDP-like + TP-like)
- **DTensor Replicate placement**: Verify Replicate placement works (DDP-like)
- **DTensor mixed parameters**: Test handling of mixed DTensor and regular parameters
- **TP correctness**: Verify tensor parallelism gather/redistribute operations
- **FSDP + TP (2D)**: Test 2D parallelism correctness
- **Per-parameter TP dimensions**: Verify different layers can use different TP dims
- **Prefetch performance**: Measure and validate prefetch speedup (>1.2x)
- **Async parallelism**: Verify async GPU operations work correctly
- **Basic stress tests**: 10+ training steps with deterministic results

**Exit criteria**: All P0+P1 tests pass. Performance meets targets (prefetch speedup >1.2x).

### P2: Nice to Have

These tests provide additional confidence but are not blockers:

- **DTensor with Partial placement**: Test Partial() placement strategy
- **DTensor with hybrid placements**: Test combinations of Shard/Replicate/Partial
- **DTensor N-D meshes**: Test 3D and higher dimensional device meshes
- **Expert Parallel (EP)**: EP-only and EP+FSDP combinations
- **Context Parallel (CP)**: CP correctness and combinations
- **3D+ parallelism**: FSDP+TP+DDP/EP combinations
- **Large-scale tests**: 8+ GPU configurations
- **Extended stress tests**: 100+ training steps, various model sizes
- **Performance benchmarks**: Detailed profiling and optimization validation
- **DTensor vs process group performance**: Compare DTensor vs equivalent process group setup

**Exit criteria**: Best effort. Address failures if they indicate serious bugs.

### Test Coverage Goals

With P0-P2 implemented:
- **Line coverage**: >90%
- **Branch coverage**: >80%
- **P0 strategies** (DTensor basic, FSDP, DDP, config validation): 100% coverage
- **P1 strategies** (DTensor 2D, TP, 2D parallelism): 100% coverage
- **P2 strategies** (DTensor N-D, EP, CP, 3D+): Best effort
- **Error paths**: 100% coverage

### DTensor-Specific Test Recommendations

When testing DTensor configurations:

1. **Test all placement strategies**:
   - `Shard(dim)`: Most common, test with different dimensions
   - `Replicate()`: DDP-like behavior, ensure zero redundancy maintained
   - `Partial()`: Reduce-scatter patterns
   - Hybrid combinations: `[Shard(0), Replicate()]`, etc.

2. **Test device mesh dimensions**:
   - 1D mesh: Simple sharding/replication
   - 2D mesh: FSDP + TP patterns
   - 3D+ mesh: Complex distributed strategies

3. **Validate metadata preservation**:
   - Verify `device_mesh` is correctly extracted and stored
   - Ensure `placements` are preserved through optimization step
   - Test with mixed DTensor and regular parameters

4. **Compare with equivalent process group setups**:
   - DTensor with `Shard(0)` should match FSDP behavior
   - DTensor with `Replicate()` should match DDP behavior
   - DTensor with 2D `[Shard(0), Shard(1)]` should match FSDP+TP

5. **Edge cases**:
   - Empty device meshes (single GPU)
   - Parameters that aren't DTensors (mixed models)
   - DTensors with incompatible device meshes (should error)
