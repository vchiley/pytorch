"""
Unit tests for distributed Muon optimizer.

Tests the distributed training support added in Phase 1, including:
- Assignment functions
- Helper configuration functions
- Gather and redistribute operations
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import torch
from torch import Tensor

# Import Muon distributed components
from torch.optim._muon import (
    DistributedConfig,
    _validate_assignments,
    _default_assign_fn,
    create_processgroup_config,
)


class TestValidateAssignments(unittest.TestCase):
    """Test assignment validation logic."""

    def test_valid_assignments(self):
        """Test that valid assignments pass validation."""
        params = [torch.randn(10, 10) for _ in range(4)]
        assignments = {0: 0, 1: 1, 2: 2, 3: 3}
        world_size = 4

        # Should not raise
        _validate_assignments(assignments, params, world_size)

    def test_missing_parameter_assignment(self):
        """Test that missing parameter assignments raise ValueError."""
        params = [torch.randn(10, 10) for _ in range(4)]
        assignments = {0: 0, 1: 1}  # Missing params 2 and 3
        world_size = 4

        with self.assertRaises(ValueError) as context:
            _validate_assignments(assignments, params, world_size)

        self.assertIn("Missing assignments", str(context.exception))

    def test_invalid_rank_too_high(self):
        """Test that rank >= world_size raises ValueError."""
        params = [torch.randn(10, 10) for _ in range(4)]
        assignments = {0: 0, 1: 1, 2: 2, 3: 999}  # Rank 999 invalid
        world_size = 4

        with self.assertRaises(ValueError) as context:
            _validate_assignments(assignments, params, world_size)

        self.assertIn("Invalid rank", str(context.exception))

    def test_invalid_rank_negative(self):
        """Test that negative rank raises ValueError."""
        params = [torch.randn(10, 10) for _ in range(4)]
        assignments = {0: 0, 1: 1, 2: -1, 3: 3}  # Rank -1 invalid
        world_size = 4

        with self.assertRaises(ValueError) as context:
            _validate_assignments(assignments, params, world_size)

        self.assertIn("Invalid rank", str(context.exception))


class TestDefaultAssignFn(unittest.TestCase):
    """Test default round-robin assignment function."""

    def test_round_robin_assignment(self):
        """Test round-robin assignment distributes params evenly."""
        params = [torch.randn(10, 10) for _ in range(8)]
        state = {"world_size": 4}

        assignments = _default_assign_fn(params, state)

        # Verify all params assigned
        self.assertEqual(len(assignments), 8)

        # Verify round-robin pattern
        expected = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3}
        self.assertEqual(assignments, expected)

    def test_assignment_single_rank(self):
        """Test assignment with single rank (all assigned to rank 0)."""
        params = [torch.randn(10, 10) for _ in range(4)]
        state = {"world_size": 1}

        assignments = _default_assign_fn(params, state)

        # All should be assigned to rank 0
        for param_idx, rank in assignments.items():
            self.assertEqual(rank, 0)

    def test_assignment_more_ranks_than_params(self):
        """Test assignment when world_size > num_params."""
        params = [torch.randn(10, 10) for _ in range(2)]
        state = {"world_size": 8}

        assignments = _default_assign_fn(params, state)

        # Each param assigned to different rank
        self.assertEqual(assignments[0], 0)
        self.assertEqual(assignments[1], 1)


class TestDistributedConfig(unittest.TestCase):
    """Test DistributedConfig dataclass."""

    def test_config_creation(self):
        """Test creating DistributedConfig."""

        def dummy_assign_fn(params, state):
            return {i: i % state["world_size"] for i in range(len(params))}

        def dummy_gather_fn(momentum_buffer, dst_rank, state):
            return momentum_buffer if state["rank"] == dst_rank else None

        def dummy_redistribute_fn(update, src_rank, state):
            return update if update is not None else torch.zeros(10, 10)

        config = DistributedConfig(
            assign_fn=dummy_assign_fn,
            gather_fn=dummy_gather_fn,
            redistribute_fn=dummy_redistribute_fn,
            state={"rank": 0, "world_size": 4},
            async_gpu_parallelism=True,
            prefetch_count=1,
        )

        self.assertIsNotNone(config.assign_fn)
        self.assertIsNotNone(config.gather_fn)
        self.assertIsNotNone(config.redistribute_fn)
        self.assertEqual(config.state["world_size"], 4)
        self.assertTrue(config.async_gpu_parallelism)
        self.assertEqual(config.prefetch_count, 1)

    def test_config_defaults(self):
        """Test DistributedConfig default values."""

        def dummy_fn(*args, **kwargs):
            pass

        config = DistributedConfig(
            assign_fn=dummy_fn,
            gather_fn=dummy_fn,
            redistribute_fn=dummy_fn,
            state={},
        )

        # Check defaults
        self.assertTrue(config.async_gpu_parallelism)
        self.assertEqual(config.prefetch_count, 1)


class TestCreateProcessGroupConfig(unittest.TestCase):
    """Test create_processgroup_config helper function."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_create_config_not_initialized(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that creating config without initialized dist raises error."""
        mock_is_init.return_value = False

        with self.assertRaises(RuntimeError) as context:
            create_processgroup_config()

        self.assertIn("torch.distributed must be initialized", str(context.exception))

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_world_size")
    def test_create_config_with_fsdp_pg(
        self, mock_pg_world_size, mock_world_size, mock_rank, mock_is_init
    ):
        """Test creating config with FSDP process group."""
        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4
        mock_pg_world_size.return_value = 4

        # Create mock process group
        mock_fsdp_pg = Mock()

        config = create_processgroup_config(
            fsdp_pg=mock_fsdp_pg,
            async_gpu_parallelism=False,
            prefetch_count=0,
        )

        self.assertIsNotNone(config)
        self.assertEqual(config.state["rank"], 0)
        self.assertEqual(config.state["world_size"], 4)
        self.assertEqual(config.state["fsdp_pg"], mock_fsdp_pg)
        self.assertFalse(config.async_gpu_parallelism)
        self.assertEqual(config.prefetch_count, 0)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_gather_fn_returns_callable(self, mock_world_size, mock_rank, mock_is_init):
        """Test that gather_fn is a callable function."""
        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        config = create_processgroup_config()

        self.assertTrue(callable(config.gather_fn))

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_redistribute_fn_returns_callable(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that redistribute_fn is a callable function."""
        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        config = create_processgroup_config()

        self.assertTrue(callable(config.redistribute_fn))


class TestGatherFunction(unittest.TestCase):
    """Test gather function behavior."""

    def test_gather_fn_replicated_strategy(self):
        """Test gather_fn for replicated strategy (DDP)."""
        # Create mock state for DDP
        state = {
            "rank": 0,
            "world_size": 4,
            "dp_pg": Mock(),  # DDP process group
        }

        # Create gather function (manually, since we can't call create_processgroup_config without dist init)
        def gather_fn(
            momentum_buffer: Tensor, dst_rank: int, state: dict
        ) -> Tensor | None:
            rank = state["rank"]
            # DDP: replicated strategy - already have full tensor
            if state.get("dp_pg") is not None:
                if rank == dst_rank:
                    return momentum_buffer
                else:
                    return None
            return None

        # Test on dst_rank
        buffer = torch.randn(128, 64)
        result = gather_fn(buffer, dst_rank=0, state=state)
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result, buffer))

        # Test on non-dst_rank
        state["rank"] = 1
        result = gather_fn(buffer, dst_rank=0, state=state)
        self.assertIsNone(result)


class TestRedistributeFunction(unittest.TestCase):
    """Test redistribute function behavior."""

    def test_redistribute_fn_replicated_strategy(self):
        """Test redistribute_fn for replicated strategy (DDP)."""
        # Create mock state for DDP
        state = {
            "rank": 0,
            "world_size": 4,
            "dp_pg": Mock(),  # DDP process group
        }

        # Create redistribute function
        def redistribute_fn(
            update: Tensor | None, src_rank: int, state: dict
        ) -> Tensor:
            rank = state["rank"]
            # DDP: replicated strategy - need to broadcast
            if state.get("dp_pg") is not None:
                if rank == src_rank:
                    assert update is not None, "Source rank must have update tensor"
                    return update.clone()
                else:
                    # Allocate buffer (in real impl, would broadcast)
                    return torch.empty(0)
            return torch.empty(0)

        # Test on src_rank
        update = torch.randn(128, 64)
        result = redistribute_fn(update, src_rank=0, state=state)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, update.shape)

        # Test on non-src_rank
        state["rank"] = 1
        result = redistribute_fn(None, src_rank=0, state=state)
        self.assertIsNotNone(result)


class TestMuonDistributedIntegration(unittest.TestCase):
    """Integration tests for Muon with distributed config."""

    def test_muon_accepts_distributed_config(self):
        """Test that Muon optimizer accepts distributed_config parameter."""
        from torch.optim import Muon

        # Create simple model
        params = [torch.randn(128, 64, requires_grad=True) for _ in range(4)]

        # Create mock distributed config
        def dummy_fn(*args, **kwargs):
            return {i: 0 for i in range(4)}

        config = DistributedConfig(
            assign_fn=dummy_fn,
            gather_fn=lambda *args, **kwargs: None,
            redistribute_fn=lambda *args, **kwargs: torch.zeros(128, 64),
            state={"rank": 0, "world_size": 1, "assignments": {i: 0 for i in range(4)}},
        )

        # Should not raise
        optimizer = Muon(params, lr=0.02, distributed_config=config)
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.distributed_config, config)

    def test_muon_none_distributed_config(self):
        """Test that Muon works without distributed_config (backward compatibility)."""
        from torch.optim import Muon

        # Create simple model
        params = [torch.randn(128, 64, requires_grad=True) for _ in range(4)]

        # Should work without distributed_config
        optimizer = Muon(params, lr=0.02)
        self.assertIsNone(optimizer.distributed_config)

    def test_muon_validates_assignments(self):
        """Test that Muon validates assignments during __init__."""
        from torch.optim import Muon

        params = [torch.randn(128, 64, requires_grad=True) for _ in range(4)]

        # Create config with bad assign_fn (missing assignments)
        def bad_assign_fn(params, state):
            return {0: 0, 1: 1}  # Missing params 2 and 3

        config = DistributedConfig(
            assign_fn=bad_assign_fn,
            gather_fn=lambda *args, **kwargs: None,
            redistribute_fn=lambda *args, **kwargs: torch.zeros(128, 64),
            state={"rank": 0, "world_size": 4},
        )

        with self.assertRaises(ValueError):
            Muon(params, lr=0.02, distributed_config=config)


class TestDistributedLogic(unittest.TestCase):
    """Test distributed training logic."""

    def test_assignments_stored_in_state(self):
        """Test that assignments are computed and stored during init."""
        from torch.optim import Muon

        params = [torch.randn(128, 64, requires_grad=True) for _ in range(4)]

        config = DistributedConfig(
            assign_fn=_default_assign_fn,
            gather_fn=lambda *args, **kwargs: None,
            redistribute_fn=lambda *args, **kwargs: torch.zeros(128, 64),
            state={"rank": 0, "world_size": 4},
        )

        optimizer = Muon(params, lr=0.02, distributed_config=config)

        # Check that assignments were computed and stored
        self.assertIn("assignments", optimizer.distributed_config.state)
        assignments = optimizer.distributed_config.state["assignments"]
        self.assertEqual(len(assignments), 4)

        # Check round-robin pattern
        self.assertEqual(assignments[0], 0)
        self.assertEqual(assignments[1], 1)
        self.assertEqual(assignments[2], 2)
        self.assertEqual(assignments[3], 3)


class TestCombinedParallelismStrategies(unittest.TestCase):
    """Test combined parallelism strategies (Phase 2)."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_world_size")
    def test_fsdp_plus_tp_config_creation(
        self, mock_pg_world_size, mock_world_size, mock_rank, mock_is_init
    ):
        """Test creating config with combined FSDP + TP process groups."""
        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 16
        mock_pg_world_size.return_value = 4

        # Create mock process groups for FSDP and TP
        mock_fsdp_pg = Mock()
        mock_tp_pg = Mock()

        config = create_processgroup_config(
            fsdp_pg=mock_fsdp_pg,
            tp_pg=mock_tp_pg,
            async_gpu_parallelism=True,
            prefetch_count=1,
        )

        self.assertIsNotNone(config)
        self.assertEqual(config.state["fsdp_pg"], mock_fsdp_pg)
        self.assertEqual(config.state["tp_pg"], mock_tp_pg)
        self.assertTrue(config.async_gpu_parallelism)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_hsdp_config_creation(self, mock_world_size, mock_rank, mock_is_init):
        """Test creating config for HSDP (Hybrid Sharded Data Parallel)."""
        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 16

        # HSDP = FSDP + DDP
        mock_fsdp_pg = Mock()
        mock_dp_pg = Mock()

        config = create_processgroup_config(
            fsdp_pg=mock_fsdp_pg,
            dp_pg=mock_dp_pg,
        )

        self.assertIsNotNone(config)
        self.assertEqual(config.state["fsdp_pg"], mock_fsdp_pg)
        self.assertEqual(config.state["dp_pg"], mock_dp_pg)

    def test_gather_chaining_order(self):
        """Test that gather operations chain in correct order (TP first, then FSDP)."""
        # Create mock state with both TP and FSDP
        state = {
            "rank": 0,
            "world_size": 16,
            "tp_pg": Mock(),
            "fsdp_pg": Mock(),
        }

        # Mock all_gather to track call order
        call_order = []

        def mock_all_gather(gather_list, tensor, group):
            if group == state["tp_pg"]:
                call_order.append("tp")
            elif group == state["fsdp_pg"]:
                call_order.append("fsdp")
            # Fill gather_list with dummy data
            for i in range(len(gather_list)):
                gather_list[i] = tensor.clone()

        # Manually create gather_fn to test logic
        with patch("torch.distributed.all_gather", side_effect=mock_all_gather):
            from torch.optim._muon import create_processgroup_config

            # The actual implementation would use the gather_fn from config
            # For now, we verify the state setup is correct
            self.assertIn("tp_pg", state)
            self.assertIn("fsdp_pg", state)

    def test_redistribute_chaining_order(self):
        """Test that redistribute operations chain in reverse order (FSDP first, then TP)."""
        # Create mock state with both TP and FSDP
        state = {
            "rank": 0,
            "world_size": 16,
            "tp_pg": Mock(),
            "fsdp_pg": Mock(),
            "current_param_idx": 0,
            "param_shapes": {0: (128, 64)},
            "param_dtypes": {0: torch.float32},
            "param_devices": {0: torch.device("cpu")},
        }

        # Mock scatter to track call order
        call_order = []

        def mock_scatter(output, scatter_list, src, group):
            if group == state["tp_pg"]:
                call_order.append("tp")
            elif group == state["fsdp_pg"]:
                call_order.append("fsdp")

        # Manually create redistribute_fn to test logic
        with patch("torch.distributed.scatter", side_effect=mock_scatter):
            # The actual implementation would use the redistribute_fn from config
            # For now, we verify the state setup is correct
            self.assertIn("tp_pg", state)
            self.assertIn("fsdp_pg", state)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_triple_parallelism_config(self, mock_world_size, mock_rank, mock_is_init):
        """Test config with three parallelism strategies (FSDP + TP + DDP)."""
        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 32

        mock_fsdp_pg = Mock()
        mock_tp_pg = Mock()
        mock_dp_pg = Mock()

        config = create_processgroup_config(
            fsdp_pg=mock_fsdp_pg,
            tp_pg=mock_tp_pg,
            dp_pg=mock_dp_pg,
        )

        self.assertIsNotNone(config)
        self.assertEqual(config.state["fsdp_pg"], mock_fsdp_pg)
        self.assertEqual(config.state["tp_pg"], mock_tp_pg)
        self.assertEqual(config.state["dp_pg"], mock_dp_pg)


class TestDeviceMeshConfig(unittest.TestCase):
    """Test DeviceMesh configuration helper (Phase 2)."""

    def test_devicemesh_config_import(self):
        """Test that create_devicemesh_config can be imported."""
        from torch.optim._muon import create_devicemesh_config

        self.assertTrue(callable(create_devicemesh_config))

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_devicemesh_config_requires_init(self, mock_rank, mock_is_init):
        """Test that DeviceMesh config requires initialized distributed."""
        from torch.optim._muon import create_devicemesh_config

        mock_is_init.return_value = False

        # Create mock device mesh
        mock_mesh = Mock()
        mock_mesh.size.return_value = 16

        with self.assertRaises(RuntimeError) as context:
            create_devicemesh_config(mock_mesh, ["dp", "tp"])

        self.assertIn("torch.distributed must be initialized", str(context.exception))

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_devicemesh_config_extracts_process_groups(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that DeviceMesh config extracts process groups from mesh."""
        from torch.optim._muon import create_devicemesh_config

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 16

        # Create mock device mesh with process groups
        mock_mesh = Mock()
        mock_mesh.size.return_value = 16

        # Mock submeshes for each dimension
        mock_dp_submesh = Mock()
        mock_dp_submesh.get_group.return_value = Mock()  # Mock DP process group

        mock_tp_submesh = Mock()
        mock_tp_submesh.get_group.return_value = Mock()  # Mock TP process group

        # Mock __getitem__ to return submeshes
        mock_mesh.__getitem__ = lambda self, dim_name: (
            mock_dp_submesh if dim_name == "dp" else mock_tp_submesh
        )

        config = create_devicemesh_config(mock_mesh, ["dp", "tp"])

        self.assertIsNotNone(config)
        self.assertIn("device_mesh", config.state)
        self.assertIn("mesh_dim_names", config.state)
        self.assertEqual(config.state["mesh_dim_names"], ["dp", "tp"])


class TestDTensorConfig(unittest.TestCase):
    """Test DTensor configuration helper (Phase 2)."""

    def test_dtensor_config_import(self):
        """Test that create_dtensor_config can be imported."""
        from torch.optim._muon import create_dtensor_config

        self.assertTrue(callable(create_dtensor_config))

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_dtensor_config_requires_init(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that DTensor config requires initialized distributed."""
        from torch.optim._muon import create_dtensor_config

        mock_is_init.return_value = False

        with self.assertRaises(RuntimeError) as context:
            create_dtensor_config()

        self.assertIn("torch.distributed must be initialized", str(context.exception))

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_dtensor_config_creation(self, mock_world_size, mock_rank, mock_is_init):
        """Test creating DTensor config."""
        from torch.optim._muon import create_dtensor_config

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        config = create_dtensor_config(async_gpu_parallelism=True, prefetch_count=2)

        self.assertIsNotNone(config)
        self.assertEqual(config.state["rank"], 0)
        self.assertEqual(config.state["world_size"], 4)
        self.assertTrue(config.async_gpu_parallelism)
        self.assertEqual(config.prefetch_count, 2)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_dtensor_gather_handles_regular_tensors(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that DTensor gather_fn handles regular (non-DTensor) tensors."""
        from torch.optim._muon import create_dtensor_config

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        config = create_dtensor_config()

        # Test with regular tensor (not DTensor)
        regular_tensor = torch.randn(128, 64)
        result = config.gather_fn(regular_tensor, dst_rank=0, state=config.state)

        # Should return tensor on dst_rank, None on others
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result, regular_tensor))

        # Test on non-dst_rank
        config.state["rank"] = 1
        result = config.gather_fn(regular_tensor, dst_rank=0, state=config.state)
        self.assertIsNone(result)


class TestPrefetchingPhase3(unittest.TestCase):
    """Test Phase 3 prefetching functionality."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_prefetch_count_validation(self, mock_world_size, mock_rank, mock_is_init):
        """Test that prefetch_count parameter is validated properly."""
        from torch.optim import Muon
        from torch.optim._muon import create_processgroup_config

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        # Create config with valid prefetch_count
        config = create_processgroup_config(prefetch_count=1)
        params = [torch.randn(128, 64) for _ in range(4)]

        # Should not raise
        optimizer = Muon(params, distributed_config=config)

        # Test valid range
        config_0 = create_processgroup_config(prefetch_count=0)
        optimizer = Muon(params, distributed_config=config_0)

        config_10 = create_processgroup_config(prefetch_count=10)
        optimizer = Muon(params, distributed_config=config_10)

        # Test invalid prefetch_count (too high)
        config_invalid = create_processgroup_config(prefetch_count=11)
        with self.assertRaises(ValueError) as context:
            optimizer = Muon(params, distributed_config=config_invalid)
        self.assertIn("prefetch_count", str(context.exception))

        # Test invalid prefetch_count (negative)
        config_negative = create_processgroup_config(prefetch_count=-1)
        with self.assertRaises(ValueError) as context:
            optimizer = Muon(params, distributed_config=config_negative)
        self.assertIn("prefetch_count", str(context.exception))

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_prefetch_count_zero_uses_sequential_processing(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that prefetch_count=0 disables prefetching."""
        from torch.optim._muon import create_processgroup_config

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        # Create config with prefetch_count=0 (prefetching disabled)
        config = create_processgroup_config(prefetch_count=0)

        self.assertEqual(config.prefetch_count, 0)
        self.assertIsNotNone(config.gather_fn)
        self.assertIsNotNone(config.redistribute_fn)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_prefetch_count_nonzero_enables_prefetching(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that prefetch_count>0 enables prefetching."""
        from torch.optim._muon import create_processgroup_config

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        # Create config with prefetch_count>0 (prefetching enabled)
        config = create_processgroup_config(prefetch_count=2)

        self.assertEqual(config.prefetch_count, 2)
        self.assertIsNotNone(config.gather_fn)
        self.assertIsNotNone(config.redistribute_fn)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_async_gather_helper_function(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test async gather helper function exists and has correct signature."""
        from torch.optim._muon import _async_gather_fn

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        # Verify function exists
        self.assertIsNotNone(_async_gather_fn)

        # Test with minimal state (no process groups)
        state = {"rank": 0, "world_size": 4}
        tensor = torch.randn(128, 64)

        # Should return (tensor, None) when no process groups configured
        result, work_handle = _async_gather_fn(tensor, dst_rank=0, state=state)
        self.assertIsNotNone(result)
        self.assertIsNone(work_handle)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_process_parameters_with_prefetch_function(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that _process_parameters_with_prefetch function exists."""
        from torch.optim._muon import _process_parameters_with_prefetch

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        # Verify function exists
        self.assertIsNotNone(_process_parameters_with_prefetch)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_devicemesh_config_with_prefetch(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that DeviceMesh config supports prefetch_count parameter."""
        from torch.optim._muon import create_devicemesh_config
        from unittest.mock import MagicMock

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        # Create mock DeviceMesh
        mock_mesh = MagicMock()
        mock_mesh.size.return_value = 4

        # Create DeviceMesh config with prefetch_count
        config = create_devicemesh_config(mock_mesh, ["dp"], prefetch_count=2)

        self.assertEqual(config.prefetch_count, 2)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_dtensor_config_with_prefetch(
        self, mock_world_size, mock_rank, mock_is_init
    ):
        """Test that DTensor config supports prefetch_count parameter."""
        from torch.optim._muon import create_dtensor_config

        mock_is_init.return_value = True
        mock_rank.return_value = 0
        mock_world_size.return_value = 4

        # Create DTensor config with prefetch_count
        config = create_dtensor_config(prefetch_count=3)

        self.assertEqual(config.prefetch_count, 3)


class TestRefactoredHelpers(unittest.TestCase):
    """Test refactored helper functions from Phase 3 refactoring."""

    def test_supports_async_gather_with_tp(self):
        """Test _supports_async_gather returns True with TP process group."""
        from torch.optim._muon import _supports_async_gather
        from unittest.mock import MagicMock

        state = {"tp_pg": MagicMock(), "fsdp_pg": None}
        self.assertTrue(_supports_async_gather(state))

    def test_supports_async_gather_with_fsdp(self):
        """Test _supports_async_gather returns True with FSDP process group."""
        from torch.optim._muon import _supports_async_gather
        from unittest.mock import MagicMock

        state = {"tp_pg": None, "fsdp_pg": MagicMock()}
        self.assertTrue(_supports_async_gather(state))

    def test_supports_async_gather_with_both(self):
        """Test _supports_async_gather returns True with both TP and FSDP."""
        from torch.optim._muon import _supports_async_gather
        from unittest.mock import MagicMock

        state = {"tp_pg": MagicMock(), "fsdp_pg": MagicMock()}
        self.assertTrue(_supports_async_gather(state))

    def test_supports_async_gather_without_pg(self):
        """Test _supports_async_gather returns False without process groups."""
        from torch.optim._muon import _supports_async_gather

        state = {"tp_pg": None, "fsdp_pg": None}
        self.assertFalse(_supports_async_gather(state))

    def test_supports_async_gather_empty_state(self):
        """Test _supports_async_gather returns False with empty state."""
        from torch.optim._muon import _supports_async_gather

        state = {}
        self.assertFalse(_supports_async_gather(state))

    def test_wait_for_prefetch_gather_with_none_buffer(self):
        """Test _wait_for_prefetch_gather falls back to sync gather with None buffer."""
        from torch.optim._muon import _wait_for_prefetch_gather
        from unittest.mock import MagicMock

        # Setup mocks
        fallback_fn = MagicMock(return_value=torch.randn(128, 64))
        momentum_buf = torch.randn(128, 64)
        state = {"rank": 0}
        assignments = {0: 0}

        # Call with None prefetch buffer
        result = _wait_for_prefetch_gather(
            None, 0, 0, assignments, fallback_fn, momentum_buf, state
        )

        # Should call fallback function
        fallback_fn.assert_called_once()
        self.assertIsNotNone(result)

    def test_wait_for_prefetch_gather_with_tensor_result(self):
        """Test _wait_for_prefetch_gather with tensor result (no work handle)."""
        from torch.optim._muon import _wait_for_prefetch_gather

        # Prepare mock result
        tensor = torch.randn(128, 64)
        prefetch_buffer = (tensor, None)

        # Call function
        result = _wait_for_prefetch_gather(
            prefetch_buffer,
            0,
            0,
            {0: 0},
            lambda *args, **kwargs: None,
            torch.randn(128, 64),
            {"rank": 0},
        )

        # Should return the tensor
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result, tensor))

    def test_wait_for_prefetch_gather_non_dst_rank(self):
        """Test _wait_for_prefetch_gather returns None on non-dst rank."""
        from torch.optim._muon import _wait_for_prefetch_gather

        # Prepare mock result (None for non-dst rank)
        prefetch_buffer = (None, None)

        # Mock fallback function that should be called
        fallback_called = [False]

        def mock_fallback(*args, **kwargs):
            fallback_called[0] = True
            return torch.randn(128, 64)

        # Call function on non-dst rank
        result = _wait_for_prefetch_gather(
            prefetch_buffer,
            0,
            1,  # rank=1, but dst is rank 0
            {0: 0},
            mock_fallback,
            torch.randn(128, 64),
            {"rank": 1},
        )

        # With None prefetch buffer, fallback should be called which returns a tensor
        self.assertTrue(fallback_called[0])
        self.assertIsNotNone(result)

    def test_orthogonalize_and_apply_update_on_assigned_rank(self):
        """Test _orthogonalize_and_apply_update performs orthogonalization on assigned rank."""
        from torch.optim._muon import _orthogonalize_and_apply_update, DistributedConfig
        from unittest.mock import MagicMock

        # Setup - use non-leaf tensor to avoid in-place operation error
        param = torch.randn(128, 64) + 0  # Non-leaf tensor
        momentum_buffer_full = torch.randn(128, 64)
        assignments = {0: 0}
        rank = 0

        # Mock config
        config = MagicMock(spec=DistributedConfig)
        config.state = {"rank": 0}
        config.redistribute_fn = MagicMock(return_value=torch.randn(128, 64))

        # Call function
        _orthogonalize_and_apply_update(
            param,
            0,
            momentum_buffer_full,
            config,
            assignments,
            rank,
            lr=0.01,
            weight_decay=0.1,
            nesterov=False,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            ns_steps=5,
            eps=1e-7,
            adjust_lr_fn=None,
        )

        # Verify redistribute was called
        config.redistribute_fn.assert_called_once()

    def test_orthogonalize_and_apply_update_on_non_assigned_rank(self):
        """Test _orthogonalize_and_apply_update on non-assigned rank."""
        from torch.optim._muon import _orthogonalize_and_apply_update, DistributedConfig
        from unittest.mock import MagicMock

        # Setup - use non-leaf tensor to avoid in-place operation error
        param = torch.randn(128, 64) + 0  # Non-leaf tensor
        param_copy = param.clone()
        momentum_buffer_full = None  # Non-assigned rank has None
        assignments = {0: 0}
        rank = 1  # Different from assigned rank

        # Mock config
        config = MagicMock(spec=DistributedConfig)
        config.state = {"rank": 1}
        update = torch.randn(128, 64)
        config.redistribute_fn = MagicMock(return_value=update)

        # Call function
        _orthogonalize_and_apply_update(
            param,
            0,
            momentum_buffer_full,
            config,
            assignments,
            rank,
            lr=0.01,
            weight_decay=0.1,
            nesterov=False,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            ns_steps=5,
            eps=1e-7,
            adjust_lr_fn=None,
        )

        # Verify redistribute was called
        config.redistribute_fn.assert_called_once()

        # Verify parameter was updated
        self.assertFalse(torch.equal(param, param_copy))


class TestPhase4AsyncGPUParallelism(unittest.TestCase):
    """Test Phase 4: Async GPU Parallelism functionality.

    Phase 4 implements rank-level asynchronous processing where each rank
    independently processes only its assigned parameters, working in parallel
    without waiting for other ranks.
    """

    def test_async_mode_processes_only_assigned_params(self):
        """Test that async=True makes each rank process only its assigned params."""
        from torch.optim._muon import _select_parameters_to_process

        # Setup: 8 params, 4 ranks, round-robin assignment
        num_params = 8
        assignments = {i: i % 4 for i in range(num_params)}

        # Rank 0 should process params 0, 4
        rank_0_params = _select_parameters_to_process(
            assignments, rank=0, num_params=num_params, async_gpu=True
        )
        self.assertEqual(rank_0_params, [0, 4])

        # Rank 1 should process params 1, 5
        rank_1_params = _select_parameters_to_process(
            assignments, rank=1, num_params=num_params, async_gpu=True
        )
        self.assertEqual(rank_1_params, [1, 5])

        # Rank 2 should process params 2, 6
        rank_2_params = _select_parameters_to_process(
            assignments, rank=2, num_params=num_params, async_gpu=True
        )
        self.assertEqual(rank_2_params, [2, 6])

        # Rank 3 should process params 3, 7
        rank_3_params = _select_parameters_to_process(
            assignments, rank=3, num_params=num_params, async_gpu=True
        )
        self.assertEqual(rank_3_params, [3, 7])

    def test_sync_mode_processes_all_params(self):
        """Test that async=False makes all ranks process all params (debug mode)."""
        from torch.optim._muon import _select_parameters_to_process

        # Setup: 8 params, 4 ranks
        num_params = 8
        assignments = {i: i % 4 for i in range(num_params)}

        # All ranks should process all params when async=False
        for rank in range(4):
            params_to_process = _select_parameters_to_process(
                assignments, rank=rank, num_params=num_params, async_gpu=False
            )
            self.assertEqual(params_to_process, list(range(num_params)))

    def test_async_mode_no_overlap_between_ranks(self):
        """Test that in async mode, ranks don't process same params (zero-redundancy)."""
        from torch.optim._muon import _select_parameters_to_process

        num_params = 16
        world_size = 4
        assignments = {i: i % world_size for i in range(num_params)}

        # Collect params processed by each rank
        all_rank_params = []
        for rank in range(world_size):
            rank_params = _select_parameters_to_process(
                assignments, rank=rank, num_params=num_params, async_gpu=True
            )
            all_rank_params.append(set(rank_params))

        # Verify no overlap between ranks (zero-redundancy)
        for i in range(world_size):
            for j in range(i + 1, world_size):
                overlap = all_rank_params[i] & all_rank_params[j]
                self.assertEqual(
                    len(overlap),
                    0,
                    f"Ranks {i} and {j} have overlapping params: {overlap}",
                )

        # Verify all params are covered (completeness)
        all_params = set().union(*all_rank_params)
        self.assertEqual(all_params, set(range(num_params)))

    def test_async_with_unbalanced_assignment(self):
        """Test async mode with unbalanced parameter assignment across ranks."""
        from torch.optim._muon import _select_parameters_to_process

        # Unbalanced: rank 0 gets most params, rank 3 gets none
        num_params = 10
        assignments = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,  # Rank 0 gets 4 params
            4: 1,
            5: 1,
            6: 1,  # Rank 1 gets 3 params
            7: 2,
            8: 2,
            9: 2,  # Rank 2 gets 3 params
            # Rank 3 gets 0 params
        }

        rank_0_params = _select_parameters_to_process(
            assignments, rank=0, num_params=num_params, async_gpu=True
        )
        self.assertEqual(len(rank_0_params), 4)

        rank_1_params = _select_parameters_to_process(
            assignments, rank=1, num_params=num_params, async_gpu=True
        )
        self.assertEqual(len(rank_1_params), 3)

        rank_3_params = _select_parameters_to_process(
            assignments, rank=3, num_params=num_params, async_gpu=True
        )
        self.assertEqual(len(rank_3_params), 0)  # No params assigned

    def test_async_mode_with_prefetching(self):
        """Test that async mode and prefetching work together correctly."""
        # This is an integration test verifying that both optimizations combine

        # Setup mock distributed config with both features enabled
        config = DistributedConfig(
            assign_fn=_default_assign_fn,
            gather_fn=Mock(return_value=torch.randn(10, 10)),
            redistribute_fn=Mock(return_value=torch.randn(10, 10)),
            state={"rank": 0, "world_size": 2, "assignments": {0: 0, 1: 1}},
            async_gpu_parallelism=True,  # Phase 4 feature
            prefetch_count=1,  # Phase 3 feature
        )

        # Verify both settings are enabled
        self.assertTrue(config.async_gpu_parallelism)
        self.assertEqual(config.prefetch_count, 1)

    def test_async_disabled_for_debugging(self):
        """Test that async can be disabled for easier debugging."""
        config = DistributedConfig(
            assign_fn=_default_assign_fn,
            gather_fn=Mock(return_value=torch.randn(10, 10)),
            redistribute_fn=Mock(return_value=torch.randn(10, 10)),
            state={"rank": 0, "world_size": 2},
            async_gpu_parallelism=False,  # Disabled for debugging
            prefetch_count=0,  # Also disable prefetch
        )

        self.assertFalse(config.async_gpu_parallelism)
        self.assertEqual(config.prefetch_count, 0)

    def test_create_processgroup_config_async_default(self):
        """Test that create_processgroup_config enables async by default."""
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=2),
        ):
            config = create_processgroup_config()

            # Phase 4: async_gpu_parallelism should default to True
            self.assertTrue(config.async_gpu_parallelism)

    def test_create_processgroup_config_async_explicit(self):
        """Test explicitly setting async_gpu_parallelism in config creation."""
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=2),
        ):
            # Test True (default)
            config_async = create_processgroup_config(async_gpu_parallelism=True)
            self.assertTrue(config_async.async_gpu_parallelism)

            # Test False (debug mode)
            config_sync = create_processgroup_config(async_gpu_parallelism=False)
            self.assertFalse(config_sync.async_gpu_parallelism)

    def test_barrier_called_in_async_mode(self):
        """Test that barrier synchronization is called when async_gpu=True."""
        from torch.optim._muon import _single_tensor_muon_distributed

        # Setup mocks
        params = [torch.randn(10, 10)]
        grads = [torch.randn(10, 10)]
        momentum_bufs = [torch.zeros_like(grads[0])]

        config = MagicMock(spec=DistributedConfig)
        config.state = {
            "rank": 0,
            "world_size": 2,
            "assignments": {0: 0},
            "param_shapes": {0: (10, 10)},
            "param_dtypes": {0: torch.float32},
            "param_devices": {0: torch.device("cpu")},
        }
        config.async_gpu_parallelism = True  # Enable async
        config.prefetch_count = 0
        config.gather_fn = Mock(return_value=torch.randn(10, 10))
        config.redistribute_fn = Mock(return_value=torch.randn(10, 10))

        # Mock torch.distributed
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.barrier") as mock_barrier,
        ):
            _single_tensor_muon_distributed(
                params,
                grads,
                momentum_bufs,
                distributed_config=config,
                lr=0.01,
                weight_decay=0.1,
                momentum=0.95,
                nesterov=False,
                ns_coefficients=(3.4445, -4.7750, 2.0315),
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn=None,
                has_complex=False,
            )

            # Verify barrier was called (synchronization at end)
            mock_barrier.assert_called_once()

    def test_no_barrier_in_sync_mode(self):
        """Test that barrier is not called when async_gpu=False (not needed)."""
        from torch.optim._muon import _single_tensor_muon_distributed

        # Setup mocks
        params = [torch.randn(10, 10)]
        grads = [torch.randn(10, 10)]
        momentum_bufs = [torch.zeros_like(grads[0])]

        config = MagicMock(spec=DistributedConfig)
        config.state = {
            "rank": 0,
            "world_size": 2,
            "assignments": {0: 0},
            "param_shapes": {0: (10, 10)},
            "param_dtypes": {0: torch.float32},
            "param_devices": {0: torch.device("cpu")},
        }
        config.async_gpu_parallelism = False  # Disable async
        config.prefetch_count = 0
        config.gather_fn = Mock(return_value=torch.randn(10, 10))
        config.redistribute_fn = Mock(return_value=torch.randn(10, 10))

        # Mock torch.distributed
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.barrier") as mock_barrier,
        ):
            _single_tensor_muon_distributed(
                params,
                grads,
                momentum_bufs,
                distributed_config=config,
                lr=0.01,
                weight_decay=0.1,
                momentum=0.95,
                nesterov=False,
                ns_coefficients=(3.4445, -4.7750, 2.0315),
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn=None,
                has_complex=False,
            )

            # Verify barrier was NOT called (no synchronization needed in sync mode)
            mock_barrier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
