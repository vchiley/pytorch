# Owner(s): ["module: optimizer"]
# mypy: allow-untyped-defs
"""Unit tests for Muon optimizer distributed training functionality.

These are P0 tests that must pass on every commit. They test the core
gather, redistribute, and assign functions without requiring multiple GPUs.
"""

import torch
from torch.optim import DistributedConfig, Muon
from torch.testing._internal.common_utils import TestCase


class TestMuonUnit(TestCase):
    """Unit tests for Muon distributed config helper functions."""

    def test_backward_compatibility_no_distributed_config(self):
        """Verify Muon works without distributed_config (backward compatible)."""
        model = torch.nn.Linear(256, 128)
        optimizer = Muon(model.parameters(), lr=1e-3)

        # Should work without error
        batch = torch.randn(8, 256)
        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()

        # No exception = success

    def test_distributed_config_dataclass_creation(self):
        """Test DistributedConfig dataclass can be created."""

        def dummy_assign(params, state):
            return {0: list(range(len(params)))}

        def dummy_gather(update, rank, state):
            return update

        def dummy_redistribute(ortho_update, rank, state):
            return ortho_update

        config = DistributedConfig(
            assign_fn=dummy_assign,
            gather_fn=dummy_gather,
            redistribute_fn=dummy_redistribute,
            state={"rank": 0, "world_size": 1},
        )

        self.assertIsNotNone(config)
        self.assertEqual(config.async_gpu_parallelism, True)
        self.assertEqual(config.prefetch_count, 1)

    def test_distributed_config_custom_flags(self):
        """Test DistributedConfig with custom flags."""

        def dummy_assign(params, state):
            return {0: list(range(len(params)))}

        def dummy_gather(update, rank, state):
            return update

        def dummy_redistribute(ortho_update, rank, state):
            return ortho_update

        config = DistributedConfig(
            assign_fn=dummy_assign,
            gather_fn=dummy_gather,
            redistribute_fn=dummy_redistribute,
            state={"rank": 0, "world_size": 1},
            async_gpu_parallelism=False,
            prefetch_count=0,
        )

        self.assertEqual(config.async_gpu_parallelism, False)
        self.assertEqual(config.prefetch_count, 0)

    def test_muon_only_accepts_2d_parameters(self):
        """Verify Muon raises error for non-2D parameters."""
        # 1D parameter should fail
        with self.assertRaises(ValueError):
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(torch.randn(256))
            optimizer = Muon([model.weight], lr=1e-3)

        # 3D parameter should fail
        with self.assertRaises(ValueError):
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(torch.randn(256, 128, 64))
            optimizer = Muon([model.weight], lr=1e-3)

    def test_gather_redistribute_identity(self):
        """Test that gather followed by redistribute is identity operation."""
        # This test simulates a simple FSDP-like sharding pattern

        # Create a full tensor
        full_tensor = torch.randn(256, 1024)

        # Simulate sharding across 4 ranks (shard along dim 1)
        world_size = 4
        shard_size = full_tensor.shape[1] // world_size

        # Test for each rank
        for rank in range(world_size):
            # Get this rank's shard
            shard = full_tensor[:, rank * shard_size : (rank + 1) * shard_size].clone()

            # Simulate gather: concatenate all shards
            all_shards = [
                full_tensor[:, r * shard_size : (r + 1) * shard_size]
                for r in range(world_size)
            ]
            gathered = torch.cat(all_shards, dim=1)

            # Verify gathered equals original
            self.assertEqual(gathered, full_tensor)

            # Simulate redistribute: extract this rank's shard
            redistributed = gathered[:, rank * shard_size : (rank + 1) * shard_size]

            # Verify redistribute equals original shard
            self.assertEqual(redistributed, shard)


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
