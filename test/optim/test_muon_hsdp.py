# Owner(s): ["module: optimizer"]
# mypy: allow-untyped-defs
"""HSDP integration tests for Muon optimizer.

These are P1 tests that verify Muon works correctly with HSDP
(Hybrid Sharded Data Parallel).
Requires distributed setup with multiple nodes to run.
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.optim import create_auto_config, Muon
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class TinyTransformer(torch.nn.Module):
    """Tiny transformer model for testing."""

    def __init__(self, dim=256, layers=4):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim) for _ in range(layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestMuonHSDP(MultiProcessTestCase):
    """HSDP integration tests for Muon optimizer."""

    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_hsdp_basic_training(self):
        """Test basic HSDP training with Muon."""
        rank = self.rank
        device = torch.device(f"cuda:{rank}")

        # Simulate 2 nodes with 2 GPUs each
        # Shard within node, replicate across nodes
        num_gpus_per_node = 2
        node_id = rank // num_gpus_per_node
        local_rank = rank % num_gpus_per_node

        # Create shard and replicate process groups
        shard_group_ranks = list(
            range(node_id * num_gpus_per_node, (node_id + 1) * num_gpus_per_node)
        )
        replicate_group_ranks = list(range(local_rank, self.world_size, num_gpus_per_node))

        shard_pg = dist.new_group(shard_group_ranks)
        replicate_pg = dist.new_group(replicate_group_ranks)

        # Create model with HSDP
        torch.manual_seed(42)
        model = TinyTransformer(dim=128, layers=2).to(device)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            process_group=(shard_pg, replicate_pg),
        )

        # Create distributed config for HSDP
        config = create_auto_config(
            shard_process_group=shard_pg,
            replicate_process_group=replicate_pg,
        )

        # Create optimizer
        optimizer = Muon(model.parameters(), lr=1e-3, distributed_config=config)

        # Training loop
        batch = torch.randn(4, 128).to(device)
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()

        # Verify parameters are finite
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p).all())

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_hsdp_with_nesterov(self):
        """Test HSDP with Nesterov momentum."""
        rank = self.rank
        device = torch.device(f"cuda:{rank}")

        num_gpus_per_node = 2
        node_id = rank // num_gpus_per_node
        local_rank = rank % num_gpus_per_node

        shard_group_ranks = list(
            range(node_id * num_gpus_per_node, (node_id + 1) * num_gpus_per_node)
        )
        replicate_group_ranks = list(range(local_rank, self.world_size, num_gpus_per_node))

        shard_pg = dist.new_group(shard_group_ranks)
        replicate_pg = dist.new_group(replicate_group_ranks)

        torch.manual_seed(42)
        model = TinyTransformer(dim=128, layers=2).to(device)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            process_group=(shard_pg, replicate_pg),
        )

        config = create_auto_config(
            shard_process_group=shard_pg,
            replicate_process_group=replicate_pg,
        )

        optimizer = Muon(
            model.parameters(),
            lr=1e-3,
            nesterov=True,
            distributed_config=config,
        )

        batch = torch.randn(4, 128).to(device)
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()

        for p in model.parameters():
            self.assertTrue(torch.isfinite(p).all())


if __name__ == "__main__":
    run_tests()
