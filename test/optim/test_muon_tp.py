# Owner(s): ["module: optimizer"]
# mypy: allow-untyped-defs
"""Tensor Parallel integration tests for Muon optimizer.

These are P1 tests that verify Muon works correctly with Tensor Parallelism.
Requires distributed setup to run.
"""

import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.optim import create_auto_config, Muon
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


class SimpleMLPTP(torch.nn.Module):
    """Simple MLP for tensor parallel testing."""

    def __init__(self, dim=256):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestMuonTP(MultiProcessTestCase):
    """Tensor Parallel integration tests for Muon optimizer."""

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_tp_colwise_basic(self):
        """Test TP with column-wise sharding."""
        rank = self.rank
        device = torch.device(f"cuda:{rank}")

        # Create device mesh for TP
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        # Create model
        torch.manual_seed(42)
        model = SimpleMLPTP(dim=128).to(device)

        # Apply TP with column-wise sharding
        tp_plan = {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
        }
        model = parallelize_module(model, device_mesh, tp_plan)

        # Create distributed config for TP (ColwiseParallel shards dim 1)
        config = create_auto_config(
            tp_process_group=dist.group.WORLD,
            tp_shard_dim=1,
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
    @skip_if_lt_x_gpu(2)
    def test_tp_with_nesterov(self):
        """Test TP with Nesterov momentum."""
        rank = self.rank
        device = torch.device(f"cuda:{rank}")

        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        torch.manual_seed(42)
        model = SimpleMLPTP(dim=128).to(device)

        tp_plan = {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
        }
        model = parallelize_module(model, device_mesh, tp_plan)

        config = create_auto_config(
            tp_process_group=dist.group.WORLD,
            tp_shard_dim=1,
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
