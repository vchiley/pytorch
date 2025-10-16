# Owner(s): ["module: optimizer"]
# mypy: allow-untyped-defs
"""FSDP integration tests for Muon optimizer.

These are P1 tests that verify Muon works correctly with FSDP.
Requires distributed setup to run.
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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


class TestMuonFSDP(MultiProcessTestCase):
    """FSDP integration tests for Muon optimizer."""

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fsdp_basic_training(self):
        """Test basic FSDP training with Muon."""
        # Initialize distributed
        rank = self.rank
        device = torch.device(f"cuda:{rank}")

        # Create model
        torch.manual_seed(42)
        model = TinyTransformer(dim=128, layers=2).to(device)
        model = FSDP(model)

        # Create distributed config
        config = create_auto_config(fsdp_process_group=dist.group.WORLD)

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
    def test_fsdp_with_nesterov(self):
        """Test FSDP with Nesterov momentum."""
        rank = self.rank
        device = torch.device(f"cuda:{rank}")

        torch.manual_seed(42)
        model = TinyTransformer(dim=128, layers=2).to(device)
        model = FSDP(model)

        config = create_auto_config(fsdp_process_group=dist.group.WORLD)
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

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fsdp_with_weight_decay(self):
        """Test FSDP with weight decay."""
        rank = self.rank
        device = torch.device(f"cuda:{rank}")

        torch.manual_seed(42)
        model = TinyTransformer(dim=128, layers=2).to(device)
        model = FSDP(model)

        config = create_auto_config(fsdp_process_group=dist.group.WORLD)
        optimizer = Muon(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.1,
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
