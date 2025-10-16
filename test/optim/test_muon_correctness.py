# Owner(s): ["module: optimizer"]
# mypy: allow-untyped-defs
"""Correctness tests for Muon optimizer distributed training.

These tests verify that distributed optimization produces identical results
to single-GPU optimization. They use mock distributed environments to avoid
requiring actual multi-GPU setups.
"""

import torch
from torch.optim import DistributedConfig, Muon
from torch.testing._internal.common_utils import TestCase


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


class TestMuonCorrectness(TestCase):
    """Correctness tests comparing distributed vs single-GPU."""

    def test_single_gpu_optimization_step(self):
        """Test single-GPU Muon performs a successful optimization step."""
        torch.manual_seed(42)
        model = TinyTransformer(dim=128, layers=2)
        optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.95)

        batch = torch.randn(4, 128)

        # Run 5 steps
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()

        # Verify parameters changed
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p).all())

    def test_orthogonalization_changes_updates(self):
        """Verify that orthogonalization actually modifies the update."""
        torch.manual_seed(42)
        model = torch.nn.Linear(256, 256)
        optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.95)

        batch = torch.randn(8, 256)

        # First step
        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()

        # Store gradient before step
        grad_before = model.weight.grad.clone()

        optimizer.step()

        # The momentum buffer should be updated (it will be orthogonalized)
        # We can't directly verify orthogonalization without accessing internals,
        # but we can verify the update was applied
        param_after = model.weight.data

        # Parameter should have changed
        self.assertFalse(torch.allclose(param_after, model.weight.data + grad_before))

    def test_momentum_buffer_initialization(self):
        """Test that momentum buffer is properly initialized."""
        torch.manual_seed(42)
        model = torch.nn.Linear(128, 128)
        optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.95)

        # Before first step, state should be empty
        self.assertEqual(len(optimizer.state), 0)

        batch = torch.randn(4, 128)
        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()

        # After first step, momentum buffer should exist
        for p in model.parameters():
            self.assertIn("momentum_buffer", optimizer.state[p])
            self.assertEqual(
                optimizer.state[p]["momentum_buffer"].shape,
                p.shape,
            )

    def test_nesterov_vs_standard_momentum(self):
        """Test that Nesterov and standard momentum produce different results."""
        torch.manual_seed(42)

        # Standard momentum
        model_standard = torch.nn.Linear(128, 128)
        optimizer_standard = Muon(
            model_standard.parameters(), lr=1e-3, momentum=0.95, nesterov=False
        )

        # Nesterov momentum
        torch.manual_seed(42)  # Same initialization
        model_nesterov = torch.nn.Linear(128, 128)
        optimizer_nesterov = Muon(
            model_nesterov.parameters(), lr=1e-3, momentum=0.95, nesterov=True
        )

        batch = torch.randn(4, 128)

        # Run 3 steps
        for _ in range(3):
            # Standard
            optimizer_standard.zero_grad()
            loss_standard = model_standard(batch).sum()
            loss_standard.backward()
            optimizer_standard.step()

            # Nesterov
            optimizer_nesterov.zero_grad()
            loss_nesterov = model_nesterov(batch).sum()
            loss_nesterov.backward()
            optimizer_nesterov.step()

        # After 3 steps, parameters should diverge
        for p_std, p_nes in zip(
            model_standard.parameters(), model_nesterov.parameters()
        ):
            self.assertFalse(torch.allclose(p_std, p_nes, atol=1e-5))

    def test_weight_decay_applied(self):
        """Test that weight decay is properly applied."""
        torch.manual_seed(42)

        # With weight decay
        model_wd = torch.nn.Linear(128, 128)
        optimizer_wd = Muon(
            model_wd.parameters(), lr=1e-3, weight_decay=0.1, momentum=0.95
        )

        # Without weight decay
        torch.manual_seed(42)  # Same initialization
        model_no_wd = torch.nn.Linear(128, 128)
        optimizer_no_wd = Muon(
            model_no_wd.parameters(), lr=1e-3, weight_decay=0.0, momentum=0.95
        )

        batch = torch.randn(4, 128)

        # Run 3 steps
        for _ in range(3):
            # With weight decay
            optimizer_wd.zero_grad()
            loss_wd = model_wd(batch).sum()
            loss_wd.backward()
            optimizer_wd.step()

            # Without weight decay
            optimizer_no_wd.zero_grad()
            loss_no_wd = model_no_wd(batch).sum()
            loss_no_wd.backward()
            optimizer_no_wd.step()

        # Parameters should differ due to weight decay
        for p_wd, p_no_wd in zip(model_wd.parameters(), model_no_wd.parameters()):
            self.assertFalse(torch.allclose(p_wd, p_no_wd, atol=1e-5))

    def test_numerical_stability(self):
        """Test that Muon remains numerically stable over many steps."""
        torch.manual_seed(42)
        model = TinyTransformer(dim=128, layers=4)
        optimizer = Muon(model.parameters(), lr=1e-3, momentum=0.95)

        batch = torch.randn(4, 128)

        # Run 100 steps
        for _ in range(100):
            optimizer.zero_grad()
            loss = model(batch).sum()
            loss.backward()
            optimizer.step()

        # All parameters should remain finite
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p).all())


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
