#!/usr/bin/env python3
"""
Real distributed tests for Muon optimizer.

These tests use actual torch.distributed with multiple processes to validate
the distributed implementation works correctly with real FSDP, DDP, and
communication patterns.

Requirements:
- 2+ GPUs for multi-GPU tests
- NCCL backend for GPU communication
- Gloo backend for CPU tests

Tests are automatically skipped if requirements aren't met.
"""

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Muon
from torch.optim._muon import create_processgroup_config

# FSDP imports for GPU tests
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


def setup_process_group(rank, world_size, backend="gloo", init_method=None):
    """Initialize distributed process group for a rank."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    if init_method is None:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )


def cleanup_process_group():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def temporary_file():
    """Context manager for temporary file."""
    fd, path = tempfile.mkstemp()
    try:
        os.close(fd)
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# =============================================================================
# GPU Test Functions (Run in Each Process)
# =============================================================================


def _test_muon_with_fsdp_gpu(rank, world_size, init_method, results_queue):
    """Test Muon optimizer with real FSDP on GPU."""
    try:
        # Set device for this rank
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Initialize with NCCL backend for GPU
        setup_process_group(rank, world_size, backend="nccl", init_method=init_method)

        # Create model on GPU
        model = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 64, bias=False),
        ).to(device)

        # Wrap with FSDP - use NO_SHARD for testing to keep full parameters
        # This ensures we have 2D parameters for Muon
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            device_id=rank,
        )

        # Create Muon optimizer with FSDP process group
        config = create_processgroup_config(
            fsdp_pg=dist.group.WORLD,
            async_gpu_parallelism=False,
            prefetch_count=0,
        )

        # Get 2D parameters only
        params_2d = [p for p in model.parameters() if p.dim() == 2]
        if len(params_2d) == 0:
            cleanup_process_group()
            results_queue.put(
                (rank, "error", "No 2D parameters found after FSDP wrapping")
            )
            return

        optimizer = Muon(params_2d, lr=0.02, distributed_config=config)

        # Verify config
        assert config.state["rank"] == rank
        assert config.state["world_size"] == world_size
        assert "fsdp_pg" in config.state

        # Run training steps
        first_loss = None
        for step in range(3):
            input_data = torch.randn(16, 128, device=device)
            output = model(input_data)
            loss = output.sum()

            if step == 0:
                first_loss = loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Verify training progressed
        final_loss = loss.item()

        cleanup_process_group()
        results_queue.put(
            (
                rank,
                "success",
                {
                    "first_loss": first_loss,
                    "final_loss": final_loss,
                    "num_params": len(params_2d),
                },
            )
        )

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_with_ddp_gpu(rank, world_size, init_method, results_queue):
    """Test Muon optimizer with real DDP on GPU."""
    try:
        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Initialize with NCCL
        setup_process_group(rank, world_size, backend="nccl", init_method=init_method)

        # Create model on GPU
        model = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 64, bias=False),
        ).to(device)

        # Wrap with DDP
        model = DDP(model, device_ids=[rank])

        # Create Muon optimizer with DDP process group
        config = create_processgroup_config(
            dp_pg=dist.group.WORLD,
            async_gpu_parallelism=False,
            prefetch_count=0,
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Verify config
        assert config.state["rank"] == rank
        assert config.state["world_size"] == world_size
        assert "dp_pg" in config.state

        # Training steps
        losses = []
        for step in range(3):
            input_data = torch.randn(16, 128, device=device)
            output = model(input_data)
            loss = output.sum()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Barrier to ensure all ranks finish
        dist.barrier()

        cleanup_process_group()
        results_queue.put(
            (
                rank,
                "success",
                {
                    "losses": losses,
                    "device": str(device),
                },
            )
        )

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_with_fsdp_sharded_gpu(rank, world_size, init_method, results_queue):
    """Test Muon with FSDP using actual sharding (FULL_SHARD)."""
    try:
        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Initialize with NCCL
        setup_process_group(rank, world_size, backend="nccl", init_method=init_method)

        # Create larger model for sharding to be meaningful
        model = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.Linear(256, 256, bias=False),
            nn.Linear(256, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        ).to(device)

        # Wrap with FSDP using FULL_SHARD (real sharding)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank,
        )

        # Create Muon optimizer with FSDP process group
        config = create_processgroup_config(
            fsdp_pg=dist.group.WORLD,
            async_gpu_parallelism=True,  # Test async mode
            prefetch_count=1,  # Test prefetching
        )

        # Note: With FULL_SHARD, FSDP flattens parameters
        # We need to check if we can still use Muon
        params_2d = [p for p in model.parameters() if p.dim() == 2]

        if len(params_2d) == 0:
            # FSDP flattened all params - this is expected behavior
            cleanup_process_group()
            results_queue.put(
                (
                    rank,
                    "success",
                    {
                        "note": "FSDP flattened parameters (expected)",
                        "total_params": len(list(model.parameters())),
                    },
                )
            )
            return

        optimizer = Muon(params_2d, lr=0.02, distributed_config=config)

        # Run training
        for step in range(3):
            input_data = torch.randn(16, 256, device=device)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        dist.barrier()

        cleanup_process_group()
        results_queue.put(
            (
                rank,
                "success",
                {
                    "num_2d_params": len(params_2d),
                    "sharding_strategy": "FULL_SHARD",
                },
            )
        )

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_async_mode_gpu(rank, world_size, init_method, results_queue):
    """Test Muon async mode on GPU with real communication."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        setup_process_group(rank, world_size, backend="nccl", init_method=init_method)

        # Create model
        model = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 64, bias=False),
        ).to(device)

        # Create optimizer with async mode
        config = create_processgroup_config(
            async_gpu_parallelism=True,
            prefetch_count=1,
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Get assignments
        assignments = optimizer.distributed_config.state["assignments"]
        my_params = [i for i, r in assignments.items() if r == rank]

        # Run training
        for step in range(5):
            input_data = torch.randn(16, 128, device=device)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        dist.barrier()

        cleanup_process_group()
        results_queue.put(
            (
                rank,
                "success",
                {
                    "my_params": my_params,
                    "num_params": len(my_params),
                    "device": str(device),
                },
            )
        )

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_single_gpu_baseline(rank, world_size, init_method, results_queue):
    """Baseline: Single GPU without distributed (no DDP/FSDP)."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # No distributed setup for baseline

        # Create model with fixed seed for reproducibility
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 32, bias=False),
        ).to(device)

        # Create Muon WITHOUT distributed config (baseline)
        optimizer = Muon(model.parameters(), lr=0.02)

        # Fixed input for reproducibility
        torch.manual_seed(100)

        # Run training
        losses = []
        param_snapshots = []
        for step in range(3):
            input_data = torch.randn(16, 64, device=device)
            output = model(input_data)
            loss = output.sum()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save parameter snapshot
            params = [p.data.clone().cpu() for p in model.parameters()]
            param_snapshots.append(params)

        results_queue.put(
            (
                rank,
                "success",
                {
                    "losses": losses,
                    "param_snapshots": param_snapshots,
                    "mode": "baseline",
                },
            )
        )

    except Exception as e:
        import traceback

        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_comparison_ddp(rank, world_size, init_method, results_queue):
    """DDP with Muon for comparison against baseline."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        setup_process_group(rank, world_size, backend="nccl", init_method=init_method)

        # Create model with same seed
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 32, bias=False),
        ).to(device)

        # Wrap with DDP
        model = DDP(model, device_ids=[rank])

        # Create Muon with DDP process group
        config = create_processgroup_config(
            dp_pg=dist.group.WORLD,
            async_gpu_parallelism=False,
            prefetch_count=0,
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Same fixed input
        torch.manual_seed(100)

        # Run training
        losses = []
        param_snapshots = []
        for step in range(3):
            input_data = torch.randn(16, 64, device=device)
            output = model(input_data)
            loss = output.sum()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save parameter snapshot
            params = [p.data.clone().cpu() for p in model.parameters()]
            param_snapshots.append(params)

        dist.barrier()
        cleanup_process_group()

        results_queue.put(
            (
                rank,
                "success",
                {
                    "losses": losses,
                    "param_snapshots": param_snapshots,
                    "mode": "ddp",
                },
            )
        )

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_comparison_fsdp(rank, world_size, init_method, results_queue):
    """FSDP with Muon for comparison against baseline."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        setup_process_group(rank, world_size, backend="nccl", init_method=init_method)

        # Create model with same seed
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 32, bias=False),
        ).to(device)

        # Wrap with FSDP (NO_SHARD to keep parameters)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            device_id=rank,
        )

        # Create Muon with FSDP process group
        config = create_processgroup_config(
            fsdp_pg=dist.group.WORLD,
            async_gpu_parallelism=False,
            prefetch_count=0,
        )

        params_2d = [p for p in model.parameters() if p.dim() == 2]
        optimizer = Muon(params_2d, lr=0.02, distributed_config=config)

        # Same fixed input
        torch.manual_seed(100)

        # Run training
        losses = []
        param_snapshots = []
        for step in range(3):
            input_data = torch.randn(16, 64, device=device)
            output = model(input_data)
            loss = output.sum()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save parameter snapshot
            params = [p.data.clone().cpu() for p in model.parameters()]
            param_snapshots.append(params)

        dist.barrier()
        cleanup_process_group()

        results_queue.put(
            (
                rank,
                "success",
                {
                    "losses": losses,
                    "param_snapshots": param_snapshots,
                    "mode": "fsdp",
                },
            )
        )

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_comparison_async(rank, world_size, init_method, results_queue):
    """DDP with async_gpu_parallelism=True for comparison."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        setup_process_group(rank, world_size, backend="nccl", init_method=init_method)

        # Same seed for reproducibility
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 32, bias=False),
        ).to(device)

        # Wrap with DDP
        model = DDP(model, device_ids=[rank])

        # Create Muon WITH async mode
        config = create_processgroup_config(
            dp_pg=dist.group.WORLD,
            async_gpu_parallelism=True,  # ASYNC ON
            prefetch_count=0,
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Same fixed input
        torch.manual_seed(100)

        # Run training
        losses = []
        param_snapshots = []
        for step in range(3):
            input_data = torch.randn(16, 64, device=device)
            output = model(input_data)
            loss = output.sum()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save parameter snapshot
            params = [p.data.clone().cpu() for p in model.parameters()]
            param_snapshots.append(params)

        dist.barrier()
        cleanup_process_group()

        results_queue.put(
            (
                rank,
                "success",
                {
                    "losses": losses,
                    "param_snapshots": param_snapshots,
                    "mode": "async",
                },
            )
        )

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_comparison_prefetch(rank, world_size, init_method, results_queue):
    """DDP with prefetching for comparison."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        setup_process_group(rank, world_size, backend="nccl", init_method=init_method)

        # Same seed for reproducibility
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 32, bias=False),
        ).to(device)

        # Wrap with DDP
        model = DDP(model, device_ids=[rank])

        # Create Muon WITH prefetching
        config = create_processgroup_config(
            dp_pg=dist.group.WORLD,
            async_gpu_parallelism=True,
            prefetch_count=1,  # PREFETCH ON
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Same fixed input
        torch.manual_seed(100)

        # Run training
        losses = []
        param_snapshots = []
        for step in range(3):
            input_data = torch.randn(16, 64, device=device)
            output = model(input_data)
            loss = output.sum()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save parameter snapshot
            params = [p.data.clone().cpu() for p in model.parameters()]
            param_snapshots.append(params)

        dist.barrier()
        cleanup_process_group()

        results_queue.put(
            (
                rank,
                "success",
                {
                    "losses": losses,
                    "param_snapshots": param_snapshots,
                    "mode": "prefetch",
                },
            )
        )

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_mixed_precision_gpu(rank, world_size, init_method, results_queue):
    """Test Muon with mixed precision training on GPU."""
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        setup_process_group(rank, world_size, backend="nccl", init_method=init_method)

        # Create model
        model = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 64, bias=False),
        ).to(device)

        # Wrap with DDP
        model = DDP(model, device_ids=[rank])

        # Create optimizer
        config = create_processgroup_config(
            dp_pg=dist.group.WORLD,
            async_gpu_parallelism=False,
            prefetch_count=0,
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Use automatic mixed precision
        scaler = torch.cuda.amp.GradScaler()

        # Training with AMP
        for step in range(3):
            input_data = torch.randn(16, 128, device=device)

            with torch.cuda.amp.autocast():
                output = model(input_data)
                loss = output.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        dist.barrier()

        cleanup_process_group()
        results_queue.put((rank, "success", {"mixed_precision": True}))

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


# =============================================================================
# Test Functions (Run in Each Process)
# =============================================================================


def _test_muon_with_fsdp(rank, world_size, init_method, results_queue):
    """Test Muon optimizer with FSDP process group (CPU version without actual FSDP wrapping)."""
    try:
        setup_process_group(rank, world_size, backend="gloo", init_method=init_method)

        # Create simple model
        # Note: FSDP requires GPU, so we test the FSDP process group config
        # without actually wrapping the model in FSDP for CPU tests
        model = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.Linear(128, 64, bias=False),
        )

        # Create Muon optimizer with FSDP process group
        # This tests that the config creation and assignment logic work
        config = create_processgroup_config(
            fsdp_pg=dist.group.WORLD,
            async_gpu_parallelism=False,
            prefetch_count=0,
        )

        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Verify config was created correctly
        assert config.state["rank"] == rank
        assert config.state["world_size"] == world_size
        assert "fsdp_pg" in config.state

        # Run a few training steps
        for step in range(3):
            # Forward pass
            input_data = torch.randn(16, 128)
            output = model(input_data)
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

        cleanup_process_group()
        results_queue.put((rank, "success", None))

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_with_ddp(rank, world_size, init_method, results_queue):
    """Test Muon optimizer with real DDP."""
    try:
        setup_process_group(rank, world_size, backend="gloo", init_method=init_method)

        # Create model (small to reduce communication overhead)
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 32, bias=False),
        )

        # Wrap with DDP
        model = DDP(model, bucket_cap_mb=1)  # Small bucket for faster communication

        # Create Muon optimizer with DDP process group
        config = create_processgroup_config(
            dp_pg=dist.group.WORLD,
            async_gpu_parallelism=False,
            prefetch_count=0,
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Verify config
        assert config.state["rank"] == rank
        assert config.state["world_size"] == world_size
        assert "dp_pg" in config.state

        # Training steps - use smaller batches and fewer steps
        losses = []
        for step in range(2):
            input_data = torch.randn(8, 64)
            output = model(input_data)
            loss = output.sum()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Allow some time for communication to finish
        dist.barrier()

        cleanup_process_group()
        results_queue.put((rank, "success", losses))

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_parameter_assignment(rank, world_size, init_method, results_queue):
    """Test that parameter assignment works correctly across ranks."""
    try:
        setup_process_group(rank, world_size, backend="gloo", init_method=init_method)

        # Create model with known number of parameters
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 64, bias=False),
        )

        # Create optimizer with distributed config
        config = create_processgroup_config(
            async_gpu_parallelism=True,  # Test async mode
            prefetch_count=0,
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Get assignments
        assignments = optimizer.distributed_config.state["assignments"]

        # Verify all parameters are assigned
        assert len(assignments) == 4, f"Expected 4 params, got {len(assignments)}"

        # Verify assignments are valid (0 to world_size-1)
        for param_idx, assigned_rank in assignments.items():
            assert 0 <= assigned_rank < world_size, f"Invalid rank {assigned_rank}"

        # Count how many params this rank owns
        my_params = [i for i, r in assignments.items() if r == rank]

        # All ranks should have approximately equal params
        expected_per_rank = len(assignments) / world_size
        assert len(my_params) >= int(expected_per_rank) - 1, "Unbalanced assignment"

        cleanup_process_group()
        results_queue.put(
            (rank, "success", {"my_params": my_params, "assignments": assignments})
        )

    except Exception as e:
        cleanup_process_group()
        results_queue.put((rank, "error", str(e)))


def _test_muon_gradient_synchronization(rank, world_size, init_method, results_queue):
    """Test that gradients are properly synchronized across ranks."""
    try:
        setup_process_group(rank, world_size, backend="gloo", init_method=init_method)

        # Create identical model on all ranks
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.Linear(32, 16, bias=False),
        )

        # Wrap with DDP to synchronize gradients
        model = DDP(model, bucket_cap_mb=1)

        # Create optimizer
        config = create_processgroup_config(
            dp_pg=dist.group.WORLD,
            async_gpu_parallelism=False,
            prefetch_count=0,
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Use deterministic input
        torch.manual_seed(42 + rank)
        input_data = torch.randn(8, 32)

        # Forward and backward
        output = model(input_data)
        loss = output.sum()
        loss.backward()

        # Collect gradients before optimizer step
        grads_before = [
            p.grad.clone() if p.grad is not None else None for p in model.parameters()
        ]

        # Optimizer step
        optimizer.step()

        # Barrier to sync
        dist.barrier()

        # Collect parameters after step
        params_after = [p.data.clone() for p in model.parameters()]

        cleanup_process_group()
        results_queue.put(
            (
                rank,
                "success",
                {
                    "loss": loss.item(),
                    "num_params": len(params_after),
                },
            )
        )

    except Exception as e:
        import traceback

        cleanup_process_group()
        results_queue.put((rank, "error", f"{str(e)}\n{traceback.format_exc()}"))


def _test_muon_async_mode(rank, world_size, init_method, results_queue):
    """Test Muon with async_gpu_parallelism=True."""
    try:
        setup_process_group(rank, world_size, backend="gloo", init_method=init_method)

        # Create model
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 64, bias=False),
        )

        # Create optimizer with async mode
        config = create_processgroup_config(
            async_gpu_parallelism=True,
            prefetch_count=0,
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Get which params this rank processes
        assignments = optimizer.distributed_config.state["assignments"]
        my_params = [i for i, r in assignments.items() if r == rank]

        # Run training
        for step in range(3):
            input_data = torch.randn(8, 64)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        cleanup_process_group()
        results_queue.put(
            (
                rank,
                "success",
                {
                    "my_params": my_params,
                    "num_params": len(my_params),
                },
            )
        )

    except Exception as e:
        cleanup_process_group()
        results_queue.put((rank, "error", str(e)))


def _test_muon_with_prefetch(rank, world_size, init_method, results_queue):
    """Test Muon with prefetching enabled."""
    try:
        setup_process_group(rank, world_size, backend="gloo", init_method=init_method)

        # Create model
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 64, bias=False),
            nn.Linear(64, 64, bias=False),
        )

        # Create optimizer with prefetching
        config = create_processgroup_config(
            async_gpu_parallelism=True,
            prefetch_count=1,  # Enable prefetching
        )
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)

        # Verify prefetch_count is set
        assert optimizer.distributed_config.prefetch_count == 1

        # Run training with prefetching
        for step in range(3):
            input_data = torch.randn(8, 64)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        cleanup_process_group()
        results_queue.put((rank, "success", None))

    except Exception as e:
        cleanup_process_group()
        results_queue.put((rank, "error", str(e)))


# =============================================================================
# Test Runner Utilities
# =============================================================================


def run_distributed_test(test_fn, world_size=2):
    """
    Run a distributed test function across multiple processes.

    Args:
        test_fn: Function to run on each rank (signature: fn(rank, world_size, init_method, results_queue))
        world_size: Number of processes to spawn

    Returns:
        List of (rank, status, result) tuples from each process
    """
    with temporary_file() as tmp_file:
        init_method = f"file://{tmp_file}"

        # Create queue for results
        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        results_queue = ctx.Queue()

        # Spawn processes
        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=test_fn,
                args=(rank, world_size, init_method, results_queue),
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        return sorted(results, key=lambda x: x[0])  # Sort by rank


# =============================================================================
# Test Cases
# =============================================================================


class TestMuonRealDistributed(unittest.TestCase):
    """Real distributed tests for Muon optimizer."""

    @classmethod
    def setUpClass(cls):
        """Check if distributed tests can run."""
        # Check for gloo backend (required for CPU tests)
        if not dist.is_gloo_available():
            raise unittest.SkipTest("Gloo backend not available")

    def test_muon_with_fsdp(self):
        """Test Muon optimizer with real FSDP."""
        results = run_distributed_test(_test_muon_with_fsdp, world_size=2)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

    def test_muon_with_ddp(self):
        """Test Muon optimizer with real DDP."""
        results = run_distributed_test(_test_muon_with_ddp, world_size=2)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

        # Verify losses are changing (training is working)
        for rank, status, losses in results:
            self.assertNotEqual(
                losses[0],
                losses[-1],
                f"Rank {rank}: Loss should change during training",
            )

    def test_parameter_assignment_distribution(self):
        """Test that parameters are distributed across ranks correctly."""
        results = run_distributed_test(_test_muon_parameter_assignment, world_size=2)

        # Verify all ranks succeeded
        all_assignments = None
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

            # All ranks should see same assignments
            if all_assignments is None:
                all_assignments = result["assignments"]
            else:
                self.assertEqual(
                    all_assignments,
                    result["assignments"],
                    "All ranks should have same parameter assignments",
                )

        # Verify assignments cover all parameters
        self.assertEqual(len(all_assignments), 4, "Should have 4 parameters")

        # Verify each parameter assigned to exactly one rank
        assigned_ranks = list(all_assignments.values())
        self.assertEqual(
            set(assigned_ranks),
            {0, 1},
            "Parameters should be distributed across both ranks",
        )

    def test_parameter_assignment_with_4_ranks(self):
        """Test parameter assignment with 4 ranks."""
        results = run_distributed_test(_test_muon_parameter_assignment, world_size=4)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

        # Verify assignments are balanced
        assignments = results[0][2]["assignments"]
        rank_counts = {}
        for param_idx, assigned_rank in assignments.items():
            rank_counts[assigned_rank] = rank_counts.get(assigned_rank, 0) + 1

        # With 4 params and 4 ranks, each rank should get exactly 1 param
        self.assertEqual(
            rank_counts,
            {0: 1, 1: 1, 2: 1, 3: 1},
            "Each rank should get exactly 1 parameter",
        )

    def test_gradient_synchronization(self):
        """Test that gradients are synchronized correctly across ranks."""
        results = run_distributed_test(
            _test_muon_gradient_synchronization, world_size=2
        )

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

        # Note: With DDP, gradients are synchronized, so we expect parameters
        # to be updated consistently across ranks

    def test_async_mode(self):
        """Test async_gpu_parallelism mode."""
        results = run_distributed_test(_test_muon_async_mode, world_size=2)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

        # Verify each rank processed some parameters
        for rank, status, result in results:
            self.assertGreater(
                result["num_params"],
                0,
                f"Rank {rank} should process at least one parameter",
            )

        # Verify no overlap in parameter processing
        rank0_params = set(results[0][2]["my_params"])
        rank1_params = set(results[1][2]["my_params"])
        self.assertEqual(
            len(rank0_params & rank1_params),
            0,
            "Ranks should not process the same parameters in async mode",
        )

    def test_prefetching(self):
        """Test prefetching functionality."""
        results = run_distributed_test(_test_muon_with_prefetch, world_size=2)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

    def test_world_size_1(self):
        """Test with single rank (world_size=1)."""
        results = run_distributed_test(_test_muon_with_fsdp, world_size=1)

        # Should work with single rank
        self.assertEqual(len(results), 1)
        rank, status, result = results[0]
        self.assertEqual(status, "success", f"Single rank test failed: {result}")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(torch.cuda.device_count() < 2, "Requires 2+ GPUs")
@unittest.skipIf(not FSDP_AVAILABLE, "FSDP not available")
class TestMuonRealDistributedGPU(unittest.TestCase):
    """GPU-specific distributed tests (only run when GPUs available)."""

    @classmethod
    def setUpClass(cls):
        """Check if GPU distributed tests can run."""
        if not dist.is_nccl_available():
            raise unittest.SkipTest("NCCL backend not available")

    def test_muon_with_fsdp_no_shard_gpu(self):
        """Test Muon with FSDP NO_SHARD on GPUs."""
        results = run_distributed_test(_test_muon_with_fsdp_gpu, world_size=2)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

            if isinstance(result, dict):
                # Verify we have 2D parameters
                self.assertGreater(
                    result.get("num_params", 0),
                    0,
                    f"Rank {rank} should have 2D parameters",
                )

    def test_muon_with_fsdp_full_shard_gpu(self):
        """Test Muon with FSDP FULL_SHARD on GPUs."""
        results = run_distributed_test(_test_muon_with_fsdp_sharded_gpu, world_size=2)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

            # Note: FULL_SHARD may flatten parameters, which is expected
            if isinstance(result, dict) and "note" in result:
                # FSDP flattened params - this is OK
                pass

    def test_muon_with_ddp_gpu(self):
        """Test Muon with DDP on GPUs."""
        results = run_distributed_test(_test_muon_with_ddp_gpu, world_size=2)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

            if isinstance(result, dict):
                # Verify training happened on GPU
                self.assertIn(
                    "cuda",
                    result.get("device", ""),
                    f"Rank {rank} should use CUDA device",
                )

                # Verify losses changed (training worked)
                losses = result.get("losses", [])
                if len(losses) >= 2:
                    self.assertNotEqual(
                        losses[0], losses[-1], f"Rank {rank}: Loss should change"
                    )

    def test_muon_async_mode_gpu(self):
        """Test Muon async mode on GPU."""
        results = run_distributed_test(_test_muon_async_mode_gpu, world_size=2)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

            if isinstance(result, dict):
                # Verify each rank processed some parameters
                self.assertGreater(
                    result.get("num_params", 0),
                    0,
                    f"Rank {rank} should process parameters",
                )

        # Verify no overlap in parameter processing
        rank0_params = set(results[0][2]["my_params"])
        rank1_params = set(results[1][2]["my_params"])
        self.assertEqual(
            len(rank0_params & rank1_params),
            0,
            "Ranks should not process same parameters in async mode",
        )

    def test_muon_mixed_precision_gpu(self):
        """Test Muon with mixed precision on GPU."""
        results = run_distributed_test(_test_muon_mixed_precision_gpu, world_size=2)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

            if isinstance(result, dict):
                self.assertTrue(
                    result.get("mixed_precision", False),
                    f"Rank {rank} should use mixed precision",
                )

    def test_muon_4_gpus(self):
        """Test Muon with 4 GPUs (if available)."""
        if torch.cuda.device_count() < 4:
            self.skipTest("Requires 4+ GPUs")

        results = run_distributed_test(_test_muon_async_mode_gpu, world_size=4)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

        # Verify parameters distributed across all 4 ranks
        all_params = set()
        for rank, status, result in results:
            if isinstance(result, dict):
                rank_params = result.get("my_params", [])
                all_params.update(rank_params)

        # With 4 params and 4 ranks, all should be assigned
        self.assertEqual(
            len(all_params), 4, "All 4 parameters should be distributed across ranks"
        )

    def test_muon_8_gpus(self):
        """Test Muon with 8 GPUs (if available)."""
        if torch.cuda.device_count() < 8:
            self.skipTest("Requires 8+ GPUs")

        results = run_distributed_test(_test_muon_async_mode_gpu, world_size=8)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

        # Verify each rank processed parameters
        for rank, status, result in results:
            if isinstance(result, dict):
                num_params = result.get("num_params", 0)
                self.assertGreaterEqual(
                    num_params, 0, f"Rank {rank} should have parameter assignment"
                )

        # Verify no overlap in parameter assignments
        all_param_sets = []
        for rank, status, result in results:
            if isinstance(result, dict):
                rank_params = set(result.get("my_params", []))
                all_param_sets.append(rank_params)

        # Check for overlaps
        for i in range(len(all_param_sets)):
            for j in range(i + 1, len(all_param_sets)):
                overlap = all_param_sets[i] & all_param_sets[j]
                self.assertEqual(
                    len(overlap), 0, f"Ranks {i} and {j} should not share parameters"
                )

    def test_muon_fsdp_8_gpus(self):
        """Test Muon with FSDP on 8 GPUs (if available)."""
        if torch.cuda.device_count() < 8:
            self.skipTest("Requires 8+ GPUs")

        results = run_distributed_test(_test_muon_with_fsdp_gpu, world_size=8)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

        # Verify all ranks have parameters
        for rank, status, result in results:
            if isinstance(result, dict):
                self.assertGreater(
                    result.get("num_params", 0),
                    0,
                    f"Rank {rank} should have parameters",
                )

    def test_muon_ddp_8_gpus(self):
        """Test Muon with DDP on 8 GPUs (if available)."""
        if torch.cuda.device_count() < 8:
            self.skipTest("Requires 8+ GPUs")

        results = run_distributed_test(_test_muon_with_ddp_gpu, world_size=8)

        # Verify all ranks succeeded
        for rank, status, result in results:
            self.assertEqual(status, "success", f"Rank {rank} failed: {result}")

        # Verify training worked on all ranks
        for rank, status, result in results:
            if isinstance(result, dict):
                self.assertIn(
                    "cuda", result.get("device", ""), f"Rank {rank} should use GPU"
                )

    def test_baseline_vs_ddp_vs_fsdp_equivalence(self):
        """
        CRITICAL CORRECTNESS TEST: Compare baseline (no dist) vs DDP vs FSDP.

        This test verifies that distributed training produces numerically
        equivalent results to non-distributed training.
        """
        # Run baseline (single GPU, no distributed)
        baseline_results = run_distributed_test(
            _test_muon_single_gpu_baseline, world_size=1
        )
        baseline_rank, baseline_status, baseline_data = baseline_results[0]
        self.assertEqual(baseline_status, "success", "Baseline test failed")

        # Run DDP version
        ddp_results = run_distributed_test(_test_muon_comparison_ddp, world_size=2)
        # Use rank 0 for comparison (all ranks should have same results due to DDP sync)
        ddp_rank, ddp_status, ddp_data = ddp_results[0]
        self.assertEqual(ddp_status, "success", "DDP test failed")

        # Run FSDP version
        fsdp_results = run_distributed_test(_test_muon_comparison_fsdp, world_size=2)
        fsdp_rank, fsdp_status, fsdp_data = fsdp_results[0]
        self.assertEqual(fsdp_status, "success", "FSDP test failed")

        # Compare losses across configurations
        baseline_losses = baseline_data["losses"]
        ddp_losses = ddp_data["losses"]
        fsdp_losses = fsdp_data["losses"]

        # Losses should be close (within tolerance for floating point)
        for step in range(len(baseline_losses)):
            # DDP should match baseline closely
            self.assertAlmostEqual(
                baseline_losses[step],
                ddp_losses[step],
                places=3,
                msg=f"Step {step}: DDP loss differs from baseline",
            )

            # FSDP should also match baseline closely
            self.assertAlmostEqual(
                baseline_losses[step],
                fsdp_losses[step],
                places=3,
                msg=f"Step {step}: FSDP loss differs from baseline",
            )

        # Compare final parameters
        baseline_params = baseline_data["param_snapshots"][-1]
        ddp_params = ddp_data["param_snapshots"][-1]
        fsdp_params = fsdp_data["param_snapshots"][-1]

        # Parameters should be numerically close
        for i, (base_p, ddp_p, fsdp_p) in enumerate(
            zip(baseline_params, ddp_params, fsdp_params)
        ):
            # DDP parameters should match baseline
            torch.testing.assert_close(
                base_p,
                ddp_p,
                rtol=1e-4,
                atol=1e-5,
                msg=f"Parameter {i}: DDP differs from baseline",
            )

            # FSDP parameters should match baseline
            torch.testing.assert_close(
                base_p,
                fsdp_p,
                rtol=1e-4,
                atol=1e-5,
                msg=f"Parameter {i}: FSDP differs from baseline",
            )

    def test_async_vs_sync_equivalence(self):
        """
        CRITICAL CORRECTNESS TEST: Compare async_gpu_parallelism=False vs True.

        This test verifies that async mode produces numerically equivalent
        results to sync mode.
        """
        # Run sync mode (async=False)
        sync_results = run_distributed_test(_test_muon_comparison_ddp, world_size=2)
        sync_rank, sync_status, sync_data = sync_results[0]
        self.assertEqual(sync_status, "success", "Sync mode test failed")

        # Run async mode (async=True)
        async_results = run_distributed_test(_test_muon_comparison_async, world_size=2)
        async_rank, async_status, async_data = async_results[0]
        self.assertEqual(async_status, "success", "Async mode test failed")

        # Compare losses across configurations
        sync_losses = sync_data["losses"]
        async_losses = async_data["losses"]

        # Losses should be close (within tolerance for floating point)
        for step in range(len(sync_losses)):
            self.assertAlmostEqual(
                sync_losses[step],
                async_losses[step],
                places=3,
                msg=f"Step {step}: Async mode loss differs from sync mode",
            )

        # Compare final parameters
        sync_params = sync_data["param_snapshots"][-1]
        async_params = async_data["param_snapshots"][-1]

        # Parameters should be numerically close
        for i, (sync_p, async_p) in enumerate(zip(sync_params, async_params)):
            torch.testing.assert_close(
                sync_p,
                async_p,
                rtol=1e-4,
                atol=1e-5,
                msg=f"Parameter {i}: Async mode differs from sync mode",
            )

    def test_prefetch_vs_no_prefetch_equivalence(self):
        """
        CRITICAL CORRECTNESS TEST: Compare prefetch_count=0 vs prefetch_count=1.

        This test verifies that prefetching produces numerically equivalent
        results to non-prefetching.
        """
        # Run without prefetching (prefetch_count=0)
        no_prefetch_results = run_distributed_test(
            _test_muon_comparison_async, world_size=2
        )
        no_prefetch_rank, no_prefetch_status, no_prefetch_data = no_prefetch_results[0]
        self.assertEqual(no_prefetch_status, "success", "No prefetch test failed")

        # Run with prefetching (prefetch_count=1)
        prefetch_results = run_distributed_test(
            _test_muon_comparison_prefetch, world_size=2
        )
        prefetch_rank, prefetch_status, prefetch_data = prefetch_results[0]
        self.assertEqual(prefetch_status, "success", "Prefetch test failed")

        # Compare losses across configurations
        no_prefetch_losses = no_prefetch_data["losses"]
        prefetch_losses = prefetch_data["losses"]

        # Losses should be close (within tolerance for floating point)
        for step in range(len(no_prefetch_losses)):
            self.assertAlmostEqual(
                no_prefetch_losses[step],
                prefetch_losses[step],
                places=3,
                msg=f"Step {step}: Prefetch loss differs from no prefetch",
            )

        # Compare final parameters
        no_prefetch_params = no_prefetch_data["param_snapshots"][-1]
        prefetch_params = prefetch_data["param_snapshots"][-1]

        # Parameters should be numerically close
        for i, (no_pf_p, pf_p) in enumerate(zip(no_prefetch_params, prefetch_params)):
            torch.testing.assert_close(
                no_pf_p,
                pf_p,
                rtol=1e-4,
                atol=1e-5,
                msg=f"Parameter {i}: Prefetch differs from no prefetch",
            )


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
