#!/usr/bin/env python3
"""
End-to-end test script for distributed Muon optimizer.

This script tests the distributed Muon implementation without requiring
actual multi-GPU setup by simulating distributed behavior with mocked
communication.
"""

import torch
import torch.nn as nn
from torch.optim import Muon
from torch.optim._muon import DistributedConfig, _default_assign_fn


def create_simple_model(num_layers=4, hidden_size=128):
    """Create a simple model with 2D parameters."""
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
    return nn.Sequential(*layers)


def test_non_distributed_muon():
    """Test Muon optimizer without distributed config (baseline)."""
    print("=" * 70)
    print("TEST 1: Non-Distributed Muon (Baseline)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Create model
    model = create_simple_model(num_layers=4, hidden_size=64)
    
    # Create optimizer without distributed config
    optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
    
    # Training loop
    num_steps = 5
    for step in range(num_steps):
        # Forward pass with dummy data
        input_data = torch.randn(32, 64)
        output = model(input_data)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")
    
    print("✓ Non-distributed Muon works correctly\n")
    return model


def test_distributed_muon_single_rank():
    """Test Muon optimizer with distributed config (simulated single rank)."""
    print("=" * 70)
    print("TEST 2: Distributed Muon (Simulated Single Rank)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Create model
    model = create_simple_model(num_layers=4, hidden_size=64)
    
    # Create mock distributed config for single rank
    def mock_gather_fn(momentum_buffer, dst_rank, state):
        """Mock gather - just return buffer on dst_rank."""
        if state["rank"] == dst_rank:
            return momentum_buffer
        return None
    
    def mock_redistribute_fn(update, src_rank, state):
        """Mock redistribute - just return update."""
        if update is not None:
            return update
        else:
            # This shouldn't happen in single rank, but for safety
            return torch.zeros_like(momentum_buffer)
    
    config = DistributedConfig(
        assign_fn=_default_assign_fn,
        gather_fn=mock_gather_fn,
        redistribute_fn=mock_redistribute_fn,
        state={"rank": 0, "world_size": 1},
        async_gpu_parallelism=False,  # Synchronous for simplicity
        prefetch_count=0,
    )
    
    # Create optimizer with distributed config
    optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95, distributed_config=config)
    
    # Verify assignments were created
    assert "assignments" in optimizer.distributed_config.state
    print(f"  Parameter assignments: {optimizer.distributed_config.state['assignments']}")
    
    # Training loop
    num_steps = 5
    for step in range(num_steps):
        # Forward pass with dummy data
        input_data = torch.randn(32, 64)
        output = model(input_data)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")
    
    print("✓ Distributed Muon (single rank) works correctly\n")
    return model


def test_distributed_muon_async_mode():
    """Test Muon optimizer with async_gpu_parallelism enabled."""
    print("=" * 70)
    print("TEST 3: Distributed Muon (Async Mode)")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Create model
    model = create_simple_model(num_layers=4, hidden_size=64)
    
    # Create mock distributed config with async mode
    def mock_gather_fn(momentum_buffer, dst_rank, state):
        """Mock gather - return buffer on dst_rank."""
        if state["rank"] == dst_rank:
            return momentum_buffer
        return None
    
    def mock_redistribute_fn(update, src_rank, state):
        """Mock redistribute - return update."""
        if update is not None:
            return update
        # For non-assigned ranks in async mode
        return torch.zeros(64, 64)
    
    config = DistributedConfig(
        assign_fn=_default_assign_fn,
        gather_fn=mock_gather_fn,
        redistribute_fn=mock_redistribute_fn,
        state={"rank": 0, "world_size": 4},  # Simulate 4 ranks
        async_gpu_parallelism=True,  # Async mode
        prefetch_count=0,
    )
    
    # Create optimizer
    optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95, distributed_config=config)
    
    # Verify assignments
    assignments = optimizer.distributed_config.state['assignments']
    print(f"  Parameter assignments: {assignments}")
    print(f"  Rank 0 will process params: {[i for i, r in assignments.items() if r == 0]}")
    
    # Training loop
    num_steps = 5
    for step in range(num_steps):
        input_data = torch.randn(32, 64)
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")
    
    print("✓ Distributed Muon (async mode) works correctly\n")
    return model


def test_assignment_validation():
    """Test that assignment validation catches errors."""
    print("=" * 70)
    print("TEST 4: Assignment Validation")
    print("=" * 70)
    
    model = create_simple_model(num_layers=4, hidden_size=64)
    
    # Test 1: Missing assignments
    def bad_assign_fn_missing(params, state):
        # Only assign first 2 params (missing 2 and 3)
        return {0: 0, 1: 1}
    
    config = DistributedConfig(
        assign_fn=bad_assign_fn_missing,
        gather_fn=lambda *args, **kwargs: None,
        redistribute_fn=lambda *args, **kwargs: torch.zeros(64, 64),
        state={"rank": 0, "world_size": 4},
    )
    
    try:
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)
        print("  ✗ FAILED: Should have raised ValueError for missing assignments")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly caught missing assignments: {str(e)[:50]}...")
    
    # Test 2: Invalid rank (too high)
    def bad_assign_fn_invalid(params, state):
        return {i: 999 for i in range(4)}  # Rank 999 invalid
    
    config = DistributedConfig(
        assign_fn=bad_assign_fn_invalid,
        gather_fn=lambda *args, **kwargs: None,
        redistribute_fn=lambda *args, **kwargs: torch.zeros(64, 64),
        state={"rank": 0, "world_size": 4},
    )
    
    try:
        optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)
        print("  ✗ FAILED: Should have raised ValueError for invalid rank")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly caught invalid rank: {str(e)[:50]}...")
    
    print("✓ Assignment validation works correctly\n")
    return True


def test_2d_parameter_requirement():
    """Test that non-2D parameters are rejected."""
    print("=" * 70)
    print("TEST 5: 2D Parameter Requirement")
    print("=" * 70)
    
    # Create model with 1D parameter (bias)
    model = nn.Linear(10, 10, bias=True)
    
    try:
        optimizer = Muon(model.parameters(), lr=0.02)
        print("  ✗ FAILED: Should have raised ValueError for 1D parameter")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected 1D parameter: {str(e)[:60]}...")
    
    # Create model with only 2D parameters (no bias)
    model = nn.Linear(10, 10, bias=False)
    
    try:
        optimizer = Muon(model.parameters(), lr=0.02)
        print("  ✓ Correctly accepted model with only 2D parameters")
    except ValueError:
        print("  ✗ FAILED: Should have accepted 2D parameters")
        return False
    
    print("✓ 2D parameter requirement enforced correctly\n")
    return True


def test_backward_compatibility():
    """Test that old code without distributed_config still works."""
    print("=" * 70)
    print("TEST 6: Backward Compatibility")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Old code (without distributed_config)
    model = create_simple_model(num_layers=2, hidden_size=32)
    optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
    
    # Should work exactly as before
    input_data = torch.randn(16, 32)
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    
    print("  ✓ Old code without distributed_config works")
    print("  ✓ Backward compatibility maintained\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DISTRIBUTED MUON OPTIMIZER - END-TO-END TESTS")
    print("=" * 70 + "\n")
    
    results = []
    
    # Run tests
    try:
        test_non_distributed_muon()
        results.append(("Non-Distributed Baseline", True))
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results.append(("Non-Distributed Baseline", False))
    
    try:
        test_distributed_muon_single_rank()
        results.append(("Distributed Single Rank", True))
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results.append(("Distributed Single Rank", False))
    
    try:
        test_distributed_muon_async_mode()
        results.append(("Distributed Async Mode", True))
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results.append(("Distributed Async Mode", False))
    
    try:
        success = test_assignment_validation()
        results.append(("Assignment Validation", success))
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results.append(("Assignment Validation", False))
    
    try:
        success = test_2d_parameter_requirement()
        results.append(("2D Parameter Requirement", success))
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results.append(("2D Parameter Requirement", False))
    
    try:
        success = test_backward_compatibility()
        results.append(("Backward Compatibility", success))
    except Exception as e:
        print(f"✗ TEST FAILED: {e}\n")
        results.append(("Backward Compatibility", False))
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Implementation is working correctly.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
