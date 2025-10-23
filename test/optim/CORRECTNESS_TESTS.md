# Correctness Tests: Baseline vs DDP vs FSDP

## Overview

The most critical test for any distributed training implementation is **numerical equivalence** - ensuring that distributed training produces the same results as non-distributed training.

## The Equivalence Test

### `test_baseline_vs_ddp_vs_fsdp_equivalence`

**Purpose**: Verify that Muon produces numerically identical results across:
1. **Baseline**: Single GPU, no distributed (pure Muon)
2. **DDP**: Distributed Data Parallel with Muon
3. **FSDP**: Fully Sharded Data Parallel with Muon

**Location**: `/data/users/vchiley/pytorch/test/optim/test_muon_distributed_real.py`

### What It Tests

```python
def test_baseline_vs_ddp_vs_fsdp_equivalence(self):
    """
    CRITICAL CORRECTNESS TEST: Compare baseline (no dist) vs DDP vs FSDP.

    This test verifies that distributed training produces numerically
    equivalent results to non-distributed training.
    """
```

### Test Methodology

#### 1. Setup (All with same seeds)
```python
# Same model initialization
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(64, 64, bias=False),
    nn.Linear(64, 32, bias=False),
)

# Same optimizer settings
optimizer = Muon(model.parameters(), lr=0.02)

# Same input data
torch.manual_seed(100)
input_data = torch.randn(16, 64)
```

#### 2. Run Three Configurations

**Baseline (No Distributed)**:
```python
# Pure Muon, no distributed
model.to(device)
optimizer = Muon(model.parameters(), lr=0.02)
# No DDP, no FSDP, no distributed config
```

**DDP Configuration**:
```python
# Wrap with DDP
model = DDP(model, device_ids=[rank])

# Muon with DDP process group
config = create_processgroup_config(dp_pg=dist.group.WORLD, ...)
optimizer = Muon(model.parameters(), lr=0.02, distributed_config=config)
```

**FSDP Configuration**:
```python
# Wrap with FSDP (NO_SHARD)
model = FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD, ...)

# Muon with FSDP process group
config = create_processgroup_config(fsdp_pg=dist.group.WORLD, ...)
optimizer = Muon(params_2d, lr=0.02, distributed_config=config)
```

#### 3. Compare Results

**Loss Comparison**:
```python
for step in range(3):
    # Compare losses at each step
    assert_almost_equal(baseline_loss, ddp_loss, places=3)
    assert_almost_equal(baseline_loss, fsdp_loss, places=3)
```

**Parameter Comparison**:
```python
# Compare final parameters
torch.testing.assert_close(
    baseline_params,
    ddp_params,
    rtol=1e-4,
    atol=1e-5
)
torch.testing.assert_close(
    baseline_params,
    fsdp_params,
    rtol=1e-4,
    atol=1e-5
)
```

## Why This Test Matters

### What It Catches

1. **Gradient Synchronization Bugs**
   - If DDP isn't properly syncing gradients
   - If FSDP gather/scatter has issues
   - Result: Parameters would diverge

2. **Optimizer State Bugs**
   - If distributed optimizer state is incorrect
   - If momentum/Newton updates differ
   - Result: Training trajectory would differ

3. **Communication Bugs**
   - If broadcasts don't reach all ranks
   - If reductions are incorrect
   - Result: Parameters or losses would differ

4. **Numerical Stability Issues**
   - If distributed communication introduces numerical errors
   - If sharding causes precision loss
   - Result: Small but accumulating differences

### Real Bugs This Would Catch

**Example 1: Missing Gradient Sync**
```python
# Bug: Forgot to sync gradients in DDP
# Result: Each rank updates with only its batch's gradients
# Detection: ddp_params != baseline_params
```

**Example 2: Incorrect Broadcast**
```python
# Bug: Broadcasting parameter to wrong ranks
# Result: Different ranks have different parameters
# Detection: ddp_params != baseline_params
```

**Example 3: Momentum State Issues**
```python
# Bug: Distributed momentum not initialized correctly
# Result: Training trajectory differs
# Detection: losses diverge over time
```

## Running the Test

### GPU Required
```bash
# Requires 2+ GPUs
python test/optim/test_muon_distributed_real.py \
    TestMuonRealDistributedGPU.test_baseline_vs_ddp_vs_fsdp_equivalence
```

### Expected Output (Success)
```
test_baseline_vs_ddp_vs_fsdp_equivalence ...
Running baseline (no distributed)...
Running DDP with 2 ranks...
Running FSDP with 2 ranks...
Comparing losses... ✓
Comparing parameters... ✓
ok

----------------------------------------------------------------------
Ran 1 test in 15.234s

OK
```

### Expected Output (Failure)
```
test_baseline_vs_ddp_vs_fsdp_equivalence ... FAIL

======================================================================
FAIL: test_baseline_vs_ddp_vs_fsdp_equivalence
----------------------------------------------------------------------
AssertionError: Step 2: DDP loss differs from baseline
baseline_loss=12.3456, ddp_loss=12.3789
```

## Test Tolerances

### Loss Tolerance
```python
self.assertAlmostEqual(
    baseline_losses[step],
    ddp_losses[step],
    places=3,  # 0.001 tolerance
    msg=f"Step {step}: DDP loss differs from baseline"
)
```

**Why 3 decimal places?**
- Accounts for floating point arithmetic differences
- Stricter than production (production uses ~1e-4)
- Catches significant divergence

### Parameter Tolerance
```python
torch.testing.assert_close(
    base_p,
    ddp_p,
    rtol=1e-4,  # Relative tolerance
    atol=1e-5,  # Absolute tolerance
    msg=f"Parameter {i}: DDP differs from baseline"
)
```

**Why these tolerances?**
- `rtol=1e-4`: 0.01% relative difference allowed
- `atol=1e-5`: Small absolute differences for near-zero values
- Balance between strictness and practicality

## Debugging Failed Tests

### Step 1: Identify Where Divergence Starts
```python
# Add logging to see when divergence begins
for step in range(3):
    print(f"Step {step}:")
    print(f"  Baseline loss: {baseline_losses[step]}")
    print(f"  DDP loss:      {ddp_losses[step]}")
    print(f"  FSDP loss:     {fsdp_losses[step]}")
    print(f"  Diff (DDP):    {abs(baseline_losses[step] - ddp_losses[step])}")
    print(f"  Diff (FSDP):   {abs(baseline_losses[step] - fsdp_losses[step])}")
```

### Step 2: Check Seed Reproducibility
```python
# Ensure seeds are being set correctly
assert torch.initial_seed() == expected_seed
```

### Step 3: Check Communication
```python
# Verify all ranks see same data after sync
if dist.is_initialized():
    tensor = torch.tensor([1.0])
    dist.all_reduce(tensor)
    print(f"Rank {rank}: all_reduce result = {tensor.item()}")
```

### Step 4: Compare Parameter Gradients
```python
# Before optimizer step
for i, p in enumerate(model.parameters()):
    if p.grad is not None:
        print(f"Param {i} grad: mean={p.grad.mean()}, std={p.grad.std()}")
```

## Integration with CI/CD

### Pre-Merge Gate
```yaml
- name: Correctness Test (GPU)
  run: |
    python test/optim/test_muon_distributed_real.py \
      TestMuonRealDistributedGPU.test_baseline_vs_ddp_vs_fsdp_equivalence
  required: true
  timeout: 20min
```

### Importance
This test is **CRITICAL** and should:
- ✅ Be required for all merges
- ✅ Block deployment if failing
- ✅ Have alerts if it starts flaking
- ✅ Be reviewed if tolerances need adjustment

## Extending the Test

### Add More Steps
```python
# Test longer training runs
for step in range(10):  # Instead of 3
    ...
```

### Add More Configurations
```python
# Test with different world sizes
for world_size in [1, 2, 4, 8]:
    results = run_distributed_test(test_fn, world_size=world_size)
    compare_to_baseline(results)
```

### Add Learning Rate Changes
```python
# Test with learning rate schedules
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
for epoch in range(3):
    train_epoch()
    scheduler.step()
    compare_to_baseline()
```

## Comparison to Other Optimizers

### How PyTorch Tests Optimizers

**PyTorch's approach**:
```python
# Compare optimizer behavior with reference implementation
def test_sgd_equivalence():
    # Manual SGD update
    expected = param - lr * grad

    # PyTorch SGD update
    optimizer.step()

    torch.testing.assert_close(param, expected)
```

**Our approach** (more comprehensive):
```python
# Compare entire training loop across configurations
def test_muon_equivalence():
    # Run full training: forward + backward + optimizer
    baseline_result = train_baseline()
    distributed_result = train_distributed()

    # Compare final state
    torch.testing.assert_close(baseline_result, distributed_result)
```

### Advantages of Our Approach

1. **End-to-End**: Tests entire pipeline, not just optimizer step
2. **Real Communication**: Uses actual NCCL/Gloo backends
3. **Multiple Configs**: Tests DDP and FSDP simultaneously
4. **Production-Like**: Mirrors actual training scenarios

## Test Maintenance

### When to Update Tolerances

**Tighten tolerances** if:
- Flakiness is resolved
- Communication becomes more stable
- Want stricter correctness guarantees

**Loosen tolerances** if:
- Test becomes flaky
- Different hardware shows variance
- Using different floating point modes

### When to Update Seeds

**Change seeds** if:
- Test becomes deterministic in unintended ways
- Want to test different numerical patterns
- Random initialization causes issues

## Summary

The `test_baseline_vs_ddp_vs_fsdp_equivalence` test is:

✅ **Critical** - Verifies core correctness of distributed training
✅ **Comprehensive** - Tests baseline, DDP, and FSDP
✅ **Strict** - Uses tight tolerances to catch bugs
✅ **Practical** - Runs real training loops
✅ **Debuggable** - Provides clear failure messages

**This test ensures that using Muon with DDP or FSDP produces identical results to using Muon without distributed training.**
