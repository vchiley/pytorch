#!/bin/bash
# Script to run all Muon distributed tests

echo "=================================="
echo "Running Muon Distributed Tests"
echo "=================================="
echo ""

# Set Python path to include PyTorch source
export PYTHONPATH=/data/users/vchiley/pytorch:$PYTHONPATH

echo "1. Running Unit Tests..."
echo "========================"
python3 test/optim/test_muon_distributed.py
UNIT_STATUS=$?
echo ""

echo "2. Running End-to-End Tests..."
echo "==============================="
python3 test/optim/test_muon_e2e.py
E2E_STATUS=$?
echo ""

echo "=================================="
echo "Test Summary"
echo "=================================="
if [ $UNIT_STATUS -eq 0 ]; then
    echo "✅ Unit Tests: PASSED"
else
    echo "❌ Unit Tests: FAILED (exit code: $UNIT_STATUS)"
fi

if [ $E2E_STATUS -eq 0 ]; then
    echo "✅ E2E Tests: PASSED"
else
    echo "❌ E2E Tests: FAILED (exit code: $E2E_STATUS)"
fi
echo ""

if [ $UNIT_STATUS -eq 0 ] && [ $E2E_STATUS -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    exit 1
fi
