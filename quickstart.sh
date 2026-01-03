#!/bin/bash
# Quickstart script for KernelBench RL Environment

echo "=================================="
echo "KernelBench RL Environment Setup"
echo "=================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create .env with your TINKER_API_KEY"
    exit 1
fi

echo ""
echo "1. Checking environment variables..."
source .env
if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY not set in .env"
    exit 1
fi
echo "   ✓ TINKER_API_KEY found"

echo ""
echo "2. Checking Modal authentication..."
if ! modal token new --verify 2>/dev/null; then
    echo "   Setting up Modal authentication..."
    modal token new
else
    echo "   ✓ Modal authenticated"
fi

echo ""
echo "3. Testing environment setup..."
python test_env.py

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "To start training:"
echo "  python train.py --level 1"
echo ""
echo "For more options:"
echo "  python train.py --help"
echo ""
