"""
Test script for KernelBench RL environment.

This script tests the environment setup without running a full training loop.
"""

import asyncio
import os
from pathlib import Path

from kernelbench_env import (
    KernelBenchEnv,
    KernelBenchDatasetBuilder,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import model_info


async def test_single_env():
    """Test a single environment step."""

    print("=" * 80)
    print("Testing KernelBench RL Environment")
    print("=" * 80)

    # Load a simple test problem
    test_reference = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(1024, 1024).cuda()]

def get_init_inputs():
    return []
'''

    # Create renderer
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, get_tokenizer(model_name))

    print(f"\n1. Creating environment...")
    print(f"   Model: {model_name}")
    print(f"   Renderer: {renderer_name}")

    # Create environment
    env = KernelBenchEnv(
        reference_src=test_reference,
        problem_name="test_multiply",
        problem_description="Simple element-wise multiplication test",
        renderer=renderer,
        gpu="L40S",
        backend="triton",
        precision="fp32",
        num_correct_trials=1,  # Fewer trials for testing
        num_perf_trials=10,
        verbose=True,
    )

    print("\n2. Getting initial observation...")
    observation, stop_condition = await env.initial_observation()
    print(f"   Observation type: {type(observation)}")
    print(f"   Stop condition: {stop_condition}")

    print("\n3. Environment created successfully!")
    print("   To test a full step, you would need to:")
    print("   - Generate code using a Tinker sampling client")
    print("   - Call env.step(action)")
    print("   - Receive reward and metrics")

    print("\n" + "=" * 80)


async def test_dataset_builder():
    """Test dataset builder."""

    print("\n" + "=" * 80)
    print("Testing KernelBench Dataset Builder")
    print("=" * 80)

    # Create dataset builder
    print("\n1. Creating dataset builder...")
    builder = KernelBenchDatasetBuilder(
        level=1,
        batch_size=2,
        group_size=2,
        model_name_for_tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        renderer_name="qwen3",
        gpu="L40S",
        backend="triton",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=10,
        num_epochs=1,
        test_split=0.2,
        dataset_source="huggingface",
        verbose=False,
    )

    print("\n2. Building datasets...")
    train_dataset, test_dataset = await builder()

    print(f"   Train dataset size: {len(train_dataset)} batches")
    print(f"   Test dataset size: {len(test_dataset)} batches")
    print(f"   Problems per batch: {train_dataset.batch_size}")
    print(f"   Environments per problem: {train_dataset.group_size}")

    print("\n3. Getting first batch...")
    batch = train_dataset.get_batch(0)
    print(f"   Batch contains {len(batch)} environment group builders")

    if len(batch) > 0:
        builder = batch[0]
        print(f"   First problem: {builder.problem_name}")
        print(f"   GPU: {builder.gpu}")
        print(f"   Backend: {builder.backend}")

    print("\n" + "=" * 80)


async def main():
    """Run all tests."""

    # Load environment variables
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        print("Loading environment variables from .env...")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Handle both "KEY=value" and "export KEY=value" formats
                    if line.startswith("export "):
                        line = line[7:]  # Remove "export "
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
    else:
        print("Warning: .env file not found!")

    # Check for required API keys
    if "TINKER_API_KEY" not in os.environ:
        print("\nError: TINKER_API_KEY not found in environment!")
        print("Please set it in .env file or environment variables.")
        return

    try:
        # Test single environment
        await test_single_env()

        # Test dataset builder
        await test_dataset_builder()

        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Run training: python train.py --level 1")
        print("  2. Monitor with W&B: python train.py --wandb_project my-project")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
