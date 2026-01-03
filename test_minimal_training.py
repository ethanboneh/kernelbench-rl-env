"""
Minimal training test to identify where the hang occurs.

This simulates what the training loop does but with detailed logging.
"""

import asyncio
import os
import sys
from pathlib import Path

print("=" * 80)
print("Minimal Training Test - Identifying Hang Location")
print("=" * 80)

# Load environment
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

if "TINKER_API_KEY" not in os.environ:
    print("Error: TINKER_API_KEY not found!")
    sys.exit(1)


async def test_single_rollout():
    """Test a single rollout without the full training loop."""

    print("\n[1/7] Importing modules...")
    from kernelbench_env import KernelBenchDatasetBuilder
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook import model_info
    print("   ✓ Imports successful")

    print("\n[2/7] Creating dataset builder...")
    dataset_builder = KernelBenchDatasetBuilder(
        level=1,
        batch_size=1,
        group_size=1,  # Just 1 environment for testing
        model_name_for_tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        renderer_name="qwen3_instruct",
        gpu="L40S",
        backend="triton",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=10,
        num_epochs=1,
        verbose=True,
    )
    print("   ✓ Dataset builder created")

    print("\n[3/7] Building datasets...")
    train_dataset, test_dataset = await dataset_builder()
    print(f"   ✓ Train: {len(train_dataset)} batches, Test: {len(test_dataset)} batches")

    print("\n[4/7] Getting first batch...")
    batch = train_dataset.get_batch(0)
    print(f"   ✓ Batch has {len(batch)} environment group builders")

    print("\n[5/7] Creating environment from first builder...")
    env_group_builder = batch[0]
    print(f"   Problem: {env_group_builder.problem_name}")
    print(f"   GPU: {env_group_builder.gpu}")
    print(f"   Backend: {env_group_builder.backend}")

    envs = await env_group_builder.make_envs()
    env = envs[0]
    print(f"   ✓ Environment created")

    print("\n[6/7] Getting initial observation...")
    observation, stop_condition = await env.initial_observation()
    print(f"   ✓ Observation created (type: {type(observation).__name__})")
    print(f"   Stop condition: {stop_condition}")

    # Create a simple test kernel (same as in test_modal_eval.py)
    test_kernel = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def multiply_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        multiply_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output

def get_inputs():
    return [torch.randn(1024, 1024).cuda()]

def get_init_inputs():
    return []
'''

    # Convert to tokens
    renderer = env_group_builder.renderer
    action_tokens = renderer.tokenizer.encode(test_kernel)
    print(f"   Action: {len(action_tokens)} tokens")

    print("\n[7/7] Executing environment step (this calls Modal)...")
    print("   This should take 30-60 seconds for Modal...")

    import time
    start_time = time.time()

    try:
        step_result = await env.step(action_tokens)
        elapsed = time.time() - start_time

        print(f"\n   ✓ Step completed in {elapsed:.1f}s")
        print(f"\n   Results:")
        print(f"   - Reward: {step_result.reward:.4f}")
        print(f"   - Episode done: {step_result.episode_done}")
        print(f"   - Metrics: {step_result.metrics}")

        print("\n" + "=" * 80)
        print("✓ Test PASSED - Environment works correctly!")
        print("=" * 80)

        if step_result.reward == 0:
            print("\nWARNING: Reward is 0 - check if Modal evaluation worked")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n   ✗ Step FAILED after {elapsed:.1f}s")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\nStarting minimal training test...")
    print("This will:")
    print("  1. Load a single problem from KernelBench")
    print("  2. Create one environment")
    print("  3. Run one step (evaluate a test kernel on Modal)")
    print("  4. Check if metrics are returned")
    print()

    success = await test_single_rollout()

    if success:
        print("\n" + "=" * 80)
        print("Next Step: Try full training")
        print("=" * 80)
        print("\nIf this test worked, the training should work too.")
        print("The hang might be in:")
        print("  - Tinker service initialization (downloading model)")
        print("  - W&B initialization")
        print("  - RL training loop setup")
        print("\nTry running with verbose logging:")
        print("  python train.py level=1 batch_size=1 group_size=1 verbose=True")
    else:
        print("\n" + "=" * 80)
        print("Environment test FAILED - fix this before training")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
