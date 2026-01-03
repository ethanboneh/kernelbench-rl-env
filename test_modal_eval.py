"""
Test Modal evaluation integration.

This tests the full pipeline: environment -> Modal -> KernelBench evaluation.
"""

import asyncio
import os
from pathlib import Path

from kernelbench_env import KernelBenchEnv
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import model_info


async def test_modal_evaluation():
    """Test a complete evaluation cycle with Modal."""

    print("=" * 80)
    print("Testing Modal Evaluation Integration")
    print("=" * 80)

    # Simple test problem
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

    # Test kernel (intentionally simple, should work)
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

    # Create renderer
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, get_tokenizer(model_name))

    print("\n1. Creating environment...")
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
    print(f"   ✓ Observation created")

    print("\n3. Simulating agent action (using pre-written kernel)...")
    # Convert test kernel to token action
    # In real training, this would come from the model
    action_tokens = renderer.tokenizer.encode(test_kernel)
    print(f"   Action length: {len(action_tokens)} tokens")

    print("\n4. Evaluating kernel on Modal...")
    print("   This will take ~30-60 seconds for Modal cold start...")

    try:
        step_result = await env.step(action_tokens)

        print("\n5. Results:")
        print(f"   Episode done: {step_result.episode_done}")
        print(f"   Reward: {step_result.reward:.4f}")
        print(f"   Metrics: {step_result.metrics}")

        if step_result.metrics:
            compiled = step_result.metrics.get("compiled", 0)
            correct = step_result.metrics.get("correct", 0)
            runtime = step_result.metrics.get("runtime_us", "N/A")
            speedup = step_result.metrics.get("speedup", "N/A")

            print(f"\n   ✓ Compiled: {compiled}")
            print(f"   ✓ Correct: {correct}")
            print(f"   ✓ Runtime: {runtime} μs")
            print(f"   ✓ Speedup: {speedup}x")

        print("\n" + "=" * 80)
        print("✓ Modal evaluation test PASSED!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Modal evaluation test FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)


async def main():
    """Run the test."""

    # Load environment variables
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

    # Check for API key
    if "TINKER_API_KEY" not in os.environ:
        print("Error: TINKER_API_KEY not found!")
        return

    print("\nNote: This test requires Modal to be configured.")
    print("Run 'modal token new' if you haven't already.\n")

    await test_modal_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
