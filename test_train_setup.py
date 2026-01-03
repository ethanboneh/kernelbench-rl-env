"""
Diagnostic script to test training setup step by step.
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("Testing Training Setup")
print("=" * 80)

# Load environment
print("\n1. Loading environment...")
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
print("   ✓ Environment loaded")

# Test imports
print("\n2. Testing imports...")
try:
    import chz
    print("   ✓ chz")
except Exception as e:
    print(f"   ✗ chz: {e}")
    sys.exit(1)

try:
    from tinker_cookbook.rl import train
    print("   ✓ tinker_cookbook.rl.train")
except Exception as e:
    print(f"   ✗ tinker_cookbook.rl.train: {e}")
    sys.exit(1)

try:
    from kernelbench_env import KernelBenchDatasetBuilder
    print("   ✓ kernelbench_env")
except Exception as e:
    print(f"   ✗ kernelbench_env: {e}")
    sys.exit(1)

# Test config creation
print("\n3. Creating config...")
try:
    from train import CLIConfig, build_config

    cli_config = CLIConfig(
        level=1,
        batch_size=1,
        group_size=2,
        eval_every=1,
        max_tokens=512,
    )

    print(f"   ✓ CLIConfig created")
    print(f"     - Level: {cli_config.level}")
    print(f"     - Batch size: {cli_config.batch_size}")
    print(f"     - Group size: {cli_config.group_size}")

    config = build_config(cli_config)
    print(f"   ✓ Training config built")
    print(f"     - Model: {config.model_name}")
    print(f"     - Log path: {config.log_path}")

except Exception as e:
    print(f"   ✗ Config creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test dataset builder
print("\n4. Testing dataset builder...")
try:
    import asyncio

    async def test_dataset():
        dataset_builder = config.dataset_builder
        print(f"   Dataset builder type: {type(dataset_builder).__name__}")
        print(f"   Building datasets...")
        train_dataset, test_dataset = await dataset_builder()
        print(f"   ✓ Datasets built")
        print(f"     - Train batches: {len(train_dataset)}")
        print(f"     - Test batches: {len(test_dataset)}")
        return train_dataset, test_dataset

    train_dataset, test_dataset = asyncio.run(test_dataset())

except Exception as e:
    print(f"   ✗ Dataset building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All setup tests passed!")
print("=" * 80)
print("\nTraining should be able to start successfully.")
print("If train.py still hangs, it may be during Tinker API initialization.")
