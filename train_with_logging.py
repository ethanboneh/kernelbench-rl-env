"""
Training script with verbose logging to debug hangs.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add unbuffered print
import functools
print = functools.partial(print, flush=True)


def log(msg):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


log("=" * 80)
log("KernelBench RL Training (Verbose)")
log("=" * 80)

# Load environment
log("Loading environment variables...")
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
log("   ✓ Environment loaded")

# Parse arguments
import chz
from train import CLIConfig, build_config

log("Parsing command line arguments...")
cli_config = chz.entrypoint(CLIConfig)
log(f"   Level: {cli_config.level}")
log(f"   Batch size: {cli_config.batch_size}")
log(f"   Group size: {cli_config.group_size}")
log(f"   Model: {cli_config.model_name}")

log("Building training config...")
config = build_config(cli_config)
log(f"   ✓ Config built")
log(f"   Log path: {config.log_path}")

# Import training module
log("Importing Tinker training module...")
from tinker_cookbook.rl import train as rl_train
log("   ✓ Import successful")

# Create log directory
log("Creating log directory...")
from tinker_cookbook import cli_utils
cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
log("   ✓ Log directory ready")

log("=" * 80)
log("Starting Tinker RL Training")
log("=" * 80)
log("")
log("This will now:")
log("  1. Initialize Tinker service (may download model)")
log("  2. Build datasets")
log("  3. Initialize W&B (if enabled)")
log("  4. Start training loop")
log("")
log("This may take 5-15 minutes on first run...")
log("Watch for these messages:")
log("  - 'Downloading model...' (Tinker downloading weights)")
log("  - 'wandb: Syncing...' (W&B initialization)")
log("  - 'Step 1/...' (Training started)")
log("")
log("=" * 80)

async def main():
    log("Calling rl_train.main()...")
    await rl_train.main(config)

if __name__ == "__main__":
    asyncio.run(main())
