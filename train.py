"""
Training script for KernelBench RL environment.

This script trains a model to optimize GPU kernels using reinforcement learning
with the KernelBench evaluation framework and Modal cloud GPUs.

Usage:
    python train.py --level 1 --model_name "Qwen/Qwen3-4B-Instruct-2507"
    python train.py --level 2 --gpu H100 --backend triton
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

import chz
from kernelbench_env import KernelBenchDatasetBuilder
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    """CLI configuration for KernelBench RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None

    # KernelBench configuration
    level: int = 1  # KernelBench level (1-4)
    gpu: str = "L40S"  # Modal GPU type
    backend: str = "triton"  # Kernel backend (triton, cuda, cute, tilelang)
    precision: str = "fp32"  # Precision (fp32, fp16, bf16)
    num_correct_trials: int = 3  # Correctness check trials
    num_perf_trials: int = 50  # Performance measurement trials

    # RL training configuration
    group_size: int = 4  # Number of environments per problem
    batch_size: int = 2  # Number of problems per batch
    num_epochs: int = 10  # Number of epochs through the dataset
    test_split: float = 0.2  # Fraction of problems for testing

    # Training hyperparameters
    learning_rate: float = 5e-5  # Learning rate (LoRA typically needs higher LR)
    max_tokens: int = 2048  # Max tokens for generation

    # Logging and evaluation
    eval_every: int = 5  # Evaluate every N steps
    save_every: int = 20  # Save checkpoint every N steps
    wandb_project: str | None = None  # W&B project name
    wandb_name: str | None = None  # W&B run name
    log_path: str | None = None  # Local log directory

    # Dataset source
    dataset_source: str = "huggingface"  # "huggingface" or "local"

    # KernelBench prompt options
    prompt_option: str = "one_shot"  # zero_shot, one_shot, few_shot

    # Verbosity
    verbose: bool = False


def build_config(cli_config: CLIConfig) -> train.Config:
    """Build Tinker RL training config from CLI config."""

    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Create run name
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"kernelbench-level{cli_config.level}-"
        f"{cli_config.backend}-{cli_config.gpu}-"
        f"{cli_config.group_size}group-{cli_config.batch_size}batch-"
        f"{cli_config.learning_rate}lr-{date_and_time}"
    )

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/kernelbench-rl/{run_name}"

    # Set W&B name
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Create dataset builder
    dataset_builder = KernelBenchDatasetBuilder(
        level=cli_config.level,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        gpu=cli_config.gpu,
        backend=cli_config.backend,
        precision=cli_config.precision,
        num_correct_trials=cli_config.num_correct_trials,
        num_perf_trials=cli_config.num_perf_trials,
        num_epochs=cli_config.num_epochs,
        test_split=cli_config.test_split,
        verbose=cli_config.verbose,
        dataset_source=cli_config.dataset_source,
        prompt_option=cli_config.prompt_option,
    )

    # Create training config
    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
    )


def main():
    """Main entry point for training."""

    # Load environment variables (API keys, etc.)
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
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

    # Parse CLI config
    cli_config = chz.entrypoint(CLIConfig)

    # Build training config
    config = build_config(cli_config)

    print("=" * 80)
    print("KernelBench RL Training")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Level: {cli_config.level}")
    print(f"Backend: {cli_config.backend}")
    print(f"GPU: {cli_config.gpu}")
    print(f"Precision: {cli_config.precision}")
    print(f"Group size: {cli_config.group_size}")
    print(f"Batch size: {cli_config.batch_size}")
    print(f"Learning rate: {cli_config.learning_rate}")
    print(f"Max tokens: {cli_config.max_tokens}")
    print(f"Log path: {config.log_path}")
    print("=" * 80)

    # Check if log directory exists
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")

    # Run training
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
