# KernelBench RL Environment with Tinker

This project implements a Reinforcement Learning environment for optimizing GPU kernels using [KernelBench](https://github.com/ScalingIntelligence/KernelBench) and [Tinker](https://tinker-docs.thinkingmachines.ai/).

## Overview

The RL agent learns to optimize PyTorch models by writing custom GPU kernels. The environment:
- Provides reference PyTorch implementations as observations
- Evaluates agent-generated kernels on Modal cloud GPUs
- Rewards based on compilation success, correctness, and performance (runtime)

## Architecture

### Components

1. **`kernelbench_env.py`**: Core RL environment implementation
   - `KernelBenchEnv`: Single-episode environment for kernel optimization
   - `KernelBenchEnvGroupBuilder`: Creates environment groups
   - `KernelBenchDataset`: Dataset of KernelBench problems
   - `KernelBenchDatasetBuilder`: Builds train/test datasets

2. **`modal_evaluator.py`**: Modal integration for cloud GPU evaluation
   - Runs KernelBench evaluation on cloud GPUs (L40S, H100, A100, etc.)
   - Returns compilation status, correctness, and performance metrics

3. **`train.py`**: Training script
   - Configurable via command-line arguments
   - Supports W&B logging
   - Checkpoint saving and evaluation

## Setup

### Prerequisites

1. **Tinker API Key**: Sign up at [Tinker](https://tinker-docs.thinkingmachines.ai/)
2. **Modal Account**: Sign up at [Modal](https://modal.com/)

### Installation

```bash
# Install dependencies
pip install tinker modal datasets torch transformers chz

# Set up environment variables
# Edit .env file with your Tinker API key (already present)
# The file should contain:
# export TINKER_API_KEY=your_key_here

# Authenticate Modal
modal token new
```

### Directory Structure

```
kernelbench-rl-env/
├── .env                    # API keys
├── KernelBench/           # KernelBench repository
├── tinker-cookbook/       # Tinker cookbook repository
├── kernelbench_env.py     # RL environment
├── modal_evaluator.py     # Modal GPU evaluation
├── train.py               # Training script
└── README.md              # This file
```

## Usage

### Basic Training

Train on KernelBench Level 1 problems:

```bash
python train.py --level 1
```

### Advanced Configuration

```bash
python train.py \
  --level 2 \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --gpu H100 \
  --backend triton \
  --precision fp32 \
  --batch_size 4 \
  --group_size 8 \
  --learning_rate 5e-5 \
  --num_epochs 10 \
  --wandb_project kernelbench-rl \
  --log_path ./logs/run1
```

### Command-Line Arguments

**Model Configuration:**
- `--model_name`: Base model for training (default: "Qwen/Qwen3-4B-Instruct-2507")
- `--renderer_name`: Renderer for tokenization (auto-detected from model)

**KernelBench Configuration:**
- `--level`: KernelBench difficulty level 1-4 (default: 1)
- `--gpu`: Modal GPU type: L40S, H100, A100, T4, etc. (default: "L40S")
- `--backend`: Kernel backend: triton, cuda, cute, tilelang (default: "triton")
- `--precision`: Precision: fp32, fp16, bf16 (default: "fp32")
- `--num_correct_trials`: Correctness check trials (default: 3)
- `--num_perf_trials`: Performance measurement trials (default: 50)

**RL Training Configuration:**
- `--group_size`: Number of environments per problem (default: 4)
- `--batch_size`: Number of problems per batch (default: 2)
- `--num_epochs`: Number of epochs through dataset (default: 10)
- `--test_split`: Fraction of problems for testing (default: 0.2)

**Training Hyperparameters:**
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_tokens`: Max tokens for generation (default: 2048)

**Logging:**
- `--eval_every`: Evaluate every N steps (default: 5)
- `--save_every`: Save checkpoint every N steps (default: 20)
- `--wandb_project`: Weights & Biases project name
- `--wandb_name`: Weights & Biases run name
- `--log_path`: Local log directory

**Other:**
- `--dataset_source`: "huggingface" or "local" (default: "huggingface")
- `--verbose`: Enable verbose logging

## Reward Structure

The environment uses a multi-stage reward:

1. **Doesn't compile**: `reward = 0.0`
2. **Compiles but incorrect**: `reward = 0.1`
3. **Correct**: `reward = 1000.0 / runtime_us`

This incentivizes:
- Getting code to compile
- Maintaining correctness
- Optimizing for performance (faster = higher reward)

## Example Output

```
================================================================================
KernelBench RL Training
================================================================================
Model: Qwen/Qwen3-4B-Instruct-2507
Level: 1
Backend: triton
GPU: L40S
Precision: fp32
Group size: 4
Batch size: 2
Learning rate: 5e-05
Max tokens: 2048
Log path: /tmp/kernelbench-rl/...
================================================================================

[Training] Step 1/100
[KernelBench] Problem: matmul_simple
[KernelBench] Evaluating kernel on L40S...
[KernelBench] Result - Compiled: 1.0, Correct: 1.0, Runtime: 234.5 us, Speedup: 2.3x, Reward: 4.26
[Training] Metrics: {"compiled": 1.0, "correct": 1.0, "runtime_us": 234.5, "speedup": 2.3}
...
```

## Monitoring

### Local Logs

Training logs are saved to the log directory (default: `/tmp/kernelbench-rl/`):
- `metrics.json`: Training metrics per step
- `checkpoints/`: Model checkpoints
- `config.json`: Training configuration

### Weights & Biases

To enable W&B logging:

```bash
python train.py --wandb_project my-project --wandb_name my-run
```

## Troubleshooting

### Modal Issues

If you encounter Modal authentication errors:
```bash
modal token new
```

### Memory Issues

Reduce batch size or group size:
```bash
python train.py --batch_size 1 --group_size 2
```

### GPU Availability

Try different GPU types if one is unavailable:
```bash
python train.py --gpu A100
```

## Advanced Usage

### Custom Problem Descriptions

You can customize problem descriptions in `kernelbench_env.py`:

```python
def _get_problem_description(self, row: dict) -> str:
    # Add custom logic here
    return "Your custom description"
```

### Custom Reward Functions

Modify the reward function in `kernelbench_env.py`:

```python
def _compute_reward(self, eval_result: dict) -> float:
    # Customize reward calculation
    # e.g., add bonus for specific speedup thresholds
    ...
```

## References

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Modal Documentation](https://modal.com/docs)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)

## Citation

If you use this code, please cite:

```bibtex
@software{kernelbench_rl_tinker,
  title = {KernelBench RL Environment with Tinker},
  year = {2025},
}
```
# kernelbench-rl-env
