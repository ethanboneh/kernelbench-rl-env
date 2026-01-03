# Getting Started with KernelBench RL

Quick guide to get up and running with the KernelBench RL environment.

## Prerequisites

✓ You already have:
- `.env` file with `TINKER_API_KEY`
- `KernelBench/` directory
- `tinker-cookbook/` directory

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install tinker modal datasets transformers torch chz wandb
```

### 2. Set Up Modal

Authenticate with Modal (for cloud GPU evaluation):

```bash
modal token new
```

### 3. Test the Environment

Run the test script to verify everything is set up correctly:

```bash
python test_env.py
```

This will:
- Load a simple test problem
- Create the RL environment
- Test the dataset builder
- Verify your Tinker API key works

### 4. Run Your First Training

Start training on Level 1 problems:

```bash
python train.py --level 1
```

Or use the quickstart script:

```bash
./quickstart.sh
```

## Quick Examples

### Minimal Training Run

```bash
python train.py --level 1 --batch_size 1 --group_size 2
```

### Production Training

```bash
python train.py \
  --level 2 \
  --gpu H100 \
  --backend triton \
  --batch_size 4 \
  --group_size 8 \
  --learning_rate 5e-5 \
  --num_epochs 20 \
  --wandb_project kernelbench-rl \
  --log_path ./logs/production-run
```

### Different Backends

Train with CUDA kernels instead of Triton:
```bash
python train.py --level 1 --backend cuda
```

Train with TileLang (requires fp16):
```bash
python train.py --level 1 --backend tilelang --precision fp16
```

## Understanding the Output

During training, you'll see:

```
[Training] Step 1/100
[KernelBench] Problem: matmul_simple
[KernelBench] Evaluating kernel on L40S...
[KernelBench] Result - Compiled: 1.0, Correct: 1.0, Runtime: 234.5 us, Speedup: 2.3x, Reward: 4.26
```

**Metrics:**
- **Compiled**: 1.0 if code compiles, 0.0 otherwise
- **Correct**: 1.0 if output matches reference, 0.0 otherwise
- **Runtime**: Execution time in microseconds
- **Speedup**: Speedup over reference implementation
- **Reward**: Final reward (higher is better)

## Common Issues

### "TINKER_API_KEY not found"

Make sure `.env` file contains:
```
export TINKER_API_KEY=your_key_here
```

Then source it:
```bash
source .env
```

### Modal Authentication Errors

Re-authenticate:
```bash
modal token new
```

### Out of Memory

Reduce batch size and group size:
```bash
python train.py --batch_size 1 --group_size 2
```

### GPU Not Available

Try a different GPU type:
```bash
python train.py --gpu A100
# or
python train.py --gpu T4
```

## Next Steps

1. **Monitor Training**: Add W&B logging
   ```bash
   python train.py --wandb_project my-project
   ```

2. **Experiment with Hyperparameters**: Try different learning rates, batch sizes, etc.

3. **Advanced Levels**: Move to harder problems
   ```bash
   python train.py --level 2
   python train.py --level 3
   python train.py --level 4
   ```

4. **Custom Rewards**: Edit `kernelbench_env.py` to customize the reward function

5. **Different Models**: Try other base models
   ```bash
   python train.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
   ```

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Training Loop                      │
│                 (Tinker RL Train)                    │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────┐
│              KernelBench Environment                 │
│  ┌─────────────────────────────────────────────┐   │
│  │ 1. Agent receives reference implementation  │   │
│  │ 2. Agent generates optimized kernel         │   │
│  │ 3. Environment evaluates on Modal GPU       │   │
│  │ 4. Return reward (compile + correct + perf) │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────┐
│               Modal GPU Evaluator                    │
│  ┌─────────────────────────────────────────────┐   │
│  │ 1. Compile kernel                           │   │
│  │ 2. Check correctness vs reference           │   │
│  │ 3. Measure performance (runtime)            │   │
│  │ 4. Return metrics                           │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
          ┌────────────────┐
          │  KernelBench   │
          │   Evaluation   │
          └────────────────┘
```

## Files Overview

- **`kernelbench_env.py`**: Core RL environment (Env, Dataset, Builders)
- **`modal_evaluator.py`**: Modal integration for cloud GPU eval
- **`train.py`**: Training script with CLI args
- **`test_env.py`**: Test script to verify setup
- **`requirements.txt`**: Python dependencies
- **`README.md`**: Full documentation
- **`.env`**: API keys (already present)

## Resources

- [Full README](README.md) - Complete documentation
- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/) - Tinker API reference
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) - Benchmark details
- [Modal Docs](https://modal.com/docs) - Modal platform docs

## Support

If you run into issues:

1. Check the [README](README.md) for troubleshooting
2. Run `python test_env.py` to diagnose issues
3. Verify API keys are set correctly
4. Check Modal GPU availability

Happy training! 🚀
