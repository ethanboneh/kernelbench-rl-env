# W&B Logging for KernelBench RL

## What's Now Logged

The environment now properly exports metrics to W&B through Tinker's training loop.

### Environment Metrics (Per Episode)

These metrics are computed for each kernel evaluation and aggregated by Tinker:

| Metric | Description | Range | Good Value |
|--------|-------------|-------|------------|
| `env/reward` | Reward for this kernel | 0.0 - ~100,000 | Higher is better |
| `env/compiled` | Did the kernel compile? | 0.0 or 1.0 | 1.0 |
| `env/correct` | Is output correct? | 0.0 or 1.0 | 1.0 |
| `env/runtime_us` | Kernel execution time (μs) | > 0 | Lower is better |
| `env/ref_runtime_us` | Reference runtime (μs) | > 0 | Baseline |
| `env/speedup` | Performance improvement | > 0 | > 1.0 |

### Aggregated Metrics (Automatic)

Tinker automatically aggregates these across batches:

- `train/env/reward/mean` - Average reward per batch
- `train/env/reward/total` - Total cumulative reward
- `train/env/compiled/mean` - Compilation success rate
- `train/env/correct/mean` - Correctness rate
- `train/env/speedup/mean` - Average speedup
- `train/env/runtime_us/mean` - Average runtime

And similarly for test set:
- `test/env/reward/mean`
- `test/env/compiled/mean`
- etc.

### Logging Tags

Each environment group is tagged with:
- `kernelbench` - General tag
- Backend (e.g., `triton`, `cuda`)
- GPU type (e.g., `L40S`, `H100`)
- Problem name (e.g., `100_HingeLoss`)

This allows you to filter and aggregate metrics by:
- Backend type
- GPU architecture
- Specific problems

### Training Loop Metrics (From Tinker)

In addition to environment metrics, Tinker logs:

- `train/loss` - RL training loss (PPO/GRPO)
- `train/learning_rate` - Current learning rate
- `train/kl_divergence` - KL divergence from reference policy
- `train/entropy` - Policy entropy
- `train/clipfrac` - Fraction of clipped gradients
- `step` - Global training step
- `episode` - Episode number

## Example W&B Dashboard

After training, your W&B dashboard will show:

### Main Charts
1. **Reward Over Time**
   - `train/env/reward/mean` - Should increase as agent learns
   - `test/env/reward/mean` - Generalization performance

2. **Success Rates**
   - `train/env/compiled/mean` - Should approach 1.0
   - `train/env/correct/mean` - Should approach 1.0

3. **Performance Improvement**
   - `train/env/speedup/mean` - Should increase (> 1.0)
   - `train/env/runtime_us/mean` - Should decrease

4. **Training Loss**
   - `train/loss` - Should decrease and stabilize

### Filtered Views

You can create custom views:

**By Backend:**
```
Filter: tag = "triton"
Charts: env/speedup/mean, env/correct/mean
```

**By GPU:**
```
Filter: tag = "H100"
Charts: env/runtime_us/mean
```

**By Problem:**
```
Filter: tag contains "HingeLoss"
Charts: env/reward/mean
```

## Running with W&B

```bash
# Enable W&B logging
python train.py \
  level=1 \
  batch_size=2 \
  group_size=4 \
  wandb_project=kernelbench-rl \
  wandb_name=triton-l40s-run1
```

## Interpreting Results

### Healthy Training
- ✅ `env/compiled/mean` > 0.8 (most kernels compile)
- ✅ `env/correct/mean` > 0.7 (most are functionally correct)
- ✅ `env/speedup/mean` increasing (getting faster)
- ✅ `env/reward/mean` increasing (overall improvement)
- ✅ `train/loss` decreasing and stabilizing

### Warning Signs
- ⚠️ `env/compiled/mean` < 0.5 (too many compile errors)
- ⚠️ `env/correct/mean` decreasing (losing correctness)
- ⚠️ `env/speedup/mean` < 1.0 (slower than reference)
- ⚠️ `train/loss` increasing or unstable

### Debugging

If metrics aren't showing up:

1. **Check W&B is initialized:**
   ```bash
   wandb login
   wandb online  # Make sure not in offline mode
   ```

2. **Verify metrics in logs:**
   ```bash
   tail -f /tmp/kernelbench-rl/*/logs.txt
   ```

3. **Check Tinker service:**
   - Make sure TINKER_API_KEY is set
   - Verify network connectivity

4. **Look for metric names:**
   - Metrics should have `env/` prefix
   - Tinker adds `train/` or `test/` prefix
   - Final names: `train/env/reward/mean`

## Custom Metrics

To add custom metrics, edit `kernelbench_env.py`:

```python
# In _compute_reward or step function
metrics = {
    "env/compiled": ...,
    "env/correct": ...,
    # Add your custom metric
    "env/custom_score": your_calculation,
}
```

These will automatically appear in W&B!

## Summary

The environment now exports **6 core metrics** per episode:
1. `env/reward` - Main optimization signal
2. `env/compiled` - Compilation success
3. `env/correct` - Correctness
4. `env/runtime_us` - Execution time
5. `env/ref_runtime_us` - Baseline time
6. `env/speedup` - Performance ratio

All aggregated automatically by Tinker and logged to W&B with proper prefixes and tags.
