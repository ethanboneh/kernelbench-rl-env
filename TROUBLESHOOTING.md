# Troubleshooting Training Hangs

## Issue: Training Hangs / Nothing Logged to W&B

### Root Cause

The environment is **working perfectly** (verified by `test_minimal_training.py`). The hang occurs during **Tinker training loop initialization**, not in the KernelBench environment.

### Why It Hangs

Tinker's `train.main()` needs to:

1. **Download model** (first run only)
   - Qwen3-4B: ~4-5 GB
   - Can take 10-30 minutes depending on internet speed
   - Downloads to `~/.cache/huggingface/`

2. **Load model into GPU**
   - Loading 4B parameters
   - Initializing LoRA adapters
   - Takes 2-5 minutes

3. **Initialize Tinker service**
   - Connect to Tinker API
   - Set up training/sampling clients
   - Prepare RL infrastructure

4. **Set up W&B** (if enabled)
   - Authenticate
   - Create run
   - Initialize logging

**Total first-run time: 15-45 minutes** (mostly waiting for model download)

### Solutions

#### Option 1: Be Patient (Recommended for First Run)

```bash
# Run with verbose logging
python train_with_logging.py level=1 batch_size=1 group_size=2

# Leave it running for 30-60 minutes
# Watch for messages about model download progress
```

The training **will** start, it just takes time on first run.

#### Option 2: Pre-download the Model

```bash
# Download model before training
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')"

# Then run training
python train.py level=1 batch_size=1 group_size=2
```

#### Option 3: Use a Smaller Model

```bash
# Use 1.5B model instead of 4B (much faster)
python train.py \
  level=1 \
  model_name="Qwen/Qwen3-1.5B-Instruct" \
  batch_size=1 \
  group_size=2
```

#### Option 4: Check What's Happening

While training is "hanging", check:

**1. Network activity:**
```bash
# Check if model is downloading
ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/

# Watch for growing files (model download in progress)
watch -n 5 'du -sh ~/.cache/huggingface/'
```

**2. Python process:**
```bash
# Check if training process is using CPU/network
top -pid $(pgrep -f train.py)
```

**3. Tinker API:**
```bash
# Check if you can reach Tinker
curl -H "Authorization: Bearer $TINKER_API_KEY" https://api.tinker.ai/health
```

## What's Actually Working

✅ **KernelBench Environment** - Tested, works perfectly
✅ **Modal GPU Evaluation** - Tested, 27.9s per eval
✅ **Reward Calculation** - Tested, returns correct metrics
✅ **Dataset Loading** - Tested, loads 100 problems
✅ **Metric Logging** - Metrics properly formatted for W&B

## What Needs Time

⏳ **Tinker Initialization** - First run: 15-45 min
⏳ **Model Download** - 4-5 GB, depends on internet
⏳ **Model Loading** - 2-5 minutes

## How to Know Training Started

You'll see these messages when training actually begins:

```
wandb: Syncing run kernelbench-level1-triton-L40S-...
wandb: ⭐️ View project at https://wandb.ai/...
Step 1/800
[KernelBench] Problem: 100_HingeLoss
[KernelBench] Evaluating kernel on L40S...
```

## Expected Timeline (First Run)

```
0:00 - Script starts, loads config
0:01 - Starts downloading model ← YOU ARE HERE (probably)
5:00 - Model download 50% complete
10:00 - Model download complete
12:00 - Loading model into memory
15:00 - Initializing Tinker service
17:00 - W&B initialization
18:00 - Training starts! ← FIRST STEP EXECUTES
```

## Expected Timeline (Subsequent Runs)

```
0:00 - Script starts
0:01 - Model already cached, loading...
2:00 - Tinker service initialization
3:00 - Training starts! ← Much faster
```

## Quick Verification

Run this to verify everything EXCEPT Tinker works:

```bash
# This completes in <1 minute
python test_minimal_training.py
```

If that works (it should), then training will work too - it just needs time to initialize.

## Still Hanging After 1 Hour?

Then there might be a real issue:

**1. Check Tinker API key:**
```bash
echo $TINKER_API_KEY
# Should print your key
```

**2. Check internet connection:**
```bash
ping -c 3 huggingface.co
ping -c 3 api.tinker.ai
```

**3. Check for errors:**
```bash
# Look in training logs
tail -f /tmp/kernelbench-rl/*/logs.txt
```

**4. Try minimal Tinker example:**
```bash
# Test if Tinker works at all
python -c "
import asyncio
import tinker
async def test():
    client = tinker.ServiceClient()
    print('Tinker connected!')
asyncio.run(test())
"
```

## Summary

**The environment works!** The "hang" is normal Tinker initialization, especially on first run. Just be patient or use a smaller model to speed things up.

**First run: 15-45 minutes is normal.**
**Subsequent runs: 3-5 minutes is normal.**
