# Training Troubleshooting Guide

## Common Issue: Model Only Trained 3 Epochs

### Problem
Your multi-task model checkpoint shows only 3 epochs of training instead of the expected 20-100 epochs. The model produces:
- Constant regression outputs (same value for all inputs)
- Never predicts certain classes (e.g., UP class)
- Very poor performance (~33% accuracy)

### Root Cause
Training was **interrupted** before completion. Common causes:
1. Terminal window was closed
2. Pressed Ctrl+C to stop training
3. Computer went to sleep
4. Power loss or system crash
5. Process killed by system

### How to Diagnose
Check your model's training status:
```bash
python -c "
import torch
ckpt = torch.load('models/checkpoints/lstm_multitask_best.pth', map_location='cpu')
print(f'Epochs completed: {ckpt.get(\"epoch\", 0) + 1}')
print(f'Validation loss: {ckpt.get(\"val_loss\", \"N/A\")}')
if 'history' in ckpt:
    print(f'Total history length: {len(ckpt[\"history\"][\"train_loss\"])}')
"
```

**Healthy model**: 20-50 epochs, val_loss < 0.20
**Broken model**: <10 epochs, val_loss > 0.50

---

## Solution 1: Retrain from Scratch (Recommended)

### Step 1: Delete Old Checkpoint
```bash
# Windows
del models\checkpoints\lstm_multitask_best.pth

# Linux/Mac
rm models/checkpoints/lstm_multitask_best.pth
```

### Step 2: Start Fresh Training
```bash
python main.py train-multitask
```

### Step 3: Monitor Progress
Open a **second terminal** and watch the logs:
```bash
# Windows PowerShell
Get-Content logs\stock_data.log -Wait -Tail 20

# Windows CMD
powershell Get-Content logs\stock_data.log -Wait -Tail 20

# Linux/Mac
tail -f logs/stock_data.log
```

### Step 4: What to Expect

**First 10 epochs** (0-5 minutes):
```
Epoch [  1/100] (180.2s) [  1.0%] - Loss: 0.5400/0.5478 | Combined: 0.2013/0.1850 | Best: 0.5478
Epoch [  2/100] (178.5s) [  2.0%] - Loss: 0.5385/0.5463 | Combined: 0.2012/0.1825 | Best: 0.5463
Epoch [  3/100] (179.1s) [  3.0%] - Loss: 0.5380/0.5458 | Combined: 0.2012/0.1835 | Best: 0.5458
Epoch [  4/100] (180.3s) [  4.0%] - Loss: 0.4523/0.4612 | Combined: 0.1823/0.1654 | Best: 0.4612
Epoch [  5/100] (179.8s) [  5.0%] - Loss: 0.3845/0.4021 | Combined: 0.1621/0.1523 | Best: 0.4021
...
```

**After 10 epochs** (milestone):
```
📊 MILESTONE: 10 epochs completed
   Best val loss: 0.2845
   Epochs without improvement: 2/15
   Remaining: 90 epochs
```

**Good signs**:
- Loss is **decreasing** (0.54 → 0.38 → 0.28 → ...)
- Each epoch takes ~180 seconds (3 minutes) on RTX 5090
- "Best" value keeps improving

**Bad signs**:
- Loss stays around 0.54 for >10 epochs → Learning rate issue
- "Best" never changes → Model not improving

### Step 5: Let It Complete
**Do NOT interrupt!** Training will stop automatically when:
- Early stopping triggers (typical: 20-40 epochs)
- 100 epochs complete
- Whichever comes first

**Expected total time**: 45-90 minutes

---

## Solution 2: Resume from Checkpoint

If training was interrupted and you want to continue:

```bash
python resume_training.py --model-type multitask
```

This will:
1. Load your existing checkpoint
2. Resume from the next epoch
3. Train until completion

**Example output**:
```
RESUMING TRAINING (MULTITASK)
Loading checkpoint from: models\checkpoints\lstm_multitask_best.pth
Last completed epoch: 2
Best validation loss: 0.5458
Will resume from epoch: 3

Training status:
   Completed epochs: 3
   Remaining epochs: 97
   Total epochs: 100

🚀 Resuming training...
```

---

## Preventing Interruptions

### Windows

**1. Prevent Sleep**:
```powershell
# Run as Administrator - prevents sleep during training
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 30
```

**2. Use a Persistent Session**:
```powershell
# Option A: Run in background with output redirection
Start-Process python -ArgumentList "main.py","train-multitask" -RedirectStandardOutput "training_output.log" -WindowStyle Minimized

# Option B: Use Windows Task Scheduler for unattended training
```

**3. Create a Keep-Alive Script** (keep_training_alive.bat):
```batch
@echo off
:retry
python main.py train-multitask
if errorlevel 1 (
    echo Training failed, check logs/stock_data.log
    pause
) else (
    echo Training completed successfully!
)
```

### Linux/Mac

**Use `nohup` or `screen`**:
```bash
# Option 1: nohup (output to file)
nohup python main.py train-multitask > training.log 2>&1 &

# Option 2: screen (detachable session)
screen -S training
python main.py train-multitask
# Press Ctrl+A then D to detach
# screen -r training  # to reattach

# Option 3: tmux (modern alternative)
tmux new -s training
python main.py train-multitask
# Press Ctrl+B then D to detach
# tmux attach -t training  # to reattach
```

---

## Monitoring Training Progress

### Real-Time Log Monitoring

**Windows**:
```powershell
Get-Content logs\stock_data.log -Wait -Tail 20
```

**Linux/Mac**:
```bash
tail -f logs/stock_data.log
```

### Check Current Status

```bash
python -c "
import torch
from pathlib import Path

ckpt_path = Path('models/checkpoints/lstm_multitask_best.pth')
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location='cpu')
    epoch = ckpt.get('epoch', 0)
    val_loss = ckpt.get('val_loss', float('inf'))
    history = ckpt.get('history', {})

    print(f'Current Status:')
    print(f'  Epochs completed: {epoch + 1}')
    print(f'  Best val loss: {val_loss:.4f}')

    if 'train_loss' in history:
        losses = history['train_loss']
        print(f'  Total epochs in history: {len(losses)}')
        print(f'  Loss trend (last 5): {losses[-5:]}')

        if len(losses) > 10:
            improvement = (losses[0] - losses[-1]) / losses[0] * 100
            print(f'  Improvement: {improvement:.1f}%')
else:
    print('No checkpoint found - training not started yet')
"
```

---

## Expected Training Behavior

### Healthy Training Pattern

```
Epoch [  1/100] - Loss: 0.5400 → Model learning basic patterns
Epoch [ 10/100] - Loss: 0.2845 → Rapid improvement phase
Epoch [ 20/100] - Loss: 0.1823 → Refinement phase
Epoch [ 30/100] - Loss: 0.1512 → Convergence starting
Epoch [ 35/100] - ✋ Early stopping triggered
```

**Key indicators**:
- Loss decreases by 50-70% in first 10 epochs
- Improvement slows down after epoch 20
- Early stopping around epoch 25-40
- Final val_loss: 0.12-0.18

### Unhealthy Training Pattern

```
Epoch [  1/100] - Loss: 0.5400
Epoch [ 10/100] - Loss: 0.5392  ← Barely changed!
Epoch [ 20/100] - Loss: 0.5388  ← Still stuck!
```

**Problem**: Learning rate too low or data issue
**Solution**: Check preprocessing, verify GPU is being used

---

## Quick Diagnostic Checklist

Before starting training, verify:

- [ ] GPU is detected: `python main.py check-gpu`
- [ ] Data exists: `ls data/processed/train.parquet`
- [ ] Config is correct: `cat config/config.yaml | grep epochs`
- [ ] Enough disk space: >5GB free
- [ ] No other GPU processes running
- [ ] Computer won't sleep during training

During training, verify:
- [ ] Loss is decreasing
- [ ] Each epoch takes 2-4 minutes
- [ ] GPU utilization is high (use `nvidia-smi`)
- [ ] No error messages in logs
- [ ] "Best" value keeps updating

After training:
- [ ] Epoch count > 20
- [ ] Final val_loss < 0.20
- [ ] Model file > 1MB
- [ ] Training log shows completion

---

## Testing the Trained Model

After training completes, verify it works:

```bash
python debug_multitask.py
```

**Expected output**:
```
Classification logits statistics:
DOWN (class 0) logits  - mean: 0.1234, std: 0.5678  ← High variance (good!)
UP (class 1) logits    - mean: -0.0234, std: 0.6123 ← High variance (good!)
NEUTRAL (class 2) logits - mean: -0.1567, std: 0.4892 ← High variance (good!)

Prediction distribution:
DOWN (0): 35 (35.0%)   ← Balanced predictions
UP (1): 38 (38.0%)     ← Predicting all classes
NEUTRAL (2): 27 (27.0%)

Regression outputs:
Mean: 0.0012, Std: 0.0156  ← High variance (good!)
```

**Bad signs**:
- Logits all have std < 0.1 (model collapsed)
- One class never predicted
- Regression std = 0.0 (constant output)

---

## When to Delete and Retrain

Delete checkpoint and retrain if:
1. Epoch count < 10
2. Val loss > 0.50 after 20 epochs
3. Model never predicts certain classes
4. Regression output is constant
5. Training was interrupted in first 10 epochs

**How to delete**:
```bash
# Windows
del models\checkpoints\lstm_multitask_best.pth
python main.py train-multitask

# Linux/Mac
rm models/checkpoints/lstm_multitask_best.pth
python main.py train-multitask
```

---

## Summary

**Problem**: Model only trained 3 epochs → produces garbage predictions

**Cause**: Training was interrupted (Ctrl+C, window closed, etc.)

**Solution**:
1. Delete checkpoint
2. Retrain properly **without interruption**
3. Monitor progress in second terminal
4. Wait for completion (45-90 minutes)

**Prevention**:
- Use `nohup`, `screen`, or `tmux` on Linux/Mac
- Disable sleep mode on Windows
- Monitor in separate terminal
- Don't close the training window!

**Expected result after proper training**:
- 20-40 epochs completed
- Val loss: 0.12-0.18
- Classification accuracy: 45-52%
- Regression directional accuracy: 56-62%
- All 3 classes predicted
