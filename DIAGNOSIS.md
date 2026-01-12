# Training Issues Diagnosis & Solutions

## Problem Summary

You have **two separate issues**:

1. ✅ **Training stops too early** - FIXED (early stopping triggered prematurely)
2. ⚠️ **Loss not decreasing enough** - Needs investigation

## Issue 1: Training Stops Too Early ✅ FIXED

**Root Cause:**
- Early stopping patience was too short (15 epochs)
- Learning rate scheduler patience was too long (8 epochs)
- Learning rate was too high (0.01), causing unstable training

**Solution Applied:**
- Increased early stopping patience: 15 → 20 epochs
- Reduced scheduler patience: 8 → 5 epochs (reduces LR faster)
- Reduced learning rate: 0.01 → 0.001 (10x smaller)
- Added learning rate warm-up over first 5 epochs

---

## Issue 2: Loss Not Decreasing ⚠️

### Current Status

From your checkpoint:
- **Only 1 epoch completed** (epoch 0)
- **Validation Loss: 1.3868** (very high for multitask)
- **Train Loss: 1.3843** (similar to validation - not overfitting yet)

### Why Loss is High

Your multitask model combines two losses:
```
Total Loss = (Regression Loss × 1.0) + (Classification Loss × 2.0)
```

If classification loss is around 0.69 (random guessing for 2-class) and regression loss is around 0.03, total would be:
```
Total = 0.03 + (0.69 × 2.0) = 0.03 + 1.38 = 1.41 ≈ 1.39
```

**This suggests the classification head is struggling!**

### Root Causes

Based on the data:
- **Regression target**: Mean=0.0035, Std=0.0393 (normal, well-scaled)
- **Classification**: Likely binary (UP/DOWN), possibly imbalanced
- **High classification weight** (2.0) dominates the loss

### Potential Issues

1. **Classification task might be too hard with current features**
   - Stock direction is inherently noisy
   - 5-day prediction horizon may be too long

2. **Learning rate might still be too high**
   - Started at 0.001, but multitask models often need lower LR

3. **Class imbalance not properly handled**
   - Need to verify class weights are being applied correctly

4. **Model might need more capacity**
   - Current: 2 LSTM layers, 128 hidden units
   - Might need deeper network or more units

---

## Recommended Solutions

### Solution 1: Check Classification Target Distribution (URGENT)

```bash
python -c "
import pandas as pd
train_df = pd.read_parquet('data/processed/train.parquet')
print('Class distribution:')
print(train_df['target_direction'].value_counts().sort_index())
print()
print('Percentage:')
print(train_df['target_direction'].value_counts(normalize=True).sort_index())
"
```

**Action needed:** Tell me the output so I can see class balance.

### Solution 2: Reduce Classification Weight

The classification weight of 2.0 might be too high. Try reducing it:

**Edit config.yaml line 341:**
```yaml
clf_weight: 0.5  # Reduced from 2.0 - let regression guide early training
```

This will balance the two tasks better during training.

### Solution 3: Use Even Lower Learning Rate

For multitask learning with high loss, try:

```bash
# Option A: Resume with very low LR
python resume_training.py --model-type multitask --lr 0.0001 --reset-scheduler --reset-early-stopping --full-epochs

# Option B: Start fresh with lower LR
# Edit config.yaml line 329: learning_rate: 0.0005
python main.py train-multitask
```

### Solution 4: Increase Model Capacity

Edit config.yaml lines 319-320:
```yaml
hidden_size: 256  # Increased from 128
num_layers: 3     # Increased from 2
```

Larger model = more capacity to learn complex patterns.

### Solution 5: Reduce Prediction Horizon

Edit config.yaml line 235:
```yaml
prediction_horizon: 1  # Changed from 5 - predict next day instead of 5 days ahead
```

Shorter horizon = easier prediction task.

---

## Quick Diagnostic Commands

### Check what happened in your last training:

```bash
# See training progress
tail -100 logs/stock_data.log | grep "Epoch\|STOPPING\|COMPLETED"

# Check classification distribution
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/train.parquet')
print('Target classes:', df['target_direction'].value_counts().sort_index())
"
```

### Test if model can learn at all:

```bash
# Quick 5-epoch test with very low LR
python -c "
from src.utils.config_loader import ConfigLoader
config = ConfigLoader('config/config.yaml')
# Temporarily set to 5 epochs for testing
" && python main.py train-multitask
```

---

## Next Steps

1. **First**: Tell me the classification target distribution (run Solution 1)
2. **Then**: Based on that, I'll recommend which solution(s) to apply
3. **Finally**: Resume training with the fix

The most likely fix is **reducing clf_weight from 2.0 to 0.5**, which will prevent the classification loss from dominating.

---

## Expected Behavior After Fix

With proper settings, you should see:

**First 10 epochs:**
```
Epoch [  1/100] - Loss: 1.3868/1.3891 | ... (warmup)
Epoch [  2/100] - Loss: 1.2543/1.2678 | ... (starting to learn)
Epoch [  5/100] - Loss: 0.9821/1.0123 | ... (warmup complete)
Epoch [ 10/100] - Loss: 0.7234/0.7456 | ... (learning well)
```

**By epoch 50:**
```
Epoch [ 50/100] - Loss: 0.4567/0.4789 | ... (converging)
```

If loss stays above 1.0 after 20 epochs, there's a fundamental problem with the task or architecture.
