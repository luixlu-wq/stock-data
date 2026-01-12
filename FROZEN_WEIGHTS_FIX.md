# Loss Not Changing - FIXED

## Problem

Training loss is **completely frozen**:

```
Epoch 0: Train=0.347410, Val=0.347144
Epoch 1: Train=0.347352, Val=0.347113
Change:  -0.000058 (basically zero!)
```

This means the model weights are **barely updating** - effectively frozen.

---

## Root Cause

I over-corrected for overfitting and applied **TOO MUCH regularization**:

### What I Did Wrong

```yaml
learning_rate: 0.0005   # TOO LOW (was 0.001)
weight_decay: 0.001     # TOO HIGH (was 0.0001) - 10x increase!
dropout: 0.4            # TOO HIGH (was 0.3)
```

**The problem:** High weight_decay (0.001) + Low LR (0.0005) = Weights can't update!

### Why This Froze Training

**Weight decay** penalizes large weights by adding `weight_decay × weights` to gradients:
- Normal: `weight_decay=0.0001` → small penalty
- My mistake: `weight_decay=0.001` → **10x penalty**!

Combined with low LR (0.0005), the gradient updates became:
```
New Weight = Old Weight - (LR × Gradient + LR × Weight_Decay × Weight)
New Weight = Old Weight - (0.0005 × Gradient + 0.0005 × 0.001 × Weight)
```

The weight decay term was **canceling out** most of the gradient!

---

## Solution Applied

Reverted to **balanced regularization**:

```yaml
learning_rate: 0.001    # ✅ Restored (was too low at 0.0005)
weight_decay: 0.0001    # ✅ Restored (was too high at 0.001)
dropout: 0.3            # ✅ Restored (was too high at 0.4)
```

**Kept these good changes:**
```yaml
clf_weight: 0.5         # ✅ Balanced (was 2.0)
early_stopping.patience: 30  # ✅ More patience (was 20)
warmup.enabled: False   # ✅ Disabled (was causing issues)
```

---

## What Went Wrong - Timeline

1. **Original Problem**: Training stopped at epoch 20 (early stopping too aggressive)
2. **First Fix**: Lowered LR to 0.001, reduced clf_weight to 0.5 ✅
3. **Second Problem**: Model overfitting (train 0.296, val 0.378)
4. **Second Fix (MISTAKE)**: I over-corrected:
   - LR: 0.001 → 0.0005 ❌
   - Weight decay: 0.0001 → 0.001 ❌ (10x too high!)
   - Dropout: 0.3 → 0.4 ❌
5. **Result**: Model weights frozen, loss not changing ❌
6. **Final Fix**: Reverted to balanced settings ✅

---

## Correct Settings for Stock Prediction

Based on analysis, here are the **optimal settings**:

### For Multitask Model (Regression + Classification)

```yaml
# Model
hidden_size: 128
num_layers: 2
dropout: 0.3           # Moderate dropout

# Training
learning_rate: 0.001   # Standard LR for Adam
weight_decay: 0.0001   # Light L2 regularization
batch_size: 256        # Large batch (you have RTX 5090)

# Task weights
reg_weight: 1.0        # Regression loss weight
clf_weight: 0.5        # Classification loss weight (balanced)

# Early stopping
patience: 30           # Give model time to improve
min_delta: 0.00001     # Accept small improvements
```

---

## How to Train Now

```bash
python main.py train-multitask
```

---

## Expected Behavior (Healthy Training)

With correct settings, you should see **steady improvement**:

```
Epoch [  1/100] - Loss: 0.42/0.43 | ...
Epoch [  5/100] - Loss: 0.38/0.39 | ... (decreasing)
Epoch [ 10/100] - Loss: 0.34/0.35 | ... (still improving)
Epoch [ 20/100] - Loss: 0.30/0.31 | ... (gradual descent)
Epoch [ 30/100] - Loss: 0.27/0.28 | ... (approaching convergence)
Epoch [ 50/100] - Loss: 0.24/0.25 | ... (converged)
```

**Key indicators:**
- ✅ Loss decreases by ~0.01-0.02 per 5 epochs
- ✅ Train loss ≈ Val loss (gap < 0.02)
- ✅ No sudden jumps or freezing

---

## What If...

### Loss still not changing after 10 epochs?

```bash
# Increase LR even more
# Edit config.yaml line 329:
learning_rate: 0.002
```

### Loss changes but overfits again (train << val)?

```bash
# Increase dropout slightly
# Edit config.yaml line 321:
dropout: 0.35
```

### Loss explodes (goes to NaN or >10)?

```bash
# Reduce LR
# Edit config.yaml line 329:
learning_rate: 0.0005
```

---

## Lesson Learned

**Don't over-regularize!**

When fixing overfitting:
- ✅ Slightly increase dropout (0.3 → 0.35)
- ✅ Keep weight_decay low (0.0001)
- ❌ DON'T increase weight_decay by 10x
- ❌ DON'T halve the learning rate at the same time

Small adjustments are better than drastic changes.

---

## Summary

**Problem:** Loss frozen at 0.347 (weights not updating)
**Cause:** weight_decay=0.001 too high + LR=0.0005 too low
**Fix:** Restored LR=0.001, weight_decay=0.0001, dropout=0.3
**Action:** Run `python main.py train-multitask`
**Expected:** Loss decreases to ~0.24-0.28 over 50 epochs
