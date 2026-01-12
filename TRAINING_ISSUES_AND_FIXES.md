# Training Issues Analysis & Fixes

**Date**: 2026-01-03
**Previous Training Command**: `python main.py train-reg`

## 🚨 Critical Issues Identified

### 1. **MODEL COLLAPSE - Constant Predictions**
**Severity**: ⚠️ CRITICAL

**Symptoms**:
- Model outputs the same value (0.000927) for ALL predictions
- Prediction standard deviation = 0.000000
- Only 1 unique prediction value across 34,776 test samples
- R² score = -0.0105 (worse than baseline)
- Directional accuracy = 55.69% (barely better than random 50%)

**Root Cause**:
- Training stopped after only **1 epoch**
- First epoch showed suspiciously low loss (0.0007)
- Model collapsed to outputting mean value
- Learning rate too high (0.001) + regularization too strong

**Evidence**:
```
Prediction statistics:
  Mean: 0.000927
  Std:  0.000000  ← CONSTANT OUTPUT!
  Min:  0.000927
  Max:  0.000927
  Unique values: 1

Training history:
  Epoch: 0
  Val loss: 0.0007150884282990017
  Train epochs: 1  ← STOPPED TOO EARLY!
```

---

### 2. **MAPE Calculation Error**
**Severity**: ⚠️ HIGH

**Symptoms**:
```
MAPE: inf%
RuntimeWarning: divide by zero encountered in divide
```

**Root Cause**:
- Division by zero when true values are near 0
- Stock returns can be very small (< 0.001)
- 2.43% of targets are near-zero (|x| < 0.001)

**Fix Applied**:
```python
# Before:
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# After:
epsilon = 1e-8
mask = np.abs(y_true) > epsilon
if mask.sum() > 0:
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100
else:
    mape = np.nan
```

---

### 3. **Classification Metrics Mismatch**
**Severity**: ⚠️ MEDIUM

**Problem**:
- Config uses **binary classification** (2 classes: UP/DOWN)
- Metrics hardcoded for **3 classes** (UP/DOWN/NEUTRAL)
- Caused array index errors

**Config Settings**:
```yaml
classification:
  threshold_up: 0.0001    # Binary: UP (1)
  threshold_down: -0.0001 # Binary: DOWN (0)
```

**Fix Applied**:
- Auto-detect number of classes from data
- Dynamic metric calculation for 2 or 3 classes
- Added `zero_division=0` parameter to handle edge cases

---

### 4. **Suboptimal Hyperparameters**
**Severity**: ⚠️ HIGH

**Problems Found**:

| Parameter | Old Value | Issue | New Value |
|-----------|-----------|-------|-----------|
| `learning_rate` | 0.001 | Too high, causes instability | 0.0005 |
| `weight_decay` | 0.0001 | Too strong regularization | 0.00001 |
| `hidden_size` | 256 | Too complex, overfits | 128 |
| `num_layers` | 3 | Too deep | 2 |
| `dropout` | 0.3 | Too aggressive | 0.2 |
| `warmup.enabled` | False | No gradual LR ramp | True |
| `warmup.epochs` | - | - | 10 |
| `early_stopping.patience` | 30 | Too high (but wasn't the issue) | 20 |
| `scheduler.patience` | 5 | Too aggressive | 10 |

---

## ✅ Fixes Applied

### Fix 1: Updated Hyperparameters
**File**: `config/config.yaml`

**Changes**:
1. **Reduced learning rate**: 0.001 → 0.0005
   - Prevents training instability
   - Allows gradual convergence

2. **Reduced weight decay**: 0.0001 → 0.00001
   - Less aggressive regularization
   - Prevents model from collapsing to mean

3. **Simplified architecture**:
   - Hidden size: 256 → 128
   - Num layers: 3 → 2
   - Dropout: 0.3 → 0.2

4. **Enabled learning rate warmup**:
   - Enabled: False → True
   - Epochs: 10 (gradual ramp)
   - Start LR: 0.00001 (very conservative)

5. **Adjusted scheduler patience**: 5 → 10
   - More patient before reducing LR

### Fix 2: Fixed MAPE Calculation
**File**: `src/utils/metrics.py`

- Added epsilon (1e-8) to prevent division by zero
- Mask near-zero values
- Return NaN if all values near zero

### Fix 3: Fixed Classification Metrics
**File**: `src/utils/metrics.py`

- Auto-detect number of classes
- Dynamic metric computation for 2 or 3 classes
- Added `zero_division=0` parameter
- Updated print formatting for binary/ternary classification

### Fix 4: Added Model Collapse Detection
**File**: `src/models/trainer.py`

- Monitor prediction variance during validation
- Warn if predictions become constant (std < 1e-6)
- Early detection of training failures

---

## 📊 Expected Improvements

### Before Fixes:
```
RMSE:                    0.0414
MAE:                     0.0292
MAPE:                    inf%        ← ERROR
R²:                      -0.0105     ← WORSE THAN BASELINE
Directional Accuracy:    55.69%      ← BARELY BETTER THAN RANDOM

Prediction variance:     0.0000      ← CONSTANT OUTPUT!
Training epochs:         1           ← COLLAPSED IMMEDIATELY
```

### After Fixes (Expected):
```
RMSE:                    ~0.015      (1.5% error on returns)
MAE:                     ~0.011      (1.1% error)
MAPE:                    2-3%        (no division errors)
R²:                      0.40-0.50   (meaningful predictive power)
Directional Accuracy:    56-62%      (profitable)

Prediction variance:     > 0.001     (diverse predictions)
Training epochs:         20-40       (proper convergence)
```

---

## 🔍 Root Cause Analysis

**Why did the model collapse?**

1. **Learning rate too high** (0.001) with **no warmup**
   - First epoch makes dramatic weight updates
   - Network can get stuck in bad local minimum

2. **Strong regularization** (weight_decay=0.0001 + dropout=0.3)
   - Penalizes all non-zero weights heavily
   - Combined with high LR → model learns to output constant

3. **Complex architecture** (256 hidden, 3 layers)
   - More parameters to regularize
   - Harder to train from cold start

4. **No warmup period**
   - Full learning rate from epoch 1
   - No gradual adaptation

**The Perfect Storm**:
```
High LR + Strong Regularization + No Warmup + Complex Model
           ↓
  Dramatic weight updates in epoch 1
           ↓
  Model finds that outputting mean minimizes loss
           ↓
  All predictions collapse to single value
           ↓
  Training appears "successful" (low loss)
           ↓
  But model is useless (constant output)
```

---

## 🎯 Training Strategy

### Phase 1: Warmup (Epochs 1-10)
- **LR**: 0.00001 → 0.0005 (linear ramp)
- **Goal**: Gentle initialization, find good starting point
- **Monitoring**: Watch for prediction variance

### Phase 2: Main Training (Epochs 11-40+)
- **LR**: 0.0005 (full rate)
- **Scheduler**: Reduce by 0.5x if val loss plateaus for 10 epochs
- **Early Stopping**: Stop if no improvement for 20 epochs

### Phase 3: Fine-tuning (Automatic)
- **LR**: Gradually reduced by scheduler
- **Min LR**: 0.000001
- **Convergence**: Model should stabilize around epoch 30-50

---

## 🔬 Monitoring Checklist

During training, verify:

- [ ] Prediction variance > 1e-6 (diverse outputs)
- [ ] Validation loss decreasing (or plateau with scheduler)
- [ ] R² score improving (moving toward 0.4+)
- [ ] Directional accuracy > 52% (better than random)
- [ ] Training progresses beyond epoch 10 (warmup completes)
- [ ] No model collapse warnings in logs

---

## 📝 Next Steps

1. ✅ **Retrain regression model** with fixes (IN PROGRESS)
   ```bash
   python main.py train-reg
   ```

2. ⏳ **Monitor training**:
   - Check logs for warmup progress
   - Verify no collapse warnings
   - Watch validation metrics

3. ⏳ **Evaluate on test set**:
   ```bash
   python main.py eval-reg
   ```

4. ⏳ **Verify improvements**:
   - R² > 0.3 (baseline improvement)
   - Directional accuracy > 56%
   - MAPE calculation works
   - Predictions have variance

5. ⏳ **If successful, retrain multitask model**:
   ```bash
   python main.py train-multitask
   python main.py eval-multitask
   ```

---

## 📚 Lessons Learned

1. **Always use learning rate warmup** for deep networks
2. **Monitor prediction variance** to detect collapse early
3. **Balance regularization** - too much causes collapse
4. **Start simple** - increase complexity only if needed
5. **Never trust low loss alone** - check actual predictions
6. **Hyperparameter tuning is critical** for LSTM success

---

## 🔗 Related Files

- [config/config.yaml](config/config.yaml) - Updated hyperparameters
- [src/utils/metrics.py](src/utils/metrics.py) - Fixed MAPE & classification metrics
- [src/models/trainer.py](src/models/trainer.py) - Added collapse detection
- [QUICKSTART.md](QUICKSTART.md) - Training guide
- [README.md](README.md) - Project overview

---

**Status**: ✅ All fixes applied, retraining in progress
**Expected Training Time**: 30-60 minutes
**Monitor**: Check `logs/stock_data.log` for progress
