# Training Evaluation Report

## Executive Summary

**Status:** ⚠️ **Model Has Major Issues**

The model completed 32 epochs but achieved **poor performance**, especially on classification:
- **Regression**: Marginal performance (R² = -0.001, basically random)
- **Classification**: **Complete failure** (predicts only one class)

---

## Training Results

### Training Progress

```
Best validation loss: 0.3471 (epoch 2)
Final train loss:     0.3473
Final val loss:       0.3473
Total epochs:         32 (stopped early)
```

**Analysis:**
- ✅ No overfitting (train ≈ val)
- ❌ Loss stuck at 0.347 after epoch 2
- ❌ Model stopped learning after 30 consecutive epochs without improvement
- ❌ Loss of 0.347 is too high for effective prediction

---

## Test Set Performance

### Regression Metrics

```
RMSE:                    0.0412
MAE:                     0.0291
MAPE:                    inf (division by zero error)
R²:                      -0.001 ⚠️ WORSE than baseline!
Directional Accuracy:    55.69% (barely better than coin flip)
```

**Analysis:**
- R² = -0.001 means the model is **worse than just predicting the mean**
- RMSE of 0.0412 vs std of 0.0393 (from earlier) means high error
- Only 55.69% directional accuracy (should be >60% for useful predictions)

### Classification Metrics ⚠️ **CRITICAL FAILURE**

```
Accuracy: 44.31%

Per-Class Performance:
  DOWN (0):   Precision: 44.31%, Recall: 100.00%, F1: 61.41%
  UP (1):     Precision: 0.00%,   Recall: 0.00%,   F1: 0.00%
  NEUTRAL (2): Precision: 0.00%,   Recall: 0.00%,   F1: 0.00%

Confusion Matrix:
         DOWN    UP   NEUTRAL
DOWN     15409     0     0
UP       19367     0     0  ← Model NEVER predicts UP!
NEUTRAL     0     0     0  ← Model NEVER predicts NEUTRAL!
```

**Critical Issue:** The model **only predicts class 0 (DOWN)** for all samples!

**Why this happened:**
1. Class distribution in test set: DOWN=15409, UP=19367, NEUTRAL=0
2. Model learned to always predict DOWN (majority class strategy)
3. This gives 44.31% accuracy (15409 / 34776) but is completely useless

---

## Root Causes

### 1. Loss is Too High (0.347)

The multitask loss formula:
```
Total Loss = (Regression Loss × 1.0) + (Classification Loss × 0.5)
```

If classification loss is at CrossEntropy for always predicting DOWN:
```
CE = -log(0.44) ≈ 0.82 (for class 0 with 44% probability)
Total = 0.04 (reg) + 0.82 × 0.5 = 0.04 + 0.41 = 0.45
```

But we got 0.347, which means the model is doing slightly better than random but still very poor.

### 2. Model Capacity Insufficient

```yaml
hidden_size: 128
num_layers: 2
```

For 41 input features and complex time-series patterns, this might be too small.

### 3. Classification Task May Be Ill-Defined

**Issue with NEUTRAL class:**
- Training set had 3 classes (based on config: UP, DOWN, NEUTRAL)
- But test set has NEUTRAL=0 samples!
- This suggests the data preprocessing has issues

### 4. Learning Rate Still Not Optimal

With LR=0.001, the model stopped improving after epoch 2, suggesting:
- Either LR is too low (can't escape local minimum)
- Or the task is too hard with current architecture

---

## Recommended Solutions

### Solution 1: Binary Classification (RECOMMENDED)

The NEUTRAL class is problematic. Switch to **binary classification** (UP vs DOWN only):

**Edit config.yaml line 304-307:**

```yaml
# Classification labels - Binary classification for better performance
classification:
  threshold_up: 0.001    # Any gain > 0.1% = UP (class 1)
  threshold_down: -0.001 # Any loss < -0.1% = DOWN (class 0)
  # Binary classification: UP (1) or DOWN (0) - no NEUTRAL class
```

This removes the ambiguous NEUTRAL class and simplifies the task.

### Solution 2: Increase Model Capacity

**Edit config.yaml lines 319-320:**

```yaml
hidden_size: 256  # Increased from 128 (2x capacity)
num_layers: 3     # Increased from 2 (more depth)
```

Larger model = more capacity to learn complex patterns.

### Solution 3: Try Regression-Only Model

Since classification is failing completely, try **regression only** first:

```bash
python main.py train-reg
```

This will help isolate whether the issue is:
- A) The classification task itself (if regression works)
- B) The overall architecture (if regression also fails)

### Solution 4: Increase Learning Rate

Try a higher LR to escape the local minimum:

**Edit config.yaml line 329:**

```yaml
learning_rate: 0.002  # Increased from 0.001
```

### Solution 5: Reduce Prediction Horizon

5-day ahead prediction is very difficult. Try 1-day:

**Edit config.yaml line 235:**

```yaml
prediction_horizon: 1  # Changed from 5 - predict tomorrow instead of next week
```

---

## Action Plan

### Immediate Actions (Pick ONE to start)

**Option A: Fix Classification (Recommended)**
1. Edit config to binary classification (remove NEUTRAL)
2. Rerun preprocessing: `python main.py preprocess`
3. Train: `python main.py train-multitask`

**Option B: Regression Only (Diagnostic)**
1. Train regression model: `python main.py train-reg`
2. Evaluate: `python main.py eval-reg`
3. If this works, classification was the problem

**Option C: Bigger Model**
1. Edit config: `hidden_size: 256`, `num_layers: 3`
2. Train: `python main.py train-multitask`

### Expected Results After Fix

With binary classification + bigger model, you should see:

**Training:**
```
Epoch [ 20/100] - Loss: 0.25/0.26 | ... (better than 0.347!)
Epoch [ 50/100] - Loss: 0.20/0.22 | ...
```

**Evaluation:**
```
Regression:
  R²: 0.05-0.15 (positive = better than baseline)
  Directional Accuracy: 60-65%

Classification:
  Accuracy: 55-60%
  Both classes predicted (not just DOWN!)
```

---

## Current State Summary

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Training Loss** | 0.347 | <0.25 | ❌ Too high |
| **R² Score** | -0.001 | >0.05 | ❌ Worse than baseline |
| **Dir. Accuracy** | 55.69% | >60% | ⚠️ Barely useful |
| **Clf Accuracy** | 44.31% | >55% | ❌ Failed |
| **Classes Predicted** | 1/3 | 3/3 or 2/2 | ❌ Critical failure |

**Verdict:** Model needs significant changes before it's usable.

---

## Next Steps

1. **Decide:** Which solution to try first (I recommend binary classification)
2. **Implement:** Make the config changes
3. **Retrain:** Run `python main.py preprocess` then `python main.py train-multitask`
4. **Evaluate:** Check if performance improves

Let me know which approach you'd like to take!
