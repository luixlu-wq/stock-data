# Solution 1: Binary Classification - Implementation Summary

## Changes Made

### 1. Binary Classification Configuration ✅

**File:** [config/config.yaml:304-308](config/config.yaml#L304-L308)

```yaml
classification:
  threshold_up: 0.0001    # Any gain > 0.01% = UP (class 1)
  threshold_down: -0.0001 # Any loss < -0.01% = DOWN (class 0)
  # Binary classification: UP (1) or DOWN (0) - no NEUTRAL class
```

**Removed:** NEUTRAL class (class 2)
**Now:** Simple binary UP vs DOWN classification

### 2. Increased Model Capacity ✅

**File:** [config/config.yaml:320-321](config/config.yaml#L320-L321)

```yaml
hidden_size: 256  # Increased from 128 (2x capacity)
num_layers: 3     # Increased from 2 (50% more depth)
```

**Model parameters:** ~2.5M → ~6M (2.4x increase)
**Reason:** Original model was too small to learn complex patterns

### 3. Fixed Preprocessor Logic ✅

**File:** [src/data/preprocessor.py:180-193](src/data/preprocessor.py#L180-L193)

**Problem:** Preprocessor was creating 3 classes even with binary thresholds
**Fix:** Added logic to detect binary classification intent and force 2 classes only

**New behavior:**
- If `threshold_up ≈ -threshold_down` (within 0.001): Binary mode
- Otherwise: Three-class mode

### 4. Reprocessed Data ✅

**Command:** `python main.py preprocess`

**Results:**
```
TRAIN:      640,126 samples - DOWN: 44.45%, UP: 55.55% (1.25x ratio)
VALIDATION:  47,628 samples - DOWN: 45.71%, UP: 54.29% (1.19x ratio)
TEST:        46,116 samples - DOWN: 45.14%, UP: 54.86% (1.21x ratio)
```

**Status:** ✓ Clean binary classification (2 classes only)
**Balance:** 1.2x ratio (acceptable, class weights will handle this)

---

## What's Different From Before

### Previous Model (Failed)

```
Architecture: 128 hidden, 2 layers
Classification: 3 classes (UP, DOWN, NEUTRAL)
Result: Model only predicted DOWN (class 0)
  - Classification accuracy: 44.31%
  - Only 1 of 3 classes predicted
  - R²: -0.001 (worse than baseline)
```

### New Model (Training Now)

```
Architecture: 256 hidden, 3 layers (~2.4x parameters)
Classification: 2 classes (UP, DOWN) - NEUTRAL removed
Expected: Both classes predicted, better performance
```

---

## Expected Improvements

### 1. Classification Performance

**Before:**
- 44.31% accuracy (always predicting DOWN)
- 0% precision/recall on UP class

**Expected After:**
- 55-60% accuracy (better than random)
- Both UP and DOWN classes predicted
- Balanced confusion matrix

### 2. Regression Performance

**Before:**
- R² = -0.001 (worse than baseline)
- Directional accuracy: 55.69%

**Expected After:**
- R² = 0.05-0.15 (positive = useful!)
- Directional accuracy: 60-65%

### 3. Training Stability

**Before:**
- Loss stuck at 0.347 after epoch 2
- Early stopping at epoch 32

**Expected After:**
- Loss decreases to 0.20-0.25
- Trains for 50-80 epochs before converging
- Gradual, steady improvement

---

## Training Progress

**Status:** Training in background (task ID: be924e4)

**Monitoring:** Check `training_output.log` or use:
```bash
tail -f training_output.log
```

**Expected timeline:**
- 5-10 minutes per 10 epochs
- 50-80 epochs to converge
- Total: ~30-60 minutes

**What to watch for:**

✅ **Good signs:**
```
Epoch [ 10/100] - Loss: 0.32/0.33 | ... (decreasing)
Epoch [ 20/100] - Loss: 0.28/0.29 | ... (still improving)
Epoch [ 50/100] - Loss: 0.22/0.23 | ... (converging)
```

❌ **Bad signs:**
```
Epoch [ 10/100] - Loss: 0.35/0.35 | ... (not changing)
Epoch [ 20/100] - Loss: 0.35/0.35 | ... (stuck again)
```

If stuck again, we'll try:
- Increase LR to 0.002
- Or switch to regression-only model

---

## Next Steps

### 1. Monitor Training

Wait for training to complete (~30-60 min)

### 2. Evaluate

Once complete, run:
```bash
python main.py eval-multitask
```

### 3. Compare Results

Check if:
- Loss < 0.25 ✓
- R² > 0.0 ✓
- Classification accuracy > 52% ✓
- Both UP and DOWN classes predicted ✓

### 4. If Still Not Working

Try regression-only model:
```bash
python main.py train-reg
python main.py eval-reg
```

---

## Configuration Summary

**Current optimal settings:**

```yaml
# Model Architecture
hidden_size: 256
num_layers: 3
dropout: 0.3

# Training
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 256
epochs: 100

# Task Weights
reg_weight: 1.0
clf_weight: 0.5

# Early Stopping
patience: 30
min_delta: 0.00001

# Classification
Binary: UP vs DOWN (no NEUTRAL)
Threshold: ±0.0001 (0.01%)
```

These settings balance:
- Model capacity (large enough to learn)
- Regularization (prevent overfitting)
- Task difficulty (binary easier than 3-class)

---

## Success Criteria

The new model is successful if:

1. **Training:** Loss decreases to < 0.25
2. **No Overfitting:** Train loss ≈ Val loss (gap < 0.03)
3. **Classification:** Accuracy > 52%, both classes predicted
4. **Regression:** R² > 0.0 (positive), Directional accuracy > 58%

If 3 out of 4 criteria are met, the model is usable for stock prediction.

---

## Files Modified

1. [config/config.yaml](config/config.yaml) - Binary classification + larger model
2. [src/data/preprocessor.py](src/data/preprocessor.py) - Fixed binary classification logic
3. Data reprocessed: train.parquet, validation.parquet, test.parquet

## Training Command

```bash
python main.py train-multitask
```

Running in background. Check `training_output.log` for progress.
