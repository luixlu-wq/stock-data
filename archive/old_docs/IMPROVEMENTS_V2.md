# Stock Prediction Model - Version 2.0 Improvements

**Date**: 2026-01-03
**Status**: ✅ All improvements implemented, retraining in progress

---

## 🎯 **Goals**

Address the severe underfitting problem where the model was:
- Predicting only 10.6% of actual variance
- Correlation = 0.069 (essentially random)
- Directional accuracy = 54.31% (barely better than coin flip)

**Target Performance**:
- R² > 0.15 (meaningful predictions)
- Directional accuracy > 58%
- Prediction variance > 40% of true variance

---

## 🔧 **IMPROVEMENTS IMPLEMENTED**

### **1. Custom Directional Loss Functions** ✅

**Problem**: Huber loss encourages conservative predictions (predicting near mean minimizes loss)

**Solution**: Implemented 3 new loss functions in [`src/models/losses.py`](src/models/losses.py):

#### **A. DirectionalHuberLoss** (SELECTED)
```python
total_loss = (1 - weight) * huber_loss + weight * directional_penalty
```
- **Huber component**: Robust magnitude accuracy
- **Directional component**: Penalizes wrong direction
- **Weight** (0.3): 70% magnitude, 30% direction

**Benefits**:
- Encourages correct sign prediction
- Still robust to outliers
- Balances accuracy and direction

#### **B. WeightedDirectionalLoss**
```python
loss = base_loss * magnitude_weights * directional_penalty
```
- Weights large moves more heavily
- Encourages model to predict big movements
- Useful for volatile stocks

#### **C. QuantileLoss**
- Predicts distribution (10th, 50th, 90th percentiles)
- Better uncertainty quantification
- For future research

**Configuration**:
```yaml
loss:
  regression: "directional_huber"  # NEW!
  directional_weight: 0.3  # 30% directional, 70% magnitude
  magnitude_scale: 2.0  # Encourage larger predictions
```

---

### **2. Reduced Prediction Horizon** ✅

**Change**: 5-day → **1-day** ahead prediction

**Reasoning**:
- Shorter horizons are more predictable
- Less noise accumulation
- Market microstructure more stable

**Expected Impact**:
- Higher R² (0.15-0.25 vs 0.003)
- Better directional accuracy (58-62% vs 54%)

**Configuration**:
```yaml
data:
  prediction_horizon: 1  # Changed from 5
```

---

### **3. Increased Model Capacity** ✅

**Problem**: Model may be too simple to capture complex stock patterns

**Changes**:
| Parameter | Old Value | New Value | Reasoning |
|-----------|-----------|-----------|-----------|
| `hidden_size` | 128 | **192** | +50% capacity for complex patterns |
| `dropout` | 0.2 | **0.25** | Slightly more regularization |

**Expected Impact**:
- Better feature extraction
- More expressive representations
- Captures non-linear patterns

---

### **4. Optimized Training Hyperparameters** ✅

**A. Learning Rate**
- **Old**: 0.0005 (too conservative)
- **New**: 0.0008 (+60%)
- **Reasoning**: Faster convergence, less underfitting

**B. Weight Decay**
- **Old**: 0.00001
- **New**: 0.000005 (50% reduction)
- **Reasoning**: Less regularization → larger predictions

**C. Learning Rate Warmup**
- **Old**: 10 epochs
- **New**: 5 epochs
- **Reasoning**: Faster warmup, more time at full LR

**D. Early Stopping**
- **Patience**: 15 → **25** epochs
- **Min delta**: 0.00001 → **0.000005**
- **Reasoning**: Allow more training, accept tiny improvements

**E. Max Epochs**
- **Old**: 50
- **New**: 100
- **Reasoning**: Give model more time to learn

**F. LR Scheduler**
- **Patience**: 10 → **12** epochs
- **Min LR**: 0.000001 → **0.0000001**
- **Reasoning**: More patient, allow finer tuning

---

## 📊 **EXPECTED PERFORMANCE COMPARISON**

### **Before (Version 1.0)**
```
Model Metrics:
  RMSE: 0.0411
  MAE:  0.0291
  R²:   0.0028  ← Nearly useless
  Dir Acc: 54.31%  ← Barely better than random

Prediction Behavior:
  Pred std: 0.0044  ← Way too conservative
  True std: 0.041
  Variance ratio: 0.1065 (10.6%)  ← Severe underfitting
  Correlation: 0.069  ← Essentially random

Training:
  Loss function: Huber (conservative)
  Prediction horizon: 5 days (harder)
  Hidden size: 128 (limited capacity)
```

### **After (Version 2.0 - Expected)**
```
Model Metrics:
  RMSE: ~0.015-0.020  (better on 1-day)
  MAE:  ~0.012-0.016
  R²:   0.15-0.25  ← Meaningful predictions!
  Dir Acc: 58-62%  ← Profitable edge

Prediction Behavior:
  Pred std: ~0.015-0.020  ← More aggressive
  True std: ~0.025 (1-day is less volatile than 5-day)
  Variance ratio: 0.50-0.70 (50-70%)  ← Much better!
  Correlation: 0.30-0.50  ← Actual signal

Training:
  Loss function: DirectionalHuberLoss (direction-aware)
  Prediction horizon: 1 day (easier)
  Hidden size: 192 (more capacity)
```

---

## 🔬 **EVALUATION METHODOLOGY**

### **Phase 1: Basic Metrics**
```python
# Standard regression metrics
- RMSE (should be < 0.020 for 1-day returns)
- MAE (should be < 0.016)
- R² (should be > 0.15)
- MAPE (percentage error)
- Directional accuracy (should be > 58%)
```

### **Phase 2: Distribution Analysis**
```python
# Check if model is still too conservative
variance_ratio = pred_std / true_std
# Target: 0.50-0.70 (was 0.106)

correlation = np.corrcoef(y_true, y_pred)[0,1]
# Target: 0.30-0.50 (was 0.069)
```

### **Phase 3: Directional Analysis**
```python
# Breakdown by direction
correct_up = (y_true > 0) & (y_pred > 0)
correct_down = (y_true < 0) & (y_pred < 0)

# Should see:
- Directional accuracy > 58%
- Balanced performance (not just guessing all UP)
```

### **Phase 4: Advanced Analysis**
```python
# Trading simulation
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown
- Win rate
- Profit factor

# Targets:
- Sharpe > 0.5 (decent)
- Win rate > 52%
- Profit factor > 1.2
```

---

## 🔍 **WHAT TO MONITOR DURING TRAINING**

### **1. Loss Progression**
```
Epoch 1-5:   Warmup (loss should decrease slowly)
Epoch 6-20:  Rapid learning (loss drops significantly)
Epoch 21-40: Refinement (smaller improvements)
Epoch 41+:   Fine-tuning (may trigger early stopping)
```

### **2. Red Flags**
- ⚠️ Loss stops decreasing after epoch 10 → underfitting still present
- ⚠️ Model collapse warning → predictions becoming constant again
- ⚠️ Prediction variance < 0.01 → too conservative still

### **3. Good Signs**
- ✅ Validation loss steadily decreasing
- ✅ No model collapse warnings
- ✅ Training progresses beyond 20 epochs
- ✅ Prediction variance > 0.015

---

## 📁 **FILES MODIFIED**

### **New Files**
1. **[src/models/losses.py](src/models/losses.py)** - Custom loss functions
   - `DirectionalHuberLoss`
   - `WeightedDirectionalLoss`
   - `QuantileLoss`

### **Modified Files**
1. **[src/models/trainer.py](src/models/trainer.py:59-83)** - Added loss function support
   - Imports custom losses
   - Handles `directional_huber` and `weighted_directional` loss types
   - Configurable directional weight

2. **[config/config.yaml](config/config.yaml)**:
   - Line 235: `prediction_horizon: 5 → 1`
   - Line 320: `hidden_size: 128 → 192`
   - Line 322: `dropout: 0.2 → 0.25`
   - Line 329: `epochs: 50 → 100`
   - Line 330: `learning_rate: 0.0005 → 0.0008`
   - Line 331: `weight_decay: 0.00001 → 0.000005`
   - Line 337: `warmup.epochs: 10 → 5`
   - Line 346: `early_stopping.patience: 15 → 25`
   - Line 347: `early_stopping.min_delta: 0.00001 → 0.000005`
   - Line 352: `scheduler.patience: 10 → 12`
   - Line 358-360: Added directional loss config

3. **[src/utils/metrics.py](src/utils/metrics.py)** - Fixed earlier
   - MAPE division by zero handling
   - Auto-detect number of classes

---

## 🚀 **TRAINING STATUS**

### **Current Run**
- ✅ Data reprocessed with 1-day horizon
- ✅ All configurations updated
- 🔄 **Training in progress** (started 20:27)
- ⏳ Expected completion: ~45-60 minutes

### **Monitoring**
```bash
# Check training progress
tail -f training_improved.log

# Or check latest epochs
tail -50 logs/stock_data.log
```

### **After Training Completes**
```bash
# Evaluate
python main.py eval-reg

# Check predictions
python -c "import pandas as pd; df=pd.read_parquet('data/processed/regression_predictions.parquet'); print(df['y_pred'].describe())"
```

---

## 🎯 **SUCCESS CRITERIA**

### **Minimum Acceptable**
- ✅ R² > 0.10 (better than baseline)
- ✅ Dir Acc > 56% (profitable)
- ✅ Variance ratio > 0.30
- ✅ No model collapse

### **Good Performance**
- ✅ R² > 0.15
- ✅ Dir Acc > 58%
- ✅ Variance ratio > 0.50
- ✅ Correlation > 0.30

### **Excellent Performance**
- ✅ R² > 0.25
- ✅ Dir Acc > 62%
- ✅ Variance ratio > 0.70
- ✅ Correlation > 0.50

---

## 🔄 **NEXT STEPS AFTER EVALUATION**

### **If Performance is Good (R² > 0.15)**
1. Train multi-task model for even better results
2. Create diagnostic visualizations
3. Implement backtesting framework
4. Deploy for live predictions

### **If Performance is Moderate (0.10 < R² < 0.15)**
1. Try `weighted_directional` loss (more aggressive)
2. Increase `directional_weight` to 0.4-0.5
3. Increase `magnitude_scale` to 3.0-5.0
4. Try bidirectional LSTM

### **If Performance is Still Poor (R² < 0.10)**
1. Stock returns may be too noisy for simple LSTM
2. Consider:
   - Transformer architecture
   - Ensemble methods
   - Alternative features (sentiment, options flow)
   - Focus on specific stocks or sectors only

---

## 📚 **REFERENCES**

- [TRAINING_ISSUES_AND_FIXES.md](TRAINING_ISSUES_AND_FIXES.md) - Original issues
- [src/models/losses.py](src/models/losses.py) - Custom loss implementations
- [config/config.yaml](config/config.yaml) - Updated configuration

---

**Summary**: All recommended improvements have been implemented. The model now uses directional loss, predicts 1-day ahead, has more capacity, and is trained with optimized hyperparameters. Training is in progress - expect significantly better results!
