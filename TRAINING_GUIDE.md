# Model Training Guide - Fix for Plateau Issues

## What Was Fixed

Your model training was stuck at epoch 20 with no improvement in loss. Here's what has been improved:

### 1. Configuration Updates ([config.yaml:325-353](config/config.yaml#L325-L353))

**Changes:**
- **Learning Rate**: Reduced from `0.01` → `0.001` (10x smaller for more stable training)
- **Scheduler Patience**: Reduced from `8` → `5` epochs (reduces LR sooner when plateauing)
- **Early Stopping Patience**: Increased from `15` → `20` epochs (gives model more time to improve)
- **Added Learning Rate Warm-up**: Gradually increases LR from `0.0001` to `0.001` over first 5 epochs

**Why this helps:**
- The original LR of 0.01 was too aggressive, causing the optimizer to overshoot optimal weights
- Faster LR reduction helps escape plateaus
- Warm-up prevents early training instability

### 2. Enhanced Resume Training Script

The `resume_training.py` script now supports:

**New options:**
- `--lr`: Override learning rate when resuming
- `--reset-scheduler`: Reset the LR scheduler state
- `--reset-early-stopping`: Reset early stopping counter
- `--full-epochs`: Train for full 100 epochs (instead of remaining 80)

### 3. Learning Rate Warm-up in Trainer

The trainer now supports gradual learning rate warm-up:
- Starts at lower LR (0.0001)
- Linearly increases to target LR (0.001) over 5 epochs
- Prevents early training instability

## How to Use

### Option 1: Resume for Full 100 Epochs with Lower LR (Recommended)

```bash
# Resume from checkpoint but train for FULL 100 epochs with lower LR
python resume_training.py --model-type multitask --lr 0.0001 --reset-scheduler --reset-early-stopping --full-epochs
```

This will:
- Load your checkpoint weights from epoch 20
- Set LR to 0.0001 (even lower than new default)
- Reset scheduler and early stopping counters
- **Train for full 100 epochs** (not just remaining 80)

### Option 2: Resume for Remaining 80 Epochs Only

```bash
# Continue from epoch 20 and train for remaining 80 epochs
python resume_training.py --model-type multitask --lr 0.0001 --reset-scheduler --reset-early-stopping
```

This will train epochs 21-100 (80 epochs total).

### Option 3: Resume with New Default Settings

```bash
# Resume with updated config settings (LR=0.001)
python resume_training.py --model-type multitask --reset-early-stopping
```

### Option 4: Start Fresh Training with New Settings

```bash
# Start new training from scratch with improved hyperparameters
python scripts/train_model.py --model-type multitask
```

The new warm-up will:
- Epoch 1: LR = 0.0001
- Epoch 2: LR = 0.0003
- Epoch 3: LR = 0.0005
- Epoch 4: LR = 0.0007
- Epoch 5+: LR = 0.001 (then scheduler takes over)

## Understanding the Problem

Your training stopped because:

1. **High Learning Rate** (0.01) caused unstable gradients
2. **Scheduler Patience Too Long** (8 epochs) - took too long to reduce LR
3. **Loss Plateaued** - model couldn't escape local minimum

The fix:
- Lower base LR = more stable gradient descent
- Faster scheduler response = escapes plateaus quicker
- Warm-up = smooth start prevents early overfitting

## Monitoring Training

Watch for these improvements:

**Before (stuck):**
```
Epoch [ 15/100] - Loss: 0.5234/0.5234 | RMSE: 0.7234/0.7234 | Best: 0.5234
Epoch [ 16/100] - Loss: 0.5234/0.5234 | RMSE: 0.7234/0.7234 | Best: 0.5234
Epoch [ 17/100] - Loss: 0.5234/0.5234 | RMSE: 0.7234/0.7234 | Best: 0.5234
... (no improvement)
```

**After (improving):**
```
Epoch [ 21/100] - Loss: 0.5123/0.5123 | RMSE: 0.7156/0.7156 | Best: 0.5123 ✓
Epoch [ 22/100] - Loss: 0.5056/0.5045 | RMSE: 0.7112/0.7103 | Best: 0.5045 ✓
Epoch [ 23/100] - Loss: 0.4989/0.4978 | RMSE: 0.7063/0.7056 | Best: 0.4978 ✓
```

## Expected Results

With these changes, you should see:

- **Smoother training curves** (less oscillation)
- **Gradual loss reduction** (not stuck at same value)
- **Better validation performance** (lower overfitting)
- **Training completion** (reaches 100 epochs or early stopping with improvement)

## Next Steps

1. **Try Option 1** (resume with low LR) - fastest way to see if model can improve
2. **Monitor for 10-20 epochs** - should see loss decreasing
3. **If still stuck** - try reducing LR further (--lr 0.00005)
4. **If improving** - let it run to completion

## Troubleshooting

**If loss still doesn't change:**
- Try even lower LR: `--lr 0.00001`
- Check if model is too complex (overfitting)
- Verify data quality (no NaN values, proper normalization)

**If loss increases:**
- LR might still be too high
- Try: `--lr 0.00005`

**If training is too slow:**
- Increase LR slightly: `--lr 0.0005`
- Reduce batch size for more frequent updates

## Files Modified

1. [config/config.yaml](config/config.yaml) - Updated hyperparameters
2. [resume_training.py](resume_training.py) - Added LR override options
3. [src/models/trainer.py](src/models/trainer.py) - Added warm-up functionality
