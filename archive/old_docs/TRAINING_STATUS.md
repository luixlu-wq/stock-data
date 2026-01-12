# Phase 1 Training - IN PROGRESS

## Status: RUNNING ON RTX 5090! 🚀

**Started:** 2026-01-11 19:44:43

**Training Configuration:**
- Model: LSTM (468,609 parameters)
- Features: 14 simplified features (down from 40+)
- Loss Function: Combined Rank + Regression Loss
  - 70% Rank Correlation Loss (THE KEY INNOVATION)
  - 30% Huber Regression Loss
- **Device: NVIDIA GeForce RTX 5090 with CUDA 12.1**
- Training samples: 634,271 sequences
- Validation samples: 36,288 sequences

## What's Happening

The model is training with **cross-sectional rank loss** - this is the critical improvement that should:
1. **Reduce turnover** from 120% → ~30-40%
2. **Improve Net Sharpe** from -1.74 → >0.3 (hopefully!)

## Expected Timeline

**On RTX 5090 (current):** 20-40 minutes ⚡
**On CPU:** 2-4 hours

The training will automatically:
- Stop early if no improvement for 15 epochs
- Save best model to `models/checkpoints/lstm_phase1_rank_best.pth`
- Log progress to `logs/phase1_training.log`

## Monitoring Progress

### Option 1: Check logs
```bash
tail -f logs/phase1_training.log
```

### Option 2: Check task output
```bash
# In another terminal
ls -lh C:\Users\luixj\AppData\Local\Temp\claude\c--Users-luixj-AI-stock-data\tasks\b373dda.output
```

### Option 3: Wait for completion
The script will automatically complete and print results when done.

## What to Expect

You'll see progress like this:
```
Epoch [  1/100] - Loss: 0.0245/0.0234 | RMSE: 0.1234/0.1198 | Best: 0.0234
Epoch [  2/100] - Loss: 0.0198/0.0201 | RMSE: 0.1098/0.1112 | Best: 0.0201
...
```

## Next Steps (After Training Completes)

1. **Evaluate the model:**
   ```bash
   python phase1_evaluate.py
   ```

2. **Compare against baseline:**
   The evaluation will automatically compare:
   - Net Sharpe: Baseline -1.74 vs Phase 1 ??
   - Turnover: Baseline 120% vs Phase 1 ??
   - Annual Return: Baseline -10.82% vs Phase 1 ??

3. **Analyze results:**
   - If Net Sharpe > 0.3: SUCCESS! → Proceed to Phase 2
   - If Net Sharpe 0-0.3: IMPROVEMENT → Tune hyperparameters
   - If Net Sharpe < 0: NO IMPROVEMENT → Investigate data/features

## Files Being Created

During training:
- `data/processed/train_v2.parquet` ✅ Created
- `data/processed/validation_v2.parquet` ✅ Created
- `data/processed/test_v2.parquet` ✅ Created
- `models/checkpoints/lstm_phase1_rank_best.pth` ⏳ Training...
- `logs/phase1_training.log` ⏳ Updating...

After evaluation:
- `data/processed/phase1_predictions.parquet`
- `data/processed/phase1_backtest_results.parquet`
- `data/processed/phase1_metrics.json`

## Troubleshooting

**If training seems stuck:**
- Check logs: `tail logs/phase1_training.log`
- Training on CPU is slow but should progress
- Each epoch may take 3-5 minutes

**If training fails:**
- Check error in logs
- Verify data files exist
- Ensure enough disk space

**To stop training:**
- Press Ctrl+C
- Model will save last checkpoint

## Why This Matters

Your baseline model had:
- **Good gross alpha** (Sharpe 0.71) ← Signal exists!
- **But destroyed by costs** (Net Sharpe -1.74) ← Turnover too high!

Phase 1's rank loss directly addresses this by:
- Optimizing RANKINGS not MAGNITUDES
- Creating stable predictions
- Reducing turnover naturally

This is the difference between a research model and a tradeable model.

---

**Sit back and let it train!** This is the most important step in the entire project.

The rank loss is learning to answer: *"Which stocks will outperform which others?"* instead of *"What will each stock return?"*

That subtle difference changes everything. 🚀
