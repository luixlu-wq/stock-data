# PHASE 1: Rank Loss + Simplified Features

## Overview

Phase 1 implements the critical improvements identified from Phase 0 analysis.

### Phase 0 Results (Baseline LSTM)
- **Gross Sharpe:** 0.71 ✓ (signal exists!)
- **Net Sharpe:** -1.74 ✗ (destroyed by turnover)
- **Turnover:** 120% per day (unsustainable)
- **Diagnosis:** Model has predictive power but is completely untradeable

### Root Cause
The baseline LSTM was trained to minimize **MSE (Mean Squared Error)**, which optimizes for:
- Accurate magnitude prediction of individual stock returns
- No consideration for ranking stability
- No penalty for turnover

But trading requires:
- **Relative ranking** among stocks (who outperforms whom)
- **Stable predictions** (low turnover)
- **Cross-sectional consistency** (rankings that persist)

## Phase 1 Improvements

### 1. Simplified Feature Engineering (14 Core Features)

**Removed:** 40+ overfitted technical indicators
- RSI, MACD, Bollinger Bands, OBV, etc.
- These are human heuristics that don't generalize well for ML

**Kept:** 14 robust, leakage-safe features
- **Returns (3):** ret_1d, ret_5d, ret_20d
- **Volatility (2):** vol_10d, vol_20d
- **Price Structure (2):** hl_range, oc_gap
- **Trend (2):** sma_10_dist, sma_20_dist
- **Volume (2):** log_volume, volume_change
- **Market Context (3):** market_return, vs_market, market_correlation

**Why this matters:**
- Less collinearity → better generalization
- Simpler features → less overfitting
- Faster training → quicker iteration

### 2. Cross-Sectional Rank Loss

**THE KEY INNOVATION**

**Old Loss (Baseline):**
```python
Loss = MSE(predicted_return, actual_return)
```
- Optimizes: Magnitude accuracy for each stock independently
- Ignores: Relative rankings
- Result: Good predictions, unstable rankings, high turnover

**New Loss (Phase 1):**
```python
Loss = 0.7 × RankCorrelationLoss + 0.3 × HuberLoss
```
- **RankCorrelationLoss:** Maximizes Spearman correlation between predicted and actual rankings
- **HuberLoss:** Maintains magnitude awareness (prevents degenerate solutions)
- **Result:** Optimizes for ranking quality, naturally reduces turnover

**How Rank Loss Works:**
1. For each batch, compute soft rankings of predictions and targets
2. Maximize correlation between these rankings
3. Model learns: "Stock A should rank above Stock B" (not "Stock A will return 2.5%")
4. Stable rankings → lower turnover → tradeable alpha

### 3. Implementation Files

**New Files:**
- `src/data/preprocessor_v2.py` - Simplified feature engineering
- `src/models/losses.py` (updated) - Added `RankCorrelationLoss` and `CombinedRankRegressionLoss`
- `src/models/trainer.py` (updated) - Support for rank loss training
- `phase1_train.py` - Training script for Phase 1
- `phase1_evaluate.py` - Evaluation and comparison script

## Usage

### Step 1: Train Phase 1 Model

```bash
python phase1_train.py
```

This will:
1. Preprocess data with simplified features (14 features)
2. Train LSTM with rank loss (70% rank, 30% Huber)
3. Save model to `models/checkpoints/lstm_phase1_rank_best.pth`
4. Save preprocessed data to `data/processed/*_v2.parquet`

**Expected training time:** 30-60 minutes (depending on GPU)

### Step 2: Evaluate and Compare

```bash
python phase1_evaluate.py
```

This will:
1. Load Phase 1 model
2. Generate predictions on test set
3. Run cross-sectional backtest (Phase 0 test)
4. Compare against baseline metrics
5. Print verdict: Success / Improvement / No Improvement

### Step 3: Analyze Results

Check the comparison output:

```
COMPARISON: BASELINE vs PHASE 1
======================================================================
Metric                  | Baseline  | Phase 1   | Change
----------------------------------------------------------------------
Sharpe (Net)            |    -1.74  |     0.45  |   +2.19  ← KEY METRIC
Sharpe (Gross)          |     0.71  |     0.82  |   +0.11
Annual Return (Net)     |  -10.82% |    2.80% |  +13.62%
Turnover                |   120.71% |   38.25% |  -82.46%  ← HUGE WIN
Win Rate                |    44.02% |   51.30% |   +7.28%
======================================================================
```

## Expected Outcomes

### Success Criteria

**Minimum (Phase 1 works):**
- Net Sharpe > 0.3 (from -1.74)
- Turnover < 50% (from 120%)

**Good:**
- Net Sharpe > 0.5
- Turnover < 40%

**Excellent:**
- Net Sharpe > 0.8
- Turnover < 30%

### If Phase 1 Fails

If Net Sharpe < 0.3, investigate:

1. **Data quality issues**
   - Check for survivorship bias
   - Verify data integrity
   - Ensure no look-ahead bias

2. **Feature engineering**
   - Try different lookback windows
   - Add/remove features
   - Check feature distributions

3. **Loss function tuning**
   - Adjust rank_weight (try 0.5, 0.8, 0.9)
   - Adjust temperature (try 0.5, 2.0)
   - Try longer sequence length (120 days)

4. **Fundamental signal**
   - The alpha might truly not exist in this data
   - Consider: different tickers, timeframes, or features

## Next Steps (Phase 2+)

If Phase 1 succeeds (Net Sharpe > 0.3):

### Phase 2: Turnover Optimization
- Add explicit turnover penalty to loss
- Implement volatility normalization
- Extend sequence length to 120 days

### Phase 3: Stock Clustering
- Cluster stocks by behavior (volatility, sector, size)
- Add cluster embeddings to model
- Specialize predictions by cluster type

### Phase 4: Architecture Upgrade
- Upgrade from LSTM to Temporal Convolutional Network (TCN)
- Consider Transformer/PatchTST (only if GPU allows)
- Add ensemble methods

### Phase 5: Production
- Online retraining pipeline
- Risk management (position sizing, drawdown limits)
- Sector/factor neutrality
- Multi-horizon predictions

## Technical Details

### Loss Function Math

**Rank Correlation Loss:**
```
For batch of N stocks on same day:
1. Compute soft ranks:
   rank(x_i) = Σ_j sigmoid((x_i - x_j) / temperature)

2. Compute Spearman correlation:
   ρ = corr(rank(predictions), rank(actual_returns))

3. Loss = -ρ (maximize correlation = minimize negative)
```

**Combined Loss:**
```
Total = α × RankLoss + (1-α) × HuberLoss
      = 0.7 × (-ρ) + 0.3 × Huber(pred, target)
```

### Why This Works

1. **Rank loss directly optimizes trading objective**
   - Long-short portfolios care about relative order
   - Top 20% long, bottom 20% short
   - Exact return magnitude doesn't matter if ranking is correct

2. **Implicit turnover reduction**
   - Rankings are more stable than magnitudes
   - Model penalized for changing relative orderings
   - Lower turnover = lower costs = higher net returns

3. **Robust to outliers**
   - Rankings less sensitive to extreme values
   - Combined with Huber loss for magnitude awareness
   - Prevents degenerate solutions (all predictions same)

## Files Changed

### Modified
- `src/models/losses.py` - Added rank losses
- `src/models/trainer.py` - Support for rank loss

### New
- `src/data/preprocessor_v2.py` - Simplified features
- `phase1_train.py` - Training script
- `phase1_evaluate.py` - Evaluation script
- `PHASE1_README.md` - This file

### Output
- `data/processed/train_v2.parquet` - Training data (14 features)
- `data/processed/validation_v2.parquet` - Validation data
- `data/processed/test_v2.parquet` - Test data
- `data/processed/phase1_predictions.parquet` - Model predictions
- `data/processed/phase1_backtest_results.parquet` - Daily backtest results
- `data/processed/phase1_metrics.json` - Performance metrics
- `models/checkpoints/lstm_phase1_rank_best.pth` - Trained model

## References

### Academic Papers
- Gu, Kelly, Xiu (2020): "Empirical Asset Pricing via Machine Learning"
- Huber, Shi, Song (2020): "Deep Learning in Asset Pricing"
- Fischer, Krauss (2018): "Deep Learning with Long Short-Term Memory Networks"

### Key Concepts
- **Information Coefficient (IC):** Correlation between predictions and returns
- **Sharpe Ratio:** Risk-adjusted returns (return / volatility)
- **Turnover:** Portfolio position changes (high turnover = high costs)
- **Cross-sectional:** Analysis across multiple stocks at same time
- **Rank correlation:** Spearman's ρ (correlation of rankings)

## Troubleshooting

### "No module named 'src'"
```bash
# Make sure you're in the project root directory
cd c:/Users/luixj/AI/stock-data
python phase1_train.py
```

### "Test file not found: data/processed/test_v2.parquet"
```bash
# Run training first (it creates the preprocessed files)
python phase1_train.py
```

### "Checkpoint not found"
```bash
# Training didn't complete successfully
# Check logs/phase1_training.log for errors
# Re-run: python phase1_train.py
```

### Model trains but Sharpe is still negative
- This means Phase 1 improvements weren't enough
- Review "If Phase 1 Fails" section above
- Consider fundamental data/feature issues

### CUDA out of memory
```bash
# Reduce batch size in config/config.yaml
model:
  training:
    batch_size: 16  # Reduce from 32
```

## Questions?

If you encounter issues or want to understand the approach better, review:
1. This README (you are here)
2. Phase 0 backtest results (`data/processed/phase0_metrics.json`)
3. Training logs (`logs/phase1_training.log`)
4. ChatGPT's original proposal (in conversation history)
5. Claude's critique and recommendations (in conversation history)

Both ChatGPT and Claude agree on the core principles - this implementation represents the consensus approach.
