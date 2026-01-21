# Phase 1: LSTM Model Training

**Train LSTM model with rank-regression loss**

## Objective

Train LSTM neural network to predict next-day stock returns.

## Approach

- **Model**: 2-layer LSTM (128 hidden units)
- **Features**: 14 core engineered features
- **Loss**: Combined rank (70%) + regression (30%)
- **Temperature**: 0.05 (critical parameter)
- **Sequence**: 90-day lookback

## Training Results

| Metric | Value |
|--------|-------|
| **Validation Spearman** | 0.120 |
| **Information Coefficient (IC)** | 0.112 |
| **IC Information Ratio** | 1.40 |
| **Hit Rate** | 52.3% |

## Backtest Results (Gross Performance)

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | 2.53 |
| **Annual Return** | 25.8% |
| **Volatility** | 10.2% |
| **Max Drawdown** | -5.1% |

## Key Findings

1. **Temperature 0.05 is optimal** - Sharp rankings crucial
2. **Model has significant alpha** - Spearman 0.12, IC 1.40
3. **6x improvement over baseline** - Sharpe 0.42 → 2.53

## Predictions Generated

- **Period**: 2025-04-01 to 2025-10-31 (159 trading days)
- **File**: `data/processed/phase1_predictions.parquet`
- **Used for**: All subsequent phases

## Status

✅ **Complete** - Model trained, predictions generated

---

**Next**: [PHASE2A_TEMPERATURE.md](PHASE2A_TEMPERATURE.md) - Temperature tuning
