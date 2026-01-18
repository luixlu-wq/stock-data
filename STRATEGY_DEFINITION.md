# CANONICAL STRATEGY DEFINITION

**Version**: 2.0.0
**Date Frozen**: 2026-01-18
**Status**: PRODUCTION READY FOR PAPER TRADING

⚠️ **NO CHANGES ALLOWED BEYOND THIS POINT**
This strategy has been validated and frozen. Any modifications invalidate all backtests and risk analysis.

---

## Executive Summary

**Strategy Type**: LSTM-based cross-sectional equity long-short alpha
**Configuration**: S2_FilterNegative (130/70 with short filter + volatility targeting)
**Vol-Targeted Sharpe Ratio**: 1.29
**Annual Return**: 11.95%
**Volatility**: 9.27% (target: 8.0%)
**Max Drawdown**: -5.21%

**Alpha Validation**: Phase 3 complete - GREEN LIGHT for paper trading

---

## Universe Definition

**Universe**: ~189 liquid US equities
- Selection criteria: Available in Yahoo Finance data
- Market cap: No explicit filter (implicit liquidity filter via data availability)
- Exclusions: None (universe defined by data availability)

**Rebalance**: Daily (end-of-day signal → next-day execution)

---

## Signal Generation

### Model Architecture
- **Type**: 2-layer LSTM
- **Hidden Size**: 128 units per layer
- **Input Features**: 14 core features
- **Sequence Length**: 90 trading days
- **Model Checkpoint**: `models/checkpoints/lstm_phase2a_temp0.05_best.pth`

### Feature Set (14 Core Features)

**Returns (3)**:
1. Daily return (t)
2. 5-day return
3. 20-day return

**Volatility (3)**:
4. 20-day realized volatility
5. 5-day volatility
6. Volume volatility (20-day)

**Price Structure (3)**:
7. Distance from 20-day high
8. Distance from 20-day low
9. Price range compression (high-low / close)

**Trend (2)**:
10. 20-day momentum
11. 5-day momentum

**Volume (2)**:
12. Volume ratio (current / 20-day average)
13. Volume trend (5-day / 20-day)

**Market Context (1)**:
14. Market return (cross-sectional context)

### Loss Function
- **Type**: Combined rank-regression loss
- **Rank Weight**: 70% (Spearman rank correlation)
- **Regression Weight**: 30% (Huber loss, δ=0.05)
- **Temperature**: 0.05 (sharp rankings - CRITICAL PARAMETER)

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 50
- **Early Stopping**: Patience 15, min_delta 0.0001
- **Device**: CUDA (GPU required for production training)

---

## Portfolio Construction

### Signal Processing
1. **Daily Prediction**: Model outputs predicted returns for all stocks
2. **Ranking**: Stocks ranked by predicted return (cross-sectional)
3. **Target Positions (S2_FilterNegative)**:
   - **Long**: Top K stocks (K = 38, equal-weighted)
   - **Short**: Bottom K stocks **FILTERED BY y_pred < 0** (equal-weighted)
   - **Critical Filter**: `shorts = day[day['y_pred_reg'] < 0].tail(K)`
   - **Effect**: Eliminates -1.69 Sharpe short drag → +0.61 Sharpe contribution

### Position Smoothing (CRITICAL FOR PERFORMANCE)
- **Method**: Exponentially Weighted Moving Average (EWMA)
- **Smoothing Parameter**: α = 0.15
- **Formula**: `pos_t = (1 - α) × pos_{t-1} + α × target_t`
- **Effect**: Reduces turnover while preserving signal

### Portfolio Constraints
- **Long Exposure**: 65% (130% of base)
- **Short Exposure**: 35% (70% of base)
- **Gross Exposure**: 100% (long + |short|)
- **Net Exposure**: 30% (long-biased)
- **Single Position Limit**: 1 / K (currently: 1/38 ≈ 2.6% gross)

### Rebalancing
- **Frequency**: Daily
- **Timing**: End-of-day signal → next-day market open execution
- **Method**:
  1. Generate predictions at market close (t)
  2. Apply EWMA smoothing using t-1 positions
  3. Enforce dollar neutrality
  4. Execute trades at market open (t+1)

---

## Cost Model

### Transaction Costs
- **Assumption**: 5 basis points per trade (one-way)
- **Calculation**: `cost = (|Δpos| / 2) × 0.0005`
- **Rationale**:
  - Electronic market maker spread: ~2-3 bps
  - Market impact (small size): ~1-2 bps
  - Commission: ~1 bp
  - **Total**: ~5 bps (conservative estimate)

### Cost Impact
- **Daily Turnover**: 22% average
- **Daily Cost**: ~1.1 bps (0.22 × 5 bps)
- **Annual Cost**: ~280 bps (assuming 252 trading days)

---

## Performance Characteristics

### Production Configuration (S2_FilterNegative + Vol Targeting)

**Final Validated Metrics** (Phase 3.5):
- **Vol-Targeted Sharpe**: 1.29
- **Annual Return**: 11.95%
- **Volatility**: 9.27% (target: 8.0%)
- **Max Drawdown**: -5.21%
- **Raw Sharpe** (no vol targeting): 1.30
- **Raw Return**: 18.76%
- **Raw Volatility**: 14.42%

**Long/Short Attribution**:
- **Long Sharpe**: 0.92
- **Short Sharpe**: 0.29 (after negative filter)
- **Short Sharpe (before filter)**: -1.69 (CRITICAL FIX)

**Risk Management**:
- **Kill Switch Events**: 15 total (8.0% of days)
  - KS1 (3-sigma loss): 2 events
  - KS2 (8% drawdown): 0 events
  - KS3 (Sharpe < 0): 13 events
- **Deployment Status**: GREEN LIGHT

### Signal Characteristics
- **Type**: Medium-horizon cross-sectional ranking alpha
- **Persistence**: 2-5 days (signal decays slowly)
- **Nature**: Relative regime estimation (not momentum, not trend)
- **Market Beta**: 0.03 (effectively market-neutral despite 30% net exposure)

---

## Data Requirements

### Historical Data
- **Source**: Yahoo Finance (via yfinance Python library)
- **Frequency**: Daily OHLCV
- **History Required**: 120 trading days minimum (90-day sequences + 30-day buffer)
- **Data Quality**:
  - Must handle missing data (forward-fill up to 5 days)
  - Must handle corporate actions (splits, dividends)

### Train/Test Split
- **Training**: 2020-01-01 to 2024-09-30
- **Validation**: 2024-10-01 to 2024-12-31
- **Test**: 2025-01-01 onwards

**Critical**: No lookahead bias. All features computed causally.

---

## Risk Parameters

### Volatility Targeting (Phase 3.5)
- **Target Volatility**: 8% annual
- **Lookback Window**: 20 trading days
- **Scaling Range**: [0.5, 2.0] (capped for stability)
- **Formula**: `scale = min(2.0, max(0.5, target_vol / rolling_vol))`
- **Effect**: Stabilizes portfolio volatility across regimes

### Kill Switches (Phase 3.5)
1. **Daily Loss > 3-Sigma**: Triggered on extreme loss days
   - Threshold: `mean_pnl - 3 × std_pnl`
   - Action: Flatten positions for the day

2. **Rolling 5-Day Drawdown > 8%**: Triggered on sustained drawdowns
   - Lookback: 5 trading days
   - Action: Halt trading until recovery

3. **Rolling 60-Day Sharpe < 0**: Triggered on strategy breakdown
   - Lookback: 60 trading days
   - Action: Disable strategy until review

### Position Limits
- **Single Position**: No more than 1 / K (currently: 1/38 ≈ 2.6% gross)
- **Long Exposure**: 65% (130% of base capital)
- **Short Exposure**: 35% (70% of base capital)
- **Max Gross Exposure**: 100%

### Risk Metrics to Monitor
- **Daily Sharpe**: Rolling 60-day (kill switch if < 0)
- **Rolling Volatility**: 20-day (for vol targeting)
- **Drawdown**: 5-day rolling (kill switch if > 8%)
- **Factor Exposure**: Beta to market (validated at 0.03)

---

## Frozen Components

### DO NOT CHANGE
These components are frozen and define the validated strategy:

**Model**:
- ✅ Architecture (2-layer LSTM, 128 hidden)
- ✅ Features (14 core features)
- ✅ Sequence length (90 days)
- ✅ Loss function (70% rank + 30% Huber)
- ✅ Temperature (0.05)
- ✅ Model checkpoint (lstm_phase2a_temp0.05_best.pth)

**Portfolio**:
- ✅ K = 38 (top/bottom stocks)
- ✅ S2_FilterNegative short filter: `y_pred_reg < 0`
- ✅ Long exposure: 65% (130/70 asymmetry)
- ✅ Short exposure: 35%
- ✅ Equal-weighting within buckets
- ✅ EWMA smoothing (α=0.15)
- ✅ Daily rebalance

**Risk Management**:
- ✅ Volatility targeting (8% annual, 20-day lookback)
- ✅ Kill switches (3-sigma, 8% DD, Sharpe < 0)
- ✅ Scaling range [0.5, 2.0]

**Costs**:
- ✅ 5 bps transaction cost

### COMPLETED (Phase 3)
- ✅ Phase 3.1: Strategy canonicalization
- ✅ Phase 3.2: Risk decomposition (beta=0.03)
- ✅ Phase 3.3: Portfolio comparison (P0/P1/P2 tested)
- ✅ Phase 3.4: Short salvage (S2 winner)
- ✅ Phase 3.5: Risk management (vol targeting + kill switches)

**NO NEW FEATURES. NO RETRAINING. NO MODEL CHANGES.**

---

## Next Steps

### Phase 4: Paper Trading (60+ days required)
- **Duration**: Minimum 60 trading days
- **Zero parameter changes**
- **Log every fill and cost**
- **Compare paper vs backtest metrics**
- **Decision gate**: Proceed to live only if Sharpe > 1.0

### Phase 5: Live Deployment (Phased)
- **Week 1-2**: 10% of target capital
- **Week 3-4**: 25% of target capital (if Sharpe > 1.0)
- **Week 5-8**: 50% of target capital (if Sharpe > 1.0)
- **Week 9+**: 100% of target capital (if Sharpe > 1.0)

**Rollback criteria**: If any kill switch triggers twice in one week → reduce capital by 50%

---

## Phase 3 Completion Summary

### Phase 3.1: Canonicalization ✅
- Documented frozen strategy (v1.0.0)
- Committed to version control

### Phase 3.2: Risk Decomposition ✅
- **Market Beta**: 0.03 (effectively neutral)
- **R²**: 0.0015 (alpha not explained by market)
- **Long Sharpe**: 0.92
- **Short Sharpe**: -1.69 (before filter) → 0.29 (after filter)

### Phase 3.3: Portfolio Comparison ✅
- **Tested**: P0 (50/50), P1 (70/30), P2 (130/30)
- **Finding**: P2 failed deployment gates (Sharpe 0.61, MaxDD -13%)
- **Decision**: Proceed to Phase 3.4 short salvage

### Phase 3.4: Short Salvage ✅
- **Winner**: S2_FilterNegative (filter shorts by y_pred < 0)
- **Raw Sharpe**: 1.30
- **Configuration**: 130/70 long/short
- **Finding**: Short filter critical for profitability

### Phase 3.5: Risk Management ✅
- **Vol-Targeted Sharpe**: 1.29
- **Annual Return**: 11.95%
- **Volatility**: 9.27% (target: 8.0%)
- **Max Drawdown**: -5.21%
- **Kill Switch Events**: 8.0% of days (acceptable)
- **Deployment Status**: GREEN LIGHT for paper trading

---

## Implementation Files

### Core Strategy
- [src/data/preprocessor_v2.py](src/data/preprocessor_v2.py) - Feature engineering
- [src/models/lstm_model.py](src/models/lstm_model.py) - LSTM architecture
- [src/models/losses.py](src/models/losses.py) - Rank loss implementation
- [models/checkpoints/lstm_phase2a_temp0.05_best.pth](models/checkpoints/lstm_phase2a_temp0.05_best.pth) - Production model

### Backtesting
- [scripts/backtest/phase2b_monetization.py](scripts/backtest/phase2b_monetization.py) - Portfolio engineering backtest
- [scripts/validation/phase2b_validate_baseline.py](scripts/validation/phase2b_validate_baseline.py) - Baseline validation

### Configuration
- [config/config.yaml](config/config.yaml) - Frozen model configuration

---

## Version History

### v2.0.0 (2026-01-18) - PRODUCTION READY
- Phase 3 complete (3.1 through 3.5)
- Production configuration: S2_FilterNegative (130/70 with short filter)
- Volatility targeting: 8% annual (20-day lookback)
- Kill switches: 3-sigma loss, 8% DD, Sharpe < 0
- Vol-Targeted Sharpe: 1.29
- Annual Return: 11.95%
- Max Drawdown: -5.21%
- Deployment Status: GREEN LIGHT for paper trading

### v1.0.0 (2026-01-12) - INITIAL FREEZE
- Initial canonical definition
- Phase 2B validation complete
- Net Sharpe 2.20 confirmed (dollar-neutral 50/50)
- Baseline stress test passed (Net Sharpe 1.32 @ 100% turnover)

---

## Approval & Sign-off

**Strategy Validated By**: Phase 0-3.5 systematic experimentation
**Phase 3 Completion**: All risk analysis complete
**Deployment Decision**: GREEN LIGHT for paper trading (60+ days)

**Status**: PRODUCTION READY - AWAITING PAPER TRADING VALIDATION

---

**This strategy definition is now FROZEN at v2.0.0**
No changes permitted during paper trading phase (Phase 4).

Reference this document for production deployment specification.
