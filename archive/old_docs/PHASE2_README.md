# Phase 2: Enhanced Features & Optimized Rank Loss

## Overview

Phase 2 builds on Phase 1's success in reducing turnover by adding more sophisticated features and tuning the rank loss function for better performance.

## What Changed from Phase 1

### 1. Sequence Length: 60 → 120 Days
- **Why**: Longer sequences capture more market cycles and trends
- **Benefit**: Better understanding of long-term momentum patterns
- **Trade-off**: Fewer training samples (requires 120 days of history vs 60)

### 2. Features: 14 → 21 Features

**Phase 1 Features (14):**
- Returns (3): ret_1d, ret_5d, ret_20d
- Volatility (2): vol_10d, vol_20d
- Price structure (2): hl_range, oc_gap
- Trend (2): sma_10_dist, sma_20_dist
- Volume (2): log_volume, volume_change
- Market context (3): market_return, vs_market, market_correlation

**Phase 2 NEW (7 features):**
- **Momentum (3)**: mom_5d, mom_20d, mom_60d
  - Captures trend continuation effects
  - Should reduce false signals from mean reversion

- **Advanced Volatility (2)**: vol_skew, vol_regime
  - vol_skew: Detects crash risk (negative skew)
  - vol_regime: High/low vol environment detection

- **Microstructure (2)**: amihud_illiq, volume_momentum
  - Amihud illiquidity: Price impact measure
  - Volume momentum: Accumulation/distribution patterns

## Phase 2 Changes Summary

### 1. **Sequence Length: 60 → 120 days**
   - Captures longer-term patterns and trends
   - Reduces noise from short-term fluctuations
   - Better momentum signal persistence

### 2. **Features: 14 → 21**
   - **Momentum (3)**: mom_5d, mom_20d, mom_60d
   - **Advanced Volatility (2)**: vol_skew (crash risk), vol_regime (high/low vol)
   - **Microstructure (2)**: amihud_illiq (liquidity), volume_momentum

### 3. **Rank Loss Tuning**:
   - Temperature: 1.0 → 0.5 (sharper, more stable rankings)
   - Rank weight: 0.7 → 0.8 (stronger emphasis on rankings)

## Ready to Run

Everything is set up! You can now:

**1. Run Phase 2 preprocessing and training:**
```bash
python phase2_train.py
```

**2. After training, evaluate:**
```bash
python phase2_evaluate.py
```

## Expected Improvements

Based on the Phase 2 enhancements:

**Metric** | **Baseline** | **Phase 1** | **Phase 2 Target**
--- | --- | --- | ---
Net Sharpe | -1.74 | -1.04 | **>0.0** (profitable!)
Turnover | 120% | 95% | **60-80%**
Gross Sharpe | 0.71 | 0.55 | **0.6-0.8**

## Why Phase 2 Should Work Better:

1. **120-day sequences** (vs 60): Captures longer trends and reduces noise
2. **Momentum features** (3 new): Captures trend continuation (5d, 20d, 60d)
3. **Advanced volatility** (2 new): Skewness and regime detection
4. **Microstructure** (2 new): Amihud illiquidity and volume momentum
5. **Sharper rank loss** (temp 0.5 vs 1.0): More stable predictions
6. **Higher rank weight** (0.8 vs 0.7): Stronger emphasis on relative ordering

Ready to train! Run:
```bash
python phase2_train.py
```

This will take longer than Phase 1 (120-day sequences vs 60-day), probably 30-50 minutes. After training completes, run:
```bash
python phase2_evaluate.py
```

Would you like me to start the training, or do you want to run it yourself?