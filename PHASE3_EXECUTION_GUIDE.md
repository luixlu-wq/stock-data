# Phase 3 Execution Guide

**STATUS: PHASE 3 COMPLETE ✅**

All Phase 3 analysis complete. Production configuration validated and ready for paper trading.

**Next Step**: Phase 4 - Paper Trading (60+ days)

---

## Prerequisites

```bash
# Ensure you're in the project directory with venv activated
cd c:\Users\luixj\AI\stock-data
venv\Scripts\activate

# Install required packages if not already installed
pip install pyarrow yfinance scikit-learn scipy
```

---

## Phase 3.2b: Factor Exposure Analysis ✅

**Status**: COMPLETE

**What it does**: Calculates market beta by regressing PnL vs SPY returns

**Command**:
```bash
python scripts/risk/phase3_2b_factor_exposure.py
```

**Results**:
- ✅ **Beta**: 0.03 (excellent - effectively market-neutral)
- ✅ **R²**: 0.0015 (alpha not explained by market)
- ✅ **t-statistic**: 0.36 (not significant - confirms independence)

**Output**: `data/processed/phase3_2b_factor_exposure_results.json`

---

## Phase 3.3: Portfolio Comparison ✅

**Status**: COMPLETE (renamed from execution stress tests)

**What it does**: Tests P0 (50/50), P1 (70/30), P2 (130/30) portfolio configurations

**Command**:
```bash
python scripts/portfolio/phase3_3_portfolio_comparison.py
```

**Results**:
- **P0 (50/50 dollar-neutral)**: Sharpe 0.73, MaxDD -7.25% ✅ PASS
- **P1 (70/30 long-biased)**: Sharpe 0.84, MaxDD -7.27% ✅ PASS
- **P2 (130/30 levered)**: Sharpe 0.61, MaxDD -13.26% 🚨 FAIL

**Critical Finding**:
- Cost calculation bug fixed: Must use `(traded / 2.0) * tc` not `traded * tc`
- P2 failed deployment gates (MaxDD > 10%)
- Decision: Proceed to Phase 3.4 to salvage short performance

**Output**: `data/processed/phase3_3_portfolio_comparison_results.json`

---

## Phase 3.4: Short Salvage ✅

**Status**: COMPLETE

**What it does**: Tests short portfolio modifications to improve short-side performance

**Command**:
```bash
python scripts/portfolio/phase3_4_short_salvage.py
```

**Configurations Tested**:
- **S0**: Baseline 130/30 (from P2)
- **S1_ReduceShorts**: 130/15 (reduce short exposure)
- **S2_FilterNegative**: 130/70 with `y_pred < 0` filter (WINNER ✅)
- **S3_InvertShorts**: 130/-30 (invert short positions)

**Results**:
- **S2_FilterNegative Winner**: Sharpe 1.30, MaxDD -9.40%
  - Long Sharpe: 0.92
  - Short Sharpe: 0.29 (vs -1.69 without filter)
  - **Critical Filter**: `shorts = day[day['y_pred_reg'] < 0].tail(K)`

**Critical Finding**: Short filter transforms unprofitable shorts into profitable contribution

**Output**: `data/processed/phase3_4_short_salvage_results.json`

---

## Phase 3.5: Risk Management ✅

**Status**: COMPLETE

**What it does**: Applies volatility targeting and kill switches to S2_FilterNegative

**Command**:
```bash
python scripts/portfolio/phase3_5_risk_management.py
```

**Risk Controls Implemented**:
1. **Volatility Targeting**: 8% annual (20-day lookback, scale range [0.5, 2.0])
2. **Kill Switch #1**: Daily loss > 3-sigma
3. **Kill Switch #2**: Rolling 5-day DD > 8%
4. **Kill Switch #3**: Rolling 60-day Sharpe < 0

**Final Production Metrics**:
- ✅ **Vol-Targeted Sharpe**: 1.29
- ✅ **Annual Return**: 11.95%
- ✅ **Volatility**: 9.27% (target: 8.0%)
- ✅ **Max Drawdown**: -5.21%
- ✅ **Kill Switch Events**: 15 total (8.0% of days - acceptable)

**Deployment Decision**: GREEN LIGHT for paper trading

**Output**: `data/processed/phase3_5_risk_management_results.json`

---

## Completed Execution Summary

### Phase 3 Workflow (As Executed)

**Step 1: Factor Exposure** ✅
```bash
python scripts/risk/phase3_2b_factor_exposure.py
```
Result: Beta 0.03 (excellent)

**Step 2: Portfolio Comparison** ✅
```bash
python scripts/portfolio/phase3_3_portfolio_comparison.py
```
Result: P2 failed gates → proceed to short salvage

**Step 3: Short Salvage** ✅
```bash
python scripts/portfolio/phase3_4_short_salvage.py
```
Result: S2_FilterNegative winner (Sharpe 1.30)

**Step 4: Risk Management** ✅
```bash
python scripts/portfolio/phase3_5_risk_management.py
```
Result: Vol-targeted Sharpe 1.29 → GREEN LIGHT

---

## Success Criteria - ACHIEVED ✅

### Phase 3.2b (Factor Exposure) ✅
- ✅ Beta = 0.03 (excellent)
- ✅ R² = 0.0015 (alpha not explained by market)
- ✅ t-statistic = 0.36 (not significant - confirms independence)

### Phase 3.3 (Portfolio Comparison) ✅
- ✅ P0/P1 passed deployment gates
- ✅ P2 failed → triggered Phase 3.4 short salvage
- ✅ Cost calculation bug identified and fixed

### Phase 3.4 (Short Salvage) ✅
- ✅ S2_FilterNegative identified as winner
- ✅ Short filter transforms -1.69 Sharpe → +0.29 Sharpe
- ✅ Raw Sharpe 1.30 (pre vol-targeting)

### Phase 3.5 (Risk Management) ✅
- ✅ Volatility targeting implemented (8% target)
- ✅ Kill switches functional (8.0% event rate)
- ✅ Vol-Targeted Sharpe 1.29
- ✅ Max Drawdown -5.21% (acceptable)
- ✅ GREEN LIGHT deployment decision

---

## Next Steps - Paper Trading (Phase 4)

### Required Before Live Deployment

**Paper Trading Duration**: 60+ trading days minimum

**What to Monitor**:
1. Actual vs backtest Sharpe (expect ≥1.0)
2. Transaction costs vs 5 bps assumption
3. Slippage on market orders
4. Kill switch trigger frequency
5. Volatility targeting effectiveness

**Success Criteria for Going Live**:
- Paper trading Sharpe > 1.0
- Costs within 5-7 bps
- No systematic execution issues
- Kill switches working as designed

**Deployment Path**:
- Week 1-2: 10% capital
- Week 3-4: 25% capital (if Sharpe > 1.0)
- Week 5-8: 50% capital (if Sharpe > 1.0)
- Week 9+: 100% capital (if Sharpe > 1.0)

---

## Troubleshooting

### Import Errors
```bash
# Missing pyarrow
pip install pyarrow

# Missing yfinance
pip install yfinance

# Missing sklearn
pip install scikit-learn
```

### Data Not Found
```bash
# If phase1_predictions.parquet missing
python scripts/training/phase1_train.py

# Or use Phase 2A predictions
python scripts/training/phase2a_temperature_experiment.py
```

### Memory Issues
- Stress tests load full prediction history
- Expected memory usage: ~2-4 GB RAM
- Close other applications if needed

---

## Quick Results Check

### After Phase 3.2b:
```bash
# Check beta value
grep "Beta (Market Exposure)" logs/phase3_2b_factor_exposure.log
```

### After Phase 3.3:
```bash
# Check best and worst case
grep -A 5 "KEY FINDINGS" logs/phase3_3_execution_stress.log
```

### After Phase 3.4:
```bash
# Check kill switch events
grep "KILL SWITCH" logs/phase3_4_capital_scaling.log
```

---

## Critical Files

### Input:
- `data/processed/phase1_predictions.parquet` - Model predictions (188 trading days)
- `data/processed/s2_daily_pnl_history.parquet` - S2 daily PnL (for Phase 3.5)

### Phase 3 Outputs:
- `data/processed/phase3_2b_factor_exposure_results.json` - Beta analysis
- `data/processed/phase3_3_portfolio_comparison_results.json` - P0/P1/P2 comparison
- `data/processed/phase3_4_short_salvage_results.json` - S0/S1/S2/S3 comparison
- `data/processed/phase3_5_risk_management_results.json` - Final production metrics
- `data/processed/phase3_risk_decomposition_results.json` - Long/short attribution

### Documentation:
- [STRATEGY_DEFINITION.md](STRATEGY_DEFINITION.md) - v2.0.0 production spec (UPDATED ✅)
- [docs/PHASE3_COMPLETE.md](docs/PHASE3_COMPLETE.md) - Phase 3 comprehensive summary
- [docs/PHASE3_2_FINDINGS.md](docs/PHASE3_2_FINDINGS.md) - Risk decomposition details
- [docs/PHASE3_3_FIXES.md](docs/PHASE3_3_FIXES.md) - Cost calculation bug fix
- [docs/PHASE3_FINAL_DECISION.md](docs/PHASE3_FINAL_DECISION.md) - Deployment decision rationale

---

## Phase 3 Final Checklist ✅

**All items complete**:

- ✅ Run Phase 3.2b (factor exposure)
- ✅ Beta = 0.03 confirmed (excellent)
- ✅ Run Phase 3.3 (portfolio comparison)
- ✅ Identified P2 failure → short salvage needed
- ✅ Run Phase 3.4 (short salvage)
- ✅ S2_FilterNegative identified as winner
- ✅ Run Phase 3.5 (risk management)
- ✅ Vol-targeted Sharpe 1.29 → GREEN LIGHT
- ✅ Documented all findings
- ✅ Updated STRATEGY_DEFINITION.md to v2.0.0
- ✅ Updated PHASE3_EXECUTION_GUIDE.md (this file)
- ✅ Decided on deployment path (paper trading → phased live)

**Phase 3 Status**: COMPLETE ✅

**Production Configuration**:
- S2_FilterNegative (130/70 with short filter)
- 8% volatility targeting (20-day lookback)
- 3 kill switches (3-sigma, 8% DD, Sharpe < 0)
- Expected Sharpe: 1.29
- Expected Return: 11.95% annual
- Expected Volatility: 9.27%
- Expected MaxDD: -5.21%

---

**Remember**: Phase 3 is not about ML anymore. It's about execution, risk, and reality.

**The goal**: Understand how the strategy behaves under stress so you can deploy it safely.

**The output**: Confidence that your alpha is real, robust, and deployable.
