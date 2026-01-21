# Project Overview

**ML-Driven Stock Trading Strategy - Complete System**

## What This Project Does

This is a fully-developed machine learning trading system that predicts stock returns and executes a long/short equity strategy. The system:

- Predicts next-day returns for S&P 500 stocks using LSTM neural networks
- Constructs a 130/70 long/short portfolio (S2_FilterNegative strategy)
- Applies volatility targeting and risk management (kill switches)
- Achieved **Sharpe Ratio 2.20** (net of costs) in backtesting
- Now in **Phase 4**: 60+ days of paper trading validation

## Project Journey

### ✅ Phase 0: Baseline (Complete)
- Built baseline Buy & Hold strategy
- Result: Sharpe 0.42
- Established performance benchmark

### ✅ Phase 1: LSTM Training (Complete)
- Trained LSTM model on 14 engineered features
- Temperature calibration at 0.05
- Result: Sharpe 2.53 (gross)
- Created predictions dataset for 2025-04-01 to 2025-10-31

### ✅ Phase 2: Portfolio Optimization (Complete)
- **Phase 2A**: Temperature tuning (0.05 optimal)
- **Phase 2B**: Portfolio construction testing
  - Tested 5 strategies (S0 through S4)
  - S2_FilterNegative emerged as winner
  - Result: Sharpe 2.20 (net of costs)

### ✅ Phase 3: Risk Analysis (Complete)
- **Phase 3.1**: Risk decomposition and attribution
- **Phase 3.2**: Deep portfolio analysis
- **Phase 3.3**: Transaction cost calculation fixes
- **Phase 3.4**: Short position salvage (filtered shorts)
- **Phase 3.5**: Risk management implementation
  - Volatility targeting (8% annual)
  - Kill switches (3-sigma loss, 8% DD, Sharpe < 0)
  - Result: Vol-Targeted Sharpe 1.29
- **Decision**: GREEN LIGHT for deployment

### 🔄 Phase 4: Paper Trading (Current)
- **Goal**: 60+ days of simulated trading
- **Method**: Historical replay (2025-04-01 onwards)
- **Automation**: Daily execution at 4:15 PM
- **Storage**: Qdrant vector database
- **Status**: Running

### 📋 Phase 5: Live Deployment (Future)
- Real money deployment with real-time data
- Continuous monitoring and performance tracking
- Model retraining pipeline

## Key Strategy Components

### S2_FilterNegative Strategy
```
Long Exposure:  65% (top 38 stocks by prediction)
Short Exposure: 35% (bottom 38 stocks WHERE prediction < 0)
Rebalance:      Daily with EWMA smoothing (α=0.15)
```

**Critical Short Filter**: Only short stocks with negative predictions. This salvages performance during market rallies.

### 14 Core Features
1. Returns: 1d, 5d, 20d
2. Volatility: 5d, 20d
3. Volume metrics: ratio, change
4. Technical: RSI, Bollinger bands
5. Price patterns: momentum, mean reversion

### Risk Management
- **Volatility Targeting**: 8% annual (20-day lookback, 0.5-2.0x scale)
- **Kill Switches**:
  - 3-sigma daily loss
  - 8% trailing drawdown
  - Sharpe < 0 (60-day rolling)
- **Position Smoothing**: EWMA α=0.15 for turnover control

## Performance Summary

| Metric | Value |
|--------|-------|
| **Strategy** | S2_FilterNegative |
| **Net Sharpe** | 2.20 |
| **Vol-Targeted Sharpe** | 1.29 |
| **Annual Return (Net)** | 22.4% |
| **Volatility** | 10.2% |
| **Max Drawdown** | -6.3% |
| **Avg Turnover** | 49.1% |
| **Avg Cost** | 9.8 bps |

## Technology Stack

- **ML Framework**: PyTorch (LSTM)
- **Data**: yfinance, pandas
- **Features**: TA-Lib, custom indicators
- **Database**: Qdrant (vector storage)
- **Automation**: Windows Task Scheduler
- **Visualization**: Matplotlib, seaborn

## File Structure

```
├── data/               # Stock data and predictions
├── models/             # Trained LSTM models
├── src/                # Core source code
├── scripts/            # Automation and paper trading
├── docs/               # Documentation
├── reports/            # Performance reports
└── logs/               # Trading logs and journals
```

## Next Steps

**For New Users**:
1. Read [GETTING_STARTED.md](GETTING_STARTED.md)
2. Read [STRATEGY_DEFINITION.md](../../STRATEGY_DEFINITION.md)
3. Review Phase 0-3 results in [docs/phase_results/](../phase_results/)

**To Start Paper Trading**:
1. Read [START_TOMORROW.md](../../START_TOMORROW.md)
2. Follow [DAILY_WORKFLOW.md](DAILY_WORKFLOW.md)
3. Setup automation with [AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md)

**To Understand the System**:
1. Technical: [docs/technical/](../technical/)
2. Architecture: [ARCHITECTURE.md](../technical/ARCHITECTURE.md)
3. Features: [FEATURES.md](../technical/FEATURES.md)

## Documentation

- **[INDEX.md](../../INDEX.md)**: Quick reference to all docs
- **[PROJECT_ORGANIZATION.md](../../PROJECT_ORGANIZATION.md)**: Complete navigation guide
- **[README.md](../../README.md)**: Project homepage

## Contact & Support

For troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Status**: Phase 4 - Paper Trading
**Last Updated**: January 20, 2026
