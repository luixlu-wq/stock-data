# Stock Trading Strategy - LSTM Cross-Sectional Alpha

**Status**: Phase 4 - Paper Trading (started Jan 19, 2026)
**Strategy**: S2_FilterNegative (130/70 long/short with negative prediction filter)
**Vol-Targeted Sharpe**: 1.29 | **Net Sharpe**: 2.20 | **Max DD**: -5.21%

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Strategy Specification](#strategy-specification)
5. [Model Architecture](#model-architecture)
6. [Feature Engineering](#feature-engineering)
7. [Portfolio Construction](#portfolio-construction)
8. [Risk Management](#risk-management)
9. [Paper Trading (Phase 4)](#paper-trading-phase-4)
10. [Automation](#automation)
11. [Project History](#project-history)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

This is a complete ML-driven stock trading system that:

1. Predicts next-day returns for ~189 US stocks using a 2-layer LSTM
2. Constructs a 130/70 long/short portfolio (S2_FilterNegative)
3. Applies volatility targeting (8% annual) and kill switches
4. Currently in 60-day paper trading validation

### Key Innovation

Transformed a failing LSTM strategy (Net Sharpe -1.74) into institutional-quality alpha through:

- **Cross-sectional rank loss** (70% rank + 30% Huber) instead of MSE
- **Temperature calibration** (0.05) for sharp rankings
- **Short filtering** (only short stocks with negative predictions)
- **EWMA position smoothing** (alpha=0.15) to reduce turnover 120% to 22%
- **Volatility targeting** (8% annual) with kill switches

### Performance Summary

| Metric | Phase 2B (Net) | Phase 3.5 (Vol-Targeted) |
|--------|---------------|--------------------------|
| **Sharpe Ratio** | 2.20 | 1.29 |
| **Annual Return** | 22.4% | 11.95% |
| **Volatility** | 10.2% | 9.27% |
| **Max Drawdown** | -6.3% | -5.21% |
| **Avg Turnover** | 49.1% | 49.1% |

---

## Quick Start

### Installation

```bash
cd c:\Users\luixj\AI\stock-data
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run Paper Trading (one day)

```bash
python scripts/paper_trading/run_daily_paper_trading.py
```

The script auto-increments through historical dates (2025-04-01 onwards). Run daily.

### Check Results

```bash
# View summary
cat data/processed/phase4/phase4_paper_trading_summary.json

# Generate performance report
python scripts/paper_trading/phase4_performance_tracker.py
cat reports/phase4/phase4_performance_report.txt

# Check progress
cat data/processed/phase4/paper_trading_progress.json
```

### Query Qdrant Database (if automation enabled)

```bash
python scripts/automation/query_qdrant.py --type recommendations --limit 10
python scripts/automation/query_qdrant.py --type results --limit 5
python scripts/automation/query_qdrant.py --type performance
```

---

## Project Structure

```
stock-data/
├── config/
│   └── config.yaml                  # Frozen model configuration
│
├── src/                             # Core source code
│   ├── data/
│   │   ├── data_loader.py          # yfinance data download
│   │   └── preprocessor_v2.py      # 14 feature engineering
│   ├── models/
│   │   ├── lstm_model.py           # 2-layer LSTM architecture
│   │   ├── losses.py              # Rank-regression loss
│   │   └── trainer.py             # Training loop
│   └── utils/
│       ├── config_loader.py       # Config management
│       └── metrics.py             # Performance metrics
│
├── scripts/
│   ├── training/                   # Phase 1 model training (DO NOT rerun)
│   ├── backtest/                   # Phase 2 portfolio backtesting
│   ├── validation/                 # Alpha validation tests
│   ├── experiment/                 # Phase 2A temperature experiments
│   ├── portfolio/                  # Portfolio construction tests
│   ├── risk/                       # Risk analysis scripts
│   ├── stress_test/                # Stress testing
│   ├── deployment/                 # Deployment preparation
│   ├── paper_trading/              # Phase 4 daily execution
│   │   ├── run_daily_paper_trading.py         # Simple daily runner
│   │   ├── phase4_paper_trading_runner.py     # Core trading engine
│   │   ├── phase4_performance_tracker.py      # Performance reporting
│   │   └── phase4_daily_pipeline.py           # Live data pipeline (future)
│   └── automation/                 # Automated daily execution
│       ├── daily_paper_trading_qdrant.py      # Main automation + Qdrant
│       ├── query_qdrant.py                    # Database query tool
│       ├── setup_daily_task.ps1               # Windows Task Scheduler
│       └── setup_daily_task.bat               # Alternative batch setup
│
├── models/                         # Trained model checkpoints
│   └── checkpoints/
│       └── lstm_phase2a_temp0.05_best.pth    # Production model (FROZEN)
│
├── data/
│   ├── raw/                        # Downloaded stock data (CSV)
│   └── processed/
│       ├── phase1_predictions.parquet        # Pre-computed predictions
│       └── phase4/                           # Paper trading results
│           ├── phase4_paper_trading_daily.parquet
│           ├── phase4_paper_trading_summary.json
│           └── paper_trading_progress.json
│
├── logs/
│   ├── paper_trading/              # Daily execution logs
│   └── automation/                 # Automation logs
│
├── reports/
│   └── phase4/                     # Performance reports & plots
│
└── archive/                        # Old/deprecated scripts
```

---

## Strategy Specification

**Version**: 2.0.0 (Frozen 2026-01-18) - NO CHANGES ALLOWED

### Signal Generation

| Parameter | Value |
|-----------|-------|
| Model | 2-layer LSTM (128 hidden units) |
| Features | 14 core features |
| Sequence Length | 90 trading days |
| Loss Function | 70% rank + 30% Huber |
| Temperature | 0.05 (critical parameter) |
| Checkpoint | `models/checkpoints/lstm_phase2a_temp0.05_best.pth` |

### Portfolio Rules

| Parameter | Value |
|-----------|-------|
| Universe | ~189 liquid US equities |
| Long | Top 38 stocks (65% exposure, equal-weighted) |
| Short | Bottom 38 WHERE y_pred < 0 (35% exposure) |
| Position Smoothing | EWMA alpha=0.15 |
| Rebalance | Daily |
| Transaction Cost | 5 bps per trade |

### Risk Controls

| Parameter | Value |
|-----------|-------|
| Vol Target | 8% annual (20-day lookback) |
| Vol Scale Range | 0.5x to 2.0x |
| Kill Switch 1 | 3-sigma daily loss → flatten |
| Kill Switch 2 | 8% trailing drawdown → halt |
| Kill Switch 3 | Sharpe < 0 (60-day rolling) → disable |

---

## Model Architecture

### LSTM Model

```
Input: (batch, 90 days, 14 features)
  → LSTM Layer 1 (128 hidden, dropout=0.2)
  → LSTM Layer 2 (128 hidden, dropout=0.2)
  → Last time step output (128)
  → FC Layer (128 → 1)
Output: predicted next-day return
```

### Training

- **Training period**: 2020-01-01 to 2024-06-30
- **Validation period**: 2024-07-01 to 2024-12-31
- **Prediction period**: 2025-04-01 to 2025-10-31 (159 trading days)
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Early stopping**: Patience 15, min_delta 0.0001

### Loss Function

```python
loss = 0.7 * rank_loss + 0.3 * regression_loss

# Rank loss: differentiable Spearman correlation with temperature=0.05
rank_loss = 1 - soft_spearman(y_pred, y_true, temperature=0.05)

# Regression loss: Huber (robust to outliers)
regression_loss = HuberLoss(delta=0.05)
```

**Why this works**: Rank loss teaches relative ordering (long/short ranking). Regression loss calibrates magnitudes. Temperature 0.05 creates sharp rankings critical for performance.

### Model Performance

| Metric | Value |
|--------|-------|
| Validation Spearman | 0.120 |
| Information Coefficient (IC) | 0.112 |
| IC Information Ratio | 1.40 |
| Hit Rate | 52.3% |

---

## Feature Engineering

14 core features computed in `src/data/preprocessor_v2.py`:

| # | Feature | Description | Category |
|---|---------|-------------|----------|
| 1 | ret_1d | Daily return | Returns |
| 2 | ret_5d | 5-day return | Returns |
| 3 | ret_20d | 20-day return | Returns |
| 4 | volatility | 20-day realized vol (annualized) | Volatility |
| 5 | vol_5d | 5-day vol (annualized) | Volatility |
| 6 | vol_volume | Volume volatility (20-day) | Volatility |
| 7 | dist_from_high | Distance from 20-day high | Price Structure |
| 8 | dist_from_low | Distance from 20-day low | Price Structure |
| 9 | price_range | Intraday range / close | Price Structure |
| 10 | momentum | 20-day momentum | Trend |
| 11 | momentum_5d | 5-day momentum | Trend |
| 12 | volume_ratio | Volume / 20-day avg volume | Volume |
| 13 | volume_trend | 5-day avg / 20-day avg volume | Volume |
| 14 | market_return | Equal-weighted universe return | Market |

**Top features by importance**: ret_20d (18.5%), volatility (15.2%), volume_ratio (12.8%), ret_5d (11.3%), dist_from_high (9.7%)

**Design principles**:
- No look-ahead bias (all features use only past data)
- Z-score normalized per feature
- 90-day sequences as LSTM input
- Simplicity wins: 14 features > 40 features

---

## Portfolio Construction

### S2_FilterNegative Strategy

```python
# 1. Rank all stocks by prediction
sorted_stocks = predictions.sort_values('y_pred_reg', ascending=False)

# 2. Long: Top 38 stocks
longs = sorted_stocks.head(38)
long_weight = 0.65 / len(longs)  # ~1.71% each

# 3. Short: Bottom 38 WHERE prediction < 0 (CRITICAL FILTER)
short_candidates = sorted_stocks.tail(38)
shorts = short_candidates[short_candidates['y_pred_reg'] < 0]
short_weight = -0.35 / len(shorts) if len(shorts) > 0 else 0

# 4. EWMA smoothing to reduce turnover
smoothed = 0.15 * target + 0.85 * previous_positions
```

### Why the Short Filter Matters

| Strategy | Short Sharpe | Overall Sharpe |
|----------|-------------|----------------|
| S0: Naive shorts (all bottom 38) | -1.69 | 0.51 |
| **S2: FilterNegative (pred < 0 only)** | **+0.61** | **2.20** |

During bull markets, even bottom-ranked stocks may have positive predictions. Shorting them destroys value. The filter only shorts stocks genuinely expected to decline.

### Transaction Costs

```
Spread: 5 bps + Commission: 1 bps + Impact: 20 * turnover bps
Average daily cost: ~9.8 bps (at 49% turnover)
```

---

## Risk Management

### Volatility Targeting

Scales all positions to maintain 8% annualized volatility:

```python
realized_vol = returns[-20:].std() * sqrt(252)
vol_scale = clip(0.08 / realized_vol, 0.5, 2.0)
positions = {ticker: weight * vol_scale for ticker, weight in base_positions.items()}
```

- High vol periods → scale down (reduce risk)
- Low vol periods → scale up (maintain returns)

### Kill Switches

| Switch | Trigger | Action |
|--------|---------|--------|
| 3-sigma loss | Daily loss > 3 * historical std | Flatten all positions |
| 8% drawdown | Trailing drawdown > 8% | Halt trading |
| Negative Sharpe | 60-day rolling Sharpe < 0 | Disable strategy |

**Backtest results**: Kill switches triggered on 8% of days (15 events). No 8% drawdown events.

### Risk Metrics

- Market Beta: 0.03 (effectively market-neutral)
- Typical net exposure: 30% (long-biased)
- Gross exposure: 100%
- Max single position: ~2.6%
- Typical portfolio: 38 longs + 15-35 shorts

---

## Paper Trading (Phase 4)

### Overview

- **Method**: Historical replay using pre-computed predictions
- **Start date**: January 19, 2026
- **Duration**: 60+ trading days minimum
- **Goal**: Validate strategy before live deployment

### Daily Routine

Every trading day after 4:15 PM EST:

```bash
cd c:\Users\luixj\AI\stock-data
venv\Scripts\activate
python scripts/paper_trading/run_daily_paper_trading.py
```

The script automatically increments to the next historical date.

### Weekly Review

```bash
python scripts/paper_trading/phase4_performance_tracker.py
```

Generates reports in `reports/phase4/`.

### Success Criteria (must all be met for live deployment)

- Sharpe > 1.0
- Max drawdown < -10%
- Kill switch events < 15% of days
- No systematic issues
- 60+ days completed

### Red Flags (stop immediately)

- Sharpe < 0.5 for 2 consecutive weeks
- Max drawdown > -15%
- Kill switches > 25% of days
- Systematic errors or data issues

### After 60 Days

- Sharpe > 1.0 → Proceed to Phase 5 (live with 10% capital)
- Sharpe < 1.0 → DO NOT go live, investigate

---

## Automation

### Setup Windows Task Scheduler

Automates daily paper trading at 4:15 PM with Qdrant database storage.

**Prerequisites**:
```bash
pip install qdrant-client
docker run -d -p 6333:6333 --name qdrant-paper-trading qdrant/qdrant
```

**Setup** (run as Administrator):
```powershell
.\scripts\automation\setup_daily_task.ps1
```

Or use the batch file alternative:
```cmd
scripts\automation\setup_daily_task.bat
```

### Qdrant Collections

| Collection | Purpose |
|------------|---------|
| `stock_recommendations` | Daily long/short picks with vector embeddings |
| `trading_results` | Daily P&L, turnover, kill switch events |
| `performance_metrics` | Cumulative Sharpe, returns, drawdown |

### Query Database

```bash
# View latest stock recommendations
python scripts/automation/query_qdrant.py --type recommendations --limit 10

# View trading results
python scripts/automation/query_qdrant.py --type results --limit 5

# View performance metrics
python scripts/automation/query_qdrant.py --type performance

# Search similar stocks
python scripts/automation/query_qdrant.py --search AAPL
```

### Manual Run

```bash
python scripts/automation/daily_paper_trading_qdrant.py
```

### Check Automation Status

```powershell
# View scheduled task
Get-ScheduledTask -TaskName "DailyPaperTrading"

# Check last run
Get-ScheduledTaskInfo -TaskName "DailyPaperTrading"

# Run manually
Start-ScheduledTask -TaskName "DailyPaperTrading"
```

---

## Project History

### Phase 0: Baseline
- Baseline LSTM with MSE loss
- **Result**: Gross Sharpe 0.71, Net Sharpe -1.74 (120% turnover destroyed profits)

### Phase 1: Rank Loss
- Implemented cross-sectional rank loss (70% rank + 30% Huber)
- Simplified features from 40+ to 14
- **Result**: Reduced turnover but over-smoothed at temperature=1.0

### Phase 2A: Temperature Tuning
- Tested temperatures: 0.01, 0.05, 0.1, 0.5, 1.0
- **Result**: Temperature 0.05 optimal, Gross Sharpe 2.47

### Phase 2B: Portfolio Engineering
- Fixed 4 critical bugs in portfolio construction
- Tested 5 strategies (S0-S4)
- Added EWMA position smoothing
- **Result**: Net Sharpe 2.20 at 22% turnover
- **Validated**: Net Sharpe 1.32 at 100% forced turnover (alpha is real)

### Phase 3: Risk Analysis
- **3.1**: Strategy canonicalization (froze all parameters)
- **3.2**: Risk decomposition (market beta = 0.03)
- **3.3**: Portfolio comparison and transaction cost fixes
- **3.4**: Short salvage (S2_FilterNegative, short Sharpe: -1.69 → +0.61)
- **3.5**: Vol targeting (8% annual) + kill switches
- **Result**: Vol-Targeted Sharpe 1.29, GREEN LIGHT for paper trading

### Phase 4: Paper Trading (Current)
- Historical replay starting Jan 19, 2026
- Automated daily execution + Qdrant storage
- 60+ days required before live deployment

### Phase 5: Live Deployment (Future)
- Week 1-2: 10% capital
- Week 3-4: 25% capital (if Sharpe > 1.0)
- Week 5-8: 50% capital
- Week 9+: 100% capital
- Rollback: Any kill switch triggers twice in one week → reduce 50%

---

## Troubleshooting

### Common Issues

**"Module not found" errors**:
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

**"File not found: phase1_predictions.parquet"**:
```bash
ls -l data/processed/phase1_predictions.parquet
# If missing, run: python scripts/training/phase1_runner.py
```

**"KeyError: y_true_reg" or column name errors**:
```python
import pandas as pd
df = pd.read_parquet('data/processed/phase1_predictions.parquet')
print(df.columns.tolist())  # Check actual column names
```

**Unicode encoding errors**:
Ensure all file writes use `encoding='utf-8'`.

**Qdrant connection failed**:
```bash
docker ps | grep qdrant        # Check if running
docker start qdrant-paper-trading  # Restart if stopped
```

**Windows Task Scheduler not running**:
```powershell
Get-ScheduledTaskInfo -TaskName "DailyPaperTrading"
# Recreate if needed:
Unregister-ScheduledTask -TaskName "DailyPaperTrading" -Confirm:$false
.\scripts\automation\setup_daily_task.ps1
```

**Paper trading "No data for date"**:
Predictions only cover 2025-04-01 to 2025-10-31. Reset progress:
```python
import json
with open('data/processed/phase4/paper_trading_progress.json', 'w') as f:
    json.dump({'last_historical_date': '2025-04-01', 'days_completed': 0}, f)
```

### Key Lessons Learned

1. **Simplicity wins**: 14 features > 40 features
2. **Loss function matters**: Rank loss > MSE for trading
3. **Costs kill**: 120% turnover destroyed 0.71 gross Sharpe
4. **Temperature is critical**: 0.05 vs 1.0 is success vs failure
5. **Portfolio engineering != ML**: Smoothing increased Sharpe 67%
6. **Short filtering is essential**: Naive shorts have -1.69 Sharpe
7. **Validate everything**: Baseline stress test confirmed alpha is real

---

## Important Rules

**DO NOT**:
- Add more features or change model architecture
- Retrain the model or tune hyperparameters
- Change strategy parameters during paper trading
- Skip daily execution without documenting why

**The ML work is DONE. The strategy is FROZEN at v2.0.0.**

---

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss.

---

*Strategy v2.0.0 | Frozen 2026-01-18 | Phase 4 Paper Trading*
