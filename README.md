# Stock Trading Strategy - LSTM Cross-Sectional Alpha

**Status**: Phase 4 - Paper Trading (started Jan 19, 2026)
**Strategy**: S2_FilterNegative (130/70 long/short with negative prediction filter)
**Vol-Targeted Sharpe**: 1.29 | **Net Sharpe**: 2.20 | **Max DD**: -5.21%
**Live Predictions**: 2026-01-02 -> latest generated sim-end (actual Yahoo Finance data, LSTM inference)

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
   - [2026 Live Data Paper Trading](#2026-live-data-paper-trading)
   - [Batch Extension](#batch-extension)
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
If there is no new tradable date (for example, latest date has no `y_true_reg` yet), it exits cleanly with "No action taken" and prints a refresh hint.

Useful options:

```bash
# Manual historical date
python scripts/paper_trading/run_daily_paper_trading.py --historical-date 2026-02-19

# Use a specific predictions parquet
python scripts/paper_trading/run_daily_paper_trading.py --predictions data/processed/phase4/predictions_combined.parquet

# Skip per-day strategy save to Qdrant
python scripts/paper_trading/run_daily_paper_trading.py --skip-qdrant-strategy
```

### Auto-Generate Next Journal Entry (recommended)

```bash
# Default: generates tomorrow's pre-filled entry
python scripts/paper_trading/generate_journal_entry.py

# Preview only (no file write)
python scripts/paper_trading/generate_journal_entry.py --dry-run

# Explicit date
python scripts/paper_trading/generate_journal_entry.py --date 2026-02-22
```

### Run 2026 Live Data Paper Trading (recommended)

Downloads actual 2026 market data from Yahoo Finance, runs LSTM inference, and simulates the full 2025+2026 period. OHLCV data is cached after the first download:

```bash
python scripts/paper_trading/run_2026_paper_trading.py

# Skip Qdrant sync (local files only):
python scripts/paper_trading/run_2026_paper_trading.py --skip-qdrant

# Force re-download (bypass OHLCV cache):
python scripts/paper_trading/run_2026_paper_trading.py --force-download

# Skip all download+inference (use existing predictions_2026.parquet):
python scripts/paper_trading/run_2026_paper_trading.py --skip-download
```

See [2026 Live Data Paper Trading](#2026-live-data-paper-trading) for full details.

### Run Full Batch Simulation (2025 historical data only)

Re-runs the complete simulation over a date range in a single pass, then re-syncs all Qdrant collections:

```bash
python scripts/paper_trading/run_batch_extension.py
# defaults: --start-date 2025-04-01 --end-date 2025-12-31

# Custom range:
python scripts/paper_trading/run_batch_extension.py --start-date 2025-04-01 --end-date 2025-12-31

# Custom predictions file:
python scripts/paper_trading/run_batch_extension.py --predictions data/processed/phase4/predictions_combined.parquet

# Local files only (skip Qdrant):
python scripts/paper_trading/run_batch_extension.py --skip-qdrant
```

See [Batch Extension](#batch-extension) for full details.

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

```text
stock-data/
|-- config/
|   `-- config.yaml                  # Frozen model configuration
|
|-- src/                             # Core source code
|   |-- data/
|   |   |-- data_loader.py           # yfinance data download
|   |   `-- preprocessor_v2.py       # 14 feature engineering
|   |-- models/
|   |   |-- lstm_model.py            # 2-layer LSTM architecture
|   |   |-- losses.py                # Rank-regression loss
|   |   `-- trainer.py               # Training loop
|   `-- utils/
|       |-- config_loader.py         # Config management
|       `-- metrics.py               # Performance metrics
|
|-- scripts/
|   |-- training/                    # Phase 1 model training (DO NOT rerun)
|   |-- backtest/                    # Phase 2 portfolio backtesting
|   |-- validation/                  # Alpha validation tests
|   |-- experiment/                  # Phase 2A temperature experiments
|   |-- portfolio/                   # Portfolio construction tests
|   |-- risk/                        # Risk analysis scripts
|   |-- stress_test/                 # Stress testing
|   |-- deployment/                  # Deployment preparation
|   |-- paper_trading/               # Phase 4 daily execution
|   |   |-- run_2026_paper_trading.py       # Live 2026 data: download -> inference -> simulate
|   |   |-- run_batch_extension.py          # Batch runner: extend date range + Qdrant sync
|   |   |-- run_daily_paper_trading.py      # Daily runner (one day, historical replay)
|   |   |-- strategy_planner.py             # Build per-day BUY/SELL/HOLD/SHORT/COVER strategy payloads
|   |   |-- generate_journal_entry.py       # Append next-day pre-filled journal entry template
|   |   |-- phase4_paper_trading_runner.py  # Core trading engine
|   |   |-- phase4_performance_tracker.py   # Performance reporting
|   |   `-- phase4_daily_pipeline.py        # Live data pipeline (reference, has known bugs)
|   `-- automation/                  # Automated daily execution
|       |-- daily_paper_trading_qdrant.py   # Main automation + Qdrant
|       |-- query_qdrant.py                 # Database query tool
|       |-- setup_daily_task.ps1            # Windows Task Scheduler
|       `-- setup_daily_task.bat            # Alternative batch setup
|
|-- models/                          # Trained model checkpoints
|   `-- checkpoints/
|       `-- lstm_phase2a_temp0.05_best.pth # Production model (FROZEN)
|
|-- data/
|   |-- raw/                         # Downloaded stock data (CSV)
|   `-- processed/
|       |-- phase1_predictions.parquet      # Pre-computed 2025 predictions (2025-04-01->2025-12-29)
|       `-- phase4/                         # Paper trading results
|           |-- phase4_paper_trading_daily.parquet  # All simulation results (2025+2026)
|           |-- phase4_paper_trading_summary.json   # Cumulative performance stats
|           |-- paper_trading_progress.json         # Progress tracker
|           |-- predictions_2026.parquet            # Live 2026 LSTM predictions (generated)
|           |-- predictions_combined.parquet        # 2025+2026 merged predictions
|           |-- ohlcv_2026_cache.parquet            # Cached Yahoo Finance download
|           `-- sim_account.db                      # SQLite simulated account DB (created by stock-api)
|
|-- logs/
|   |-- paper_trading/                # Daily execution logs
|   `-- automation/                   # Automation logs
|
|-- reports/
|   `-- phase4/                       # Performance reports & plots
|
`-- archive/                          # Old/deprecated scripts
```

---

## Strategy Specification

**Version**: 2.0.0 (Frozen 2026-01-18) - NO CHANGES ALLOWED

### Signal Generation

| Parameter | Value |
|-----------|-------|
| Model | 2-layer LSTM (192 hidden units) |
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
| Kill Switch 1 | 3-sigma daily loss -> flatten |
| Kill Switch 2 | 8% trailing drawdown -> halt |
| Kill Switch 3 | Sharpe < 0 (60-day rolling) -> disable |

---

## Model Architecture

### LSTM Model

```
Input: (batch, 90 days, 14 features)
  -> LSTM Layer 1 (192 hidden, dropout=0.2)
  -> LSTM Layer 2 (192 hidden, dropout=0.2)
  -> Last time step output (192)
  -> FC Layer (192 -> 64 -> 1)
Output: predicted next-day return
```

> **Note**: The checkpoint uses `hidden_size=192` (confirmed from state_dict shape `lstm.weight_ih_l0: [768, 14]` where 768 = 4 x 192). Earlier documentation incorrectly stated 128.

### Training

- **Training period**: 2020-01-01 to 2024-06-30
- **Validation period**: 2024-07-01 to 2024-12-31
- **Historical predictions**: 2025-04-01 to 2025-12-29 (188 trading days, pre-computed)
- **Live predictions**: 2026-01-02 to latest generated `--sim-end` date (generated via Yahoo Finance + LSTM inference)
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
| 1 | `ret_1d` | Log daily return: log(close / close[-1]) | Returns |
| 2 | `ret_5d` | Log 5-day return: log(close / close[-5]) | Returns |
| 3 | `ret_20d` | Log 20-day return: log(close / close[-20]) | Returns |
| 4 | `vol_10d` | 10-day rolling std of ret_1d | Volatility |
| 5 | `vol_20d` | 20-day rolling std of ret_1d | Volatility |
| 6 | `hl_range` | (high - low) / close - intraday range | Price Structure |
| 7 | `oc_gap` | (open - prev_close) / prev_close - gap | Price Structure |
| 8 | `sma_10_dist` | (close - SMA10) / SMA10 | Trend |
| 9 | `sma_20_dist` | (close - SMA20) / SMA20 | Trend |
| 10 | `log_volume` | log(volume + 1) | Volume |
| 11 | `volume_change` | log((volume+1) / (prev_volume+1)) | Volume |
| 12 | `market_return` | Equal-weighted cross-sectional mean of ret_1d | Market |
| 13 | `vs_market` | ret_1d - market_return | Market |
| 14 | `market_correlation` | 20-day rolling correlation of ret_1d with market_return | Market |

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

- High vol periods -> scale down (reduce risk)
- Low vol periods -> scale up (maintain returns)

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

The script automatically increments to the next historical tradable date.
If no new tradable date exists, it prints "No new historical trading date available" and exits without error.
Each successful run also upserts a `daily_trade_strategies` record in Qdrant (unless `--skip-qdrant-strategy` is used).
By default, it prefers `data/processed/phase4/predictions_combined.parquet` when present; otherwise it falls back to `data/processed/phase1_predictions.parquet`.

After the run, auto-generate the next pre-filled journal block:

```bash
python scripts/paper_trading/generate_journal_entry.py
```

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

- Sharpe > 1.0 -> Proceed to Phase 5 (live with 10% capital)
- Sharpe < 1.0 -> DO NOT go live, investigate

### 2026 Live Data Paper Trading

`run_2026_paper_trading.py` extends paper trading into 2026 using **actual market data** downloaded from Yahoo Finance and **live LSTM inference** - not pre-computed predictions. It then re-runs the full combined simulation (`2025-04-01` -> `--sim-end`, default `auto` = yesterday) in one pass so vol-scaling state is continuous across the year boundary.

#### Pipeline steps

| Step | What happens |
|------|-------------|
| 1 | Download OHLCV for all tickers from Yahoo Finance (`2025-09-01` -> today) |
| 2 | Compute 14 core features via `SimplifiedStockPreprocessor.calculate_core_features()` |
| 3 | Load frozen `LSTMRegressor` checkpoint (`hidden_size=192`) |
| 4 | For each 2026 trading day: extract 90-day sequence -> run inference -> record `y_pred_reg`, `y_true_reg`, `close` |
| 5 | Save to `predictions_2026.parquet`; combine with `phase1_predictions.parquet` |
| 6 | Run full simulation `2025-04-01` -> `--sim-end` (one runner instance -> correct vol-scaling) |
| 7 | Clear and re-upload all Qdrant collections (`stock_recommendations`, `trading_results`, `performance_metrics`, `daily_trade_strategies`) |
| 8 | Update `paper_trading_progress.json` |

#### OHLCV Caching

Downloaded data is saved to `data/processed/phase4/ohlcv_2026_cache.parquet`. On every subsequent run the cache is reused automatically - the download only repeats if the cached max date is earlier than the requested `--sim-end` date (stale) or `--force-download` is passed.

```bash
# Normal run - uses cache if present
python scripts/paper_trading/run_2026_paper_trading.py

# Force a fresh Yahoo Finance download
python scripts/paper_trading/run_2026_paper_trading.py --force-download

# Skip all download+inference, re-simulate from existing predictions_2026.parquet
python scripts/paper_trading/run_2026_paper_trading.py --skip-download

# Skip Qdrant sync (local files only)
python scripts/paper_trading/run_2026_paper_trading.py --skip-qdrant

# Custom date range
python scripts/paper_trading/run_2026_paper_trading.py --sim-start 2026-01-02 --sim-end <latest-date>
```

#### Output files

| File | Content |
|------|---------|
| `data/processed/phase4/ohlcv_2026_cache.parquet` | Cached raw OHLCV (2025-09-01 -> today) |
| `data/processed/phase4/predictions_2026.parquet` | Live LSTM predictions for 2026 trading days |
| `data/processed/phase4/predictions_combined.parquet` | 2025 + 2026 merged predictions |
| `data/processed/phase4/phase4_paper_trading_daily.parquet` | Full simulation results (2025+2026) |
| `data/processed/phase4/phase4_paper_trading_summary.json` | Cumulative performance |
| `data/processed/phase4/paper_trading_progress.json` | Updated with 2026 last date |

### Sim Account Database (used by stock-api / stock-ui)

`stock-api` stores simulated account state in:

- `data/processed/phase4/sim_account.db`

This DB is created automatically on first account API access and is used by the UI `/account` page for:

- cash balance and total equity
- holdings and transaction history
- account-based 30D/90D/180D/1Y returns
- cash-flow-adjusted since-inception metrics

### Batch Extension

Use `run_batch_extension.py` to re-run the full simulation over a longer date range in one pass, rather than incrementally day-by-day. This is the correct way to extend the historical replay window because it carries vol-scaling and kill-switch state forward continuously through all days (the incremental daily runner resets state on each invocation, making vol-scale inaccurate for the first ~60 days of any new run).

```bash
# Full run: April 1 - December 31, 2025 (188 trading days)
python scripts/paper_trading/run_batch_extension.py

# Custom date range
python scripts/paper_trading/run_batch_extension.py \
    --start-date 2025-04-01 --end-date 2025-12-31

# Custom predictions file (e.g., combined 2025+2026)
python scripts/paper_trading/run_batch_extension.py \
    --predictions data/processed/phase4/predictions_combined.parquet \
    --start-date 2025-04-01 --end-date <latest-date>

# Regenerate local files only (skip Qdrant re-upload)
python scripts/paper_trading/run_batch_extension.py --skip-qdrant
```

**What it overwrites:**

| File | Action |
|---|---|
| `data/processed/phase4/phase4_paper_trading_daily.parquet` | Replaced with full date range |
| `data/processed/phase4/phase4_paper_trading_summary.json` | Recomputed over all days |
| `data/processed/phase4/paper_trading_progress.json` | Updated with new last date + day count |
| Qdrant `stock_recommendations` | Cleared and re-uploaded (all dates) |
| Qdrant `trading_results` | Cleared and re-uploaded (all dates) |
| Qdrant `performance_metrics` | Cleared and re-uploaded (final snapshot) |
| Qdrant `daily_trade_strategies` | Cleared and re-uploaded (all dates) |

**Qdrant point IDs** use a deterministic scheme (`YYYYMMDD x 10,000 + offset`) so re-running the script is idempotent - no duplicates are created.

**Date normalisation fix**: `phase4_paper_trading_runner.py` normalises the `date` column from the predictions parquet to plain `YYYY-MM-DD` strings immediately on load, so all date comparisons work regardless of whether the parquet stored dates as `datetime64` Timestamps or strings.

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
| `daily_trade_strategies` | Daily BUY/SELL/HOLD/SHORT/COVER plan with realized evaluation metrics |

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
- **3.4**: Short salvage (S2_FilterNegative, short Sharpe: -1.69 -> +0.61)
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
- Rollback: Any kill switch triggers twice in one week -> reduce 50%

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

**Qdrant `400 Bad Request: data did not match any variant of untagged enum VectorStruct`**:
A vector contains `NaN` or `Inf` - Qdrant rejects non-finite floats. Both `run_2026_paper_trading.py` and `run_batch_extension.py` now sanitize all vector values via `_safe_float()` which replaces `NaN`/`Inf` with `0.0`. If this appears on an older script, verify the `_safe_float` helper is present.

**2026 OHLCV cache is stale or wrong data**:
```bash
# Force a fresh download and regenerate all predictions
python scripts/paper_trading/run_2026_paper_trading.py --force-download
```
The cache file is at `data/processed/phase4/ohlcv_2026_cache.parquet`. Delete it manually to force a re-download on next run.

**2026 predictions don't appear in stock-ui**:
Ensure the full pipeline ran (not just `--skip-download`). Check that `predictions_combined.parquet` exists and Qdrant sync completed. The simulation must cover 2026 dates for them to appear as `trading_results` in Qdrant.

**Paper trading "No data for date"**:
`run_daily_paper_trading.py` now has two behaviors:

- Automatic mode (no `--historical-date`): if progress is already at the latest tradable date, it prints "No new historical trading date available" and exits with no action.
- Manual mode (`--historical-date`): it errors if the requested date is not tradable (missing from predictions or missing `y_true_reg`).
- Daily strategy persistence: after a successful run, the script upserts one strategy payload into Qdrant `daily_trade_strategies` (disable via `--skip-qdrant-strategy`).

If you're out of tradable dates, refresh predictions first:
```bash
python scripts/paper_trading/run_2026_paper_trading.py --sim-start <next-date> --sim-end auto
```

If you want to replay from an earlier date, reset progress:
```python
import json
with open('data/processed/phase4/paper_trading_progress.json', 'w') as f:
    json.dump({'last_historical_date': '2025-04-01', 'days_completed': 0}, f)
```

Or re-run the full batch to rebuild everything from scratch:
```bash
python scripts/paper_trading/run_batch_extension.py
```

**Journal entry auto-generator duplicates date**:
`generate_journal_entry.py` prevents duplicate date blocks by default. If an entry already exists, it prints a message and exits. Use `--force` only when you intentionally want another block for the same date.

**TypeError comparing Timestamp with str in `run_backtest_simulation`**:
Fixed in `phase4_paper_trading_runner.py` - the `date` column is now normalised to plain `YYYY-MM-DD` strings immediately after loading the parquet file. If this error reappears, check that the fix at line 68 is present:
```python
self.predictions_df['date'] = pd.to_datetime(self.predictions_df['date']).dt.strftime('%Y-%m-%d')
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

*Strategy v2.0.0 | Frozen 2026-01-18 | Phase 4 Paper Trading | Live predictions through latest generated sim-end*







