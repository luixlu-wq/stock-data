# System Architecture

**Technical design and structure of the ML trading system**

## Overview

This is a complete end-to-end machine learning trading system consisting of:
1. Data pipeline (download → preprocess → features)
2. Model training (LSTM with custom loss)
3. Backtesting engine (portfolio simulation)
4. Risk management (volatility targeting + kill switches)
5. Paper trading (automated execution)
6. Storage and monitoring (Qdrant + reporting)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ yfinance API │→ │ Raw CSV Data │→ │ Preprocessor │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                            ↓                  │
│                            ┌───────────────────────┐         │
│                            │ 14 Core Features      │         │
│                            │ (90-day sequences)    │         │
│                            └───────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                     MODEL LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  LSTM Model  │→ │  Training    │→ │  Predictions │      │
│  │  (2 layers)  │  │  Pipeline    │  │  (parquet)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│       ↓                                                       │
│  Temperature=0.05 (Rank + Regression Loss)                   │
└─────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                   STRATEGY LAYER                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Portfolio Constructor (S2_FilterNegative)        │       │
│  │  - Rank stocks by prediction                     │       │
│  │  - Long top K (38), Short bottom K (filtered)    │       │
│  │  - EWMA smoothing (α=0.15)                       │       │
│  └──────────────────────────────────────────────────┘       │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Risk Management                                  │       │
│  │  - Volatility targeting (8% annual)              │       │
│  │  - Kill switches (3-sigma, DD, Sharpe)          │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                  EXECUTION LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Paper Trade  │→ │  Simulated   │→ │   Results    │      │
│  │  Runner      │  │  Execution   │  │   Tracking   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  Daily Automation (4:15 PM) → Qdrant Storage                │
└─────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                   STORAGE LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Qdrant DB  │  │   Parquet    │  │    Logs      │      │
│  │  (vectors)   │  │   Files      │  │   (JSON)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
stock-data/
├── src/                           # Core source code
│   ├── data/
│   │   ├── data_loader.py        # yfinance data download
│   │   └── preprocessor_v2.py    # Feature engineering
│   ├── models/
│   │   ├── lstm_model.py         # LSTM architecture
│   │   └── trainer.py            # Training loop
│   └── utils/
│       ├── config_loader.py      # Config management
│       └── metrics.py            # Performance metrics
│
├── scripts/                       # Executable scripts
│   ├── phase1/                   # Model training
│   │   └── phase1_runner.py     # End-to-end training
│   ├── phase2b/                  # Backtesting
│   │   ├── phase2b_runner.py    # Strategy backtests
│   │   └── phase2b_strategies.py # Portfolio constructors
│   ├── paper_trading/            # Live simulation
│   │   ├── run_daily_paper_trading.py
│   │   ├── phase4_paper_trading_runner.py
│   │   └── phase4_performance_tracker.py
│   └── automation/               # Automated execution
│       ├── daily_paper_trading_qdrant.py
│       ├── query_qdrant.py
│       └── setup_daily_task.ps1
│
├── data/                          # Data storage
│   ├── raw/                      # Downloaded CSVs
│   ├── processed/
│   │   ├── phase1_predictions.parquet  # Model outputs
│   │   └── phase4/               # Paper trading results
│   └── external/                 # Manual data
│
├── models/                        # Trained models
│   ├── best_lstm_model.pth       # Production model
│   └── checkpoints/              # Training checkpoints
│
├── config/                        # Configuration
│   └── config.yaml               # System config
│
├── docs/                          # Documentation
│   ├── guides/                   # User guides
│   ├── phase_results/            # Historical results
│   └── technical/                # Technical docs (you are here)
│
├── logs/                          # Runtime logs
│   ├── paper_trading/
│   │   ├── JOURNAL.md            # Daily notes
│   │   └── paper_trading_*.log  # Execution logs
│   └── automation/               # Automation logs
│
└── reports/                       # Generated reports
    └── phase4/                   # Paper trading reports
```

## Component Details

### 1. Data Pipeline

**Purpose**: Download, clean, and engineer features from raw stock data

**Key Files**:
- [src/data/data_loader.py](../../src/data/data_loader.py) - Downloads OHLCV data from yfinance
- [src/data/preprocessor_v2.py](../../src/data/preprocessor_v2.py) - Creates 14 core features

**Data Flow**:
```
yfinance API
    ↓
raw CSV files (data/raw/tickers/{TICKER}.csv)
    ↓
SimplifiedStockPreprocessor.calculate_core_features()
    ↓
14 features per stock per day (90-day sequences)
    ↓
Training data (X, y) or prediction data
```

**Core Features** (see [FEATURES.md](FEATURES.md) for details):
- Returns: 1d, 5d, 20d
- Volatility: 5d, 20d, volume volatility
- Price structure: distance from high/low, range compression
- Trend: 5d, 20d momentum
- Volume: ratio, trend
- Market context: market return

### 2. Model Layer

**Purpose**: Train LSTM to predict next-day stock returns

**Key Files**:
- [src/models/lstm_model.py](../../src/models/lstm_model.py) - LSTM architecture
- [src/models/trainer.py](../../src/models/trainer.py) - Training loop and loss functions
- [scripts/phase1/phase1_runner.py](../../scripts/phase1/phase1_runner.py) - End-to-end training

**Architecture**:
```python
StockLSTM(
    input_size=14,      # 14 core features
    hidden_size=128,    # LSTM hidden dimension
    num_layers=2,       # Stacked LSTM layers
    dropout=0.2         # Regularization
)
```

**Loss Function**:
```python
loss = 0.7 * rank_loss + 0.3 * regression_loss

# Rank loss: Differentiable Spearman correlation
rank_loss = 1 - spearman_corr(y_pred, y_true, temperature=0.05)

# Regression loss: Huber (robust to outliers)
regression_loss = HuberLoss(delta=0.05)
```

**Why this works**:
- Rank loss → Model learns relative ordering (long/short ranking)
- Regression loss → Model calibrated to actual return magnitudes
- Temperature 0.05 → Sharp rankings (critical for performance)

**Training Process**:
1. Load 90-day sequences of 14 features
2. Train with combined rank-regression loss
3. Validate on out-of-sample period
4. Save best model checkpoint
5. Generate predictions for future period (2025-04-01 to 2025-10-31)

### 3. Strategy Layer

**Purpose**: Convert predictions to portfolio positions

**Key Files**:
- [scripts/phase2b/phase2b_strategies.py](../../scripts/phase2b/phase2b_strategies.py) - Portfolio constructors
- [scripts/phase2b/phase2b_runner.py](../../scripts/phase2b/phase2b_runner.py) - Backtesting engine

**Portfolio Construction (S2_FilterNegative)**:
```python
def construct_portfolio(predictions):
    # Sort by prediction
    sorted_stocks = sort_by_prediction(predictions)

    # Long: Top K stocks
    longs = sorted_stocks[:K]  # K=38
    long_weight = 0.65 / len(longs)  # 65% exposure

    # Short: Bottom K stocks, FILTERED by prediction < 0
    short_candidates = sorted_stocks[-K:]
    shorts = [s for s in short_candidates if s.prediction < 0]
    short_weight = -0.35 / len(shorts) if shorts else 0

    # Apply EWMA smoothing to reduce turnover
    positions = ewma_smooth(
        target_positions,
        previous_positions,
        alpha=0.15
    )

    return positions
```

**Why the filter matters**:
- During bull markets, bottom K stocks may have positive predictions
- Shorting stocks expected to go up destroys performance
- Filter salvages short book: -1.69 Sharpe → +0.61 Sharpe

**Transaction Costs**:
```python
def calculate_cost(turnover):
    spread_bps = 5      # Bid-ask spread
    commission_bps = 1  # Commission
    impact_bps = turnover * 20  # Market impact (linear in turnover)

    total_cost = spread_bps + commission_bps + impact_bps
    return total_cost / 10000  # Convert bps to decimal
```

### 4. Risk Management Layer

**Purpose**: Control volatility and prevent catastrophic losses

**Key Files**:
- [scripts/paper_trading/phase4_paper_trading_runner.py](../../scripts/paper_trading/phase4_paper_trading_runner.py) - Risk checks

**Volatility Targeting**:
```python
def calculate_vol_scale(returns_history, target_vol=0.08):
    # 20-day realized volatility
    realized_vol = returns_history[-20:].std() * sqrt(252)

    # Scale factor (constrained to 0.5-2.0x)
    scale = target_vol / realized_vol
    scale = clip(scale, 0.5, 2.0)

    return scale

# Apply to positions
scaled_positions = positions * vol_scale
```

**Kill Switches**:
```python
def check_kill_switches(daily_pnl, cumulative_returns, sharpe_60d):
    # 3-sigma daily loss
    if daily_pnl < -3 * historical_vol:
        return True, "3-sigma loss"

    # 8% trailing drawdown
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / (1 + peak)
    if drawdown.min() < -0.08:
        return True, "8% drawdown"

    # Sharpe < 0 (60-day rolling)
    if sharpe_60d < 0:
        return True, "Negative Sharpe"

    return False, None

# If kill switch triggered: flatten all positions
if triggered:
    positions = {}  # Go to cash
```

### 5. Execution Layer (Paper Trading)

**Purpose**: Simulate daily trading without real money

**Key Files**:
- [scripts/paper_trading/run_daily_paper_trading.py](../../scripts/paper_trading/run_daily_paper_trading.py) - Simple daily runner
- [scripts/paper_trading/phase4_paper_trading_runner.py](../../scripts/paper_trading/phase4_paper_trading_runner.py) - Full engine
- [scripts/paper_trading/phase4_performance_tracker.py](../../scripts/paper_trading/phase4_performance_tracker.py) - Reporting

**Daily Execution Flow**:
```
1. Load predictions for date T
2. Construct target portfolio (S2_FilterNegative)
3. Calculate trades needed (target - current)
4. Apply EWMA smoothing
5. Check kill switches
6. Apply volatility scaling
7. Calculate transaction costs
8. Calculate P&L using actual returns
9. Update cumulative metrics
10. Save results
```

**Output Files**:
- `data/processed/phase4/phase4_paper_trading_daily.parquet` - Daily trades/PnL
- `data/processed/phase4/phase4_paper_trading_summary.json` - Cumulative metrics
- `data/processed/phase4/paper_trading_progress.json` - Progress tracker
- `logs/paper_trading/paper_trading_YYYYMMDD.log` - Execution log

### 6. Automation Layer

**Purpose**: Run paper trading automatically every day

**Key Files**:
- [scripts/automation/daily_paper_trading_qdrant.py](../../scripts/automation/daily_paper_trading_qdrant.py) - Main automation
- [scripts/automation/setup_daily_task.ps1](../../scripts/automation/setup_daily_task.ps1) - Task scheduler setup
- [scripts/automation/query_qdrant.py](../../scripts/automation/query_qdrant.py) - Query tool

**Automation Flow**:
```
Windows Task Scheduler (4:15 PM daily)
    ↓
daily_paper_trading_qdrant.py
    ↓
1. Get next historical date
2. Run paper trading
3. Get stock recommendations
4. Save to Qdrant:
   - Stock recommendations (with vectors)
   - Trading results
   - Performance metrics
5. Update progress file
```

**Qdrant Collections**:
1. **stock_recommendations**: Daily long/short picks with embeddings for similarity search
2. **trading_results**: Daily P&L, turnover, kill switch events
3. **performance_metrics**: Cumulative Sharpe, returns, drawdown

**Vector Embeddings**:
```python
# Stock recommendation vector (dimension 10)
[
    weight,           # Position size
    prediction,       # Expected return
    actual_return,    # Realized return
    price / 1000,     # Normalized price
    position_type,    # 1.0 (long) or -1.0 (short)
    0, 0, 0, 0, 0    # Padding
]
```

### 7. Storage Layer

**Purpose**: Persist results for analysis and monitoring

**Storage Types**:
1. **Parquet Files** (data/processed/):
   - Efficient columnar storage
   - Fast filtering and aggregation
   - Used for predictions and daily results

2. **Qdrant Vector DB**:
   - Similarity search capabilities
   - Structured metadata storage
   - Real-time querying

3. **JSON Files** (logs/, data/):
   - Summary statistics
   - Progress tracking
   - Configuration

4. **Text Logs** (logs/):
   - Execution logs
   - Error tracking
   - Daily journal

## Data Flow (End-to-End)

```
Historical Data (2020-2024)
    ↓
[Phase 0: Baseline]
    ↓
[Phase 1: LSTM Training] → models/best_lstm_model.pth
    ↓
predictions (2025-04-01 to 2025-10-31) → phase1_predictions.parquet
    ↓
[Phase 2: Backtesting] → Test 5 strategies
    ↓
S2_FilterNegative selected (Sharpe 2.20)
    ↓
[Phase 3: Risk Analysis] → Add vol targeting + kill switches
    ↓
Production config frozen (STRATEGY_DEFINITION.md v2.0.0)
    ↓
[Phase 4: Paper Trading] ← YOU ARE HERE
    ↓
Daily: Load prediction → Construct portfolio → Simulate → Save
    ↓
Qdrant DB + Reports + Logs
    ↓
[Phase 5: Live Deployment] (Future)
```

## Key Design Decisions

### 1. Why Pre-computed Predictions?
- Faster paper trading (no inference overhead)
- Reproducibility (same predictions for all tests)
- Focus on strategy validation (not model performance)

### 2. Why Historical Replay vs Live Data?
- Validates process without data download complexity
- Tests full system end-to-end
- Safer for initial validation (known data quality)

### 3. Why Qdrant for Storage?
- Vector similarity search (find similar stocks)
- Efficient filtering and querying
- Professional-grade database
- Easy integration with Python

### 4. Why Daily Rebalancing?
- Predictions are for next-day returns
- Higher frequency → more alpha capture
- Turnover controlled by EWMA smoothing

### 5. Why EWMA Smoothing?
- Reduces turnover (49% vs 80%+ without)
- Reduces costs (critical for profitability)
- Minimal performance degradation

## Performance Characteristics

**Backtested Performance (Phase 2B, S2_FilterNegative)**:
- Sharpe Ratio: 2.20 (net of costs)
- Annual Return: 22.4%
- Volatility: 10.2%
- Max Drawdown: -6.3%
- Average Turnover: 49.1%
- Average Cost: 9.8 bps

**With Vol Targeting (Phase 3.5)**:
- Vol-Targeted Sharpe: 1.29
- Annual Return: 11.95%
- Volatility: 9.27%
- Max Drawdown: -5.21%

## System Requirements

**Development**:
- Python 3.8+
- 8GB RAM minimum
- GPU recommended for training (CUDA)
- 10GB disk space

**Production (Paper Trading)**:
- Python 3.8+
- 4GB RAM
- CPU only (no GPU needed)
- Docker for Qdrant
- Windows Task Scheduler or cron

**Dependencies**:
- PyTorch (model training/inference)
- pandas (data manipulation)
- yfinance (data download)
- qdrant-client (vector database)
- scikit-learn (metrics)
- matplotlib (visualization)

## Extension Points

### Adding New Features
1. Edit `src/data/preprocessor_v2.py`
2. Update input_size in model
3. Retrain model (Phase 1)
4. Re-run backtests (Phase 2)

### Adding New Strategies
1. Add to `scripts/phase2b/phase2b_strategies.py`
2. Update runner to include new strategy
3. Run backtests
4. Compare against S2_FilterNegative baseline

### Modifying Risk Management
1. Edit kill switch logic in `phase4_paper_trading_runner.py`
2. Test with historical data
3. Validate impact on Sharpe/drawdown

### Adding Live Data
1. Use `scripts/phase4/phase4_daily_pipeline.py` (already exists)
2. Modify to download today's data
3. Run inference with trained model
4. Feed predictions to paper trading runner

## Monitoring and Observability

**Logs**:
- `logs/paper_trading/paper_trading_YYYYMMDD.log` - Daily execution
- `logs/automation/daily_automation_YYYYMMDD.log` - Automation logs

**Reports**:
- `reports/phase4/phase4_performance_report.txt` - Cumulative metrics
- `reports/phase4/phase4_performance_plots.png` - Visualizations

**Queries**:
```bash
# View latest recommendations
python scripts/automation/query_qdrant.py --type recommendations

# View performance
python scripts/automation/query_qdrant.py --type performance

# Search similar stocks
python scripts/automation/query_qdrant.py --search AAPL
```

---

**See Also**:
- [FEATURES.md](FEATURES.md) - Feature engineering details
- [MODEL_DETAILS.md](MODEL_DETAILS.md) - LSTM architecture and training
- [PORTFOLIO_LOGIC.md](PORTFOLIO_LOGIC.md) - Portfolio construction rules
- [RISK_FRAMEWORK.md](RISK_FRAMEWORK.md) - Risk management system

**Last Updated**: January 20, 2026
