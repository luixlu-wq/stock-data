# Stock Trading Strategy - Project Structure

## Overview

This project implements an institutional-quality LSTM-based stock trading strategy with cross-sectional alpha. The strategy has been validated with Net Sharpe 2.20 and is ready for Phase 3 (risk management and deployment).

## Directory Structure

```
stock-data/
├── config/                 # Configuration files
│   └── config.yaml        # Model, training, and backtest configuration
│
├── data/                  # Data storage
│   ├── raw/              # Raw stock price data
│   ├── processed/        # Preprocessed features and results
│   └── embeddings/       # (unused - for future embedding features)
│
├── docs/                 # Documentation
│   ├── FINAL_RESULTS.md         # Complete project summary and results
│   ├── PHASE1_README.md         # Phase 1: Rank Loss implementation
│   ├── PHASE1_SUMMARY.md        # Phase 1 results summary
│   ├── PHASE2A_README.md        # Phase 2A: Temperature tuning
│   ├── PHASE2A_VALIDATED.md     # Phase 2A validation results
│   ├── PHASE2B_README.md        # Phase 2B: Portfolio engineering
│   └── PHASE2B_BUGFIXES.md      # Critical bugs fixed in Phase 2B
│
├── logs/                 # Training and backtest logs
│   ├── phase1_training.log
│   ├── phase2a_temperature_experiment.log
│   ├── phase2b_monetization.log
│   └── phase2b_true_baseline.log
│
├── models/               # Saved model checkpoints
│   └── checkpoints/
│       └── lstm_phase2a_temp0.05_best.pth  # Final trained model (FROZEN)
│
├── notebooks/            # Jupyter notebooks for analysis
│
├── scripts/              # Executable scripts
│   ├── training/        # Model training scripts
│   │   ├── phase1_train.py                    # Phase 1: Rank loss training
│   │   └── phase2a_temperature_experiment.py  # Phase 2A: Temperature tuning
│   │
│   ├── backtest/        # Backtesting scripts
│   │   ├── phase0_backtest.py       # Original baseline backtest
│   │   └── phase2b_monetization.py  # Portfolio engineering backtest
│   │
│   └── validation/      # Validation scripts
│       └── phase2b_validate_baseline.py  # Baseline validation (100% turnover test)
│
├── src/                  # Core library code
│   ├── data/            # Data processing
│   │   ├── downloader.py        # Stock data downloader
│   │   ├── preprocessor.py      # Feature engineering (14 core features)
│   │   └── yahoo_downloader.py  # Yahoo Finance data source
│   │
│   ├── database/        # Database interfaces (unused currently)
│   │   ├── embeddings.py
│   │   └── qdrant_client.py
│   │
│   ├── models/          # Model architectures and training
│   │   ├── dataset.py         # PyTorch dataset classes
│   │   ├── losses.py          # Rank loss + combined loss functions
│   │   ├── lstm_model.py      # LSTM model architecture
│   │   └── trainer.py         # Model training loop
│   │
│   └── utils/           # Utility functions
│       ├── config_loader.py   # YAML config loader
│       └── logger.py          # Logging setup
│
├── archive/              # Archived obsolete files
│   ├── old_docs/        # Old documentation from Phase 0
│   ├── old_scripts/     # Debug and obsolete scripts
│   ├── old_logs/        # Old training logs
│   └── old_preprocessors/  # preprocessor_v2.py, preprocessor_v3.py
│
├── .env                  # Environment variables (API keys)
├── .gitignore           # Git ignore rules
├── README.md            # Quick start guide
├── requirements.txt     # Python dependencies
└── PROJECT_STRUCTURE.md # This file
```

## Key Files

### Core Implementation

**Final Model (FROZEN)**:
- [models/checkpoints/lstm_phase2a_temp0.05_best.pth](models/checkpoints/lstm_phase2a_temp0.05_best.pth) - Production model

**Feature Engineering**:
- [src/data/preprocessor.py](src/data/preprocessor.py) - 14 core features (returns, volatility, price structure, trend, volume, market)

**Loss Functions**:
- [src/models/losses.py](src/models/losses.py) - Combined rank-regression loss (70% rank + 30% Huber, temp=0.05)

**Portfolio Engineering**:
- [scripts/backtest/phase2b_monetization.py](scripts/backtest/phase2b_monetization.py) - Position smoothing (EWMA α=0.15)

### Documentation

**Must Read**:
- [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) - Complete journey, results, and next steps
- [README.md](README.md) - Quick start guide

**Phase Documentation**:
- [docs/PHASE1_README.md](docs/PHASE1_README.md) - Rank loss implementation
- [docs/PHASE2A_VALIDATED.md](docs/PHASE2A_VALIDATED.md) - Temperature tuning results
- [docs/PHASE2B_BUGFIXES.md](docs/PHASE2B_BUGFIXES.md) - Critical portfolio engineering bugs

### Execution Scripts

**Training** (DO NOT run - model is FROZEN):
- [scripts/training/phase1_train.py](scripts/training/phase1_train.py)
- [scripts/training/phase2a_temperature_experiment.py](scripts/training/phase2a_temperature_experiment.py)

**Backtesting**:
- [scripts/backtest/phase2b_monetization.py](scripts/backtest/phase2b_monetization.py) - Test portfolio configurations
- [scripts/validation/phase2b_validate_baseline.py](scripts/validation/phase2b_validate_baseline.py) - Validate alpha is real

## Configuration

**Config file**: [config/config.yaml](config/config.yaml)

Key settings (FROZEN for Phase 2):
```yaml
model:
  type: "regression"
  architecture: "lstm"
  sequence_length: 90
  hidden_size: 128
  num_layers: 2

  loss:
    regression: "rank"
    rank_temperature: 0.05
    rank_weight: 0.7
    huber_delta: 0.05
```

## Phase History

### Phase 0: Alpha Discovery
- Baseline LSTM with MSE loss
- Result: Gross Sharpe 0.71, Net Sharpe -1.74
- Problem: 120% turnover destroyed profits

### Phase 1: Rank Loss Implementation
- Introduced cross-sectional rank loss (70% rank + 30% Huber)
- Simplified to 14 core features
- Result: Reduced turnover to 95%, but Gross Sharpe dropped to 0.55

### Phase 2A: Temperature Calibration
- Tested 6 temperatures [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
- Found optimal temperature: 0.05
- Result: Gross Sharpe 2.47, Net Sharpe 0.10

### Phase 2B: Portfolio Engineering
- Fixed 4 critical bugs in portfolio logic
- Implemented position smoothing (EWMA α=0.15)
- Validated baseline with 100% turnover stress test
- Final Result: **Net Sharpe 2.20, 22% turnover**

## Current Status

**Model**: FROZEN - DO NOT RETRAIN
**Strategy**: TRADEABLE (Net Sharpe 2.20)
**Alpha Validation**: CONFIRMED (Net Sharpe 1.32 at 100% turnover)

## Next Phase: Phase 3 - Risk Management & Deployment

**Planned Work**:
1. Volatility targeting (risk normalization)
2. Sector neutrality (reduce factor exposure)
3. Sub-period analysis (rolling Sharpe, IC stability)
4. Capacity estimation
5. Slippage modeling
6. Paper trading setup

**What NOT to do**:
- DO NOT add more features
- DO NOT change model architecture
- DO NOT retrain the model
- DO NOT tune more hyperparameters

The ML work is complete. Focus is now on risk management and deployment.

## Data Flow

```
Raw Data (Yahoo Finance)
    ↓
Preprocessor (14 features, 90-day sequences)
    ↓
LSTM Model (2-layer, 128 hidden, temp=0.05)
    ↓
Predictions (daily stock rankings)
    ↓
Portfolio Construction (top/bottom 20%, equal-weighted)
    ↓
Position Smoothing (EWMA α=0.15)
    ↓
Dollar Neutrality Enforcement
    ↓
Execution & PnL Calculation
```

## Performance Metrics

**Best Configuration** (Position Smoothing Only):
- Net Sharpe: 2.20
- Gross Sharpe: 2.47
- Turnover: 22% daily
- Max Drawdown: < -6%
- IC Mean: ~0.04-0.06
- IC IR: > 0.3

**Baseline Validation** (100% daily turnover):
- Net Sharpe: 1.32
- Confirms alpha is real and robust

## Archive

The [archive/](archive/) directory contains:
- Old documentation from early debugging phases
- Obsolete training and debug scripts
- Unused preprocessor versions (v2, v3)
- Old log files

These files are preserved for historical reference but are not part of the current workflow.

## Getting Started

See [README.md](README.md) for setup instructions.

For understanding the complete journey, read [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md).
