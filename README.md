# Stock Trading Strategy - LSTM Cross-Sectional Alpha

**Status**: ✅ **TRADEABLE STRATEGY VALIDATED**

An institutional-quality LSTM-based stock trading strategy that has been rigorously validated through systematic experimentation. The strategy achieves **Net Sharpe 2.20** with 22% turnover using cross-sectional rank loss and position smoothing.

## Quick Facts

- **Net Sharpe Ratio**: 2.20 (with position smoothing)
- **Gross Sharpe Ratio**: 2.47
- **Daily Turnover**: 22%
- **Alpha Validation**: Confirmed real (Net Sharpe 1.32 at 100% turnover)
- **Model Status**: FROZEN - production ready
- **Strategy Type**: Medium-horizon cross-sectional ranking alpha

## Key Innovation

This project successfully transformed a failing LSTM strategy (Net Sharpe -1.74) into an institutional-quality alpha signal through:

1. **Cross-sectional rank loss** (70% rank + 30% Huber) instead of MSE
2. **Temperature calibration** (0.05) for sharp rankings
3. **Portfolio engineering** (EWMA position smoothing α=0.15)
4. **Rigorous validation** (100% turnover stress test)

## Project Journey

| Phase | Focus | Result |
|-------|-------|--------|
| **Phase 0** | Alpha Discovery | Gross Sharpe 0.71, Net -1.74 (120% turnover) |
| **Phase 1** | Rank Loss Implementation | Reduced turnover but over-smoothed |
| **Phase 2A** | Temperature Calibration | Found optimal temp=0.05 (Gross 2.47) |
| **Phase 2B** | Portfolio Engineering | Net Sharpe 2.20 (22% turnover) |
| **Validation** | Baseline Stress Test | Confirmed real alpha (Net 1.32 @ 100% turnover) |

Complete journey documented in [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md).

## Quick Start

### Prerequisites

- Python 3.8+ (3.13 recommended)
- CUDA-capable GPU (RTX 5090 supported)
- 8GB+ GPU memory recommended

### Installation

```bash
# Clone and setup environment
cd c:\Users\luixj\AI\stock-data
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (RTX 5090 requires CUDA 12.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For detailed installation instructions, see [Installation Guide](#installation-guide) below.

### Run Backtest

The model is already trained and frozen. To verify the strategy:

```bash
# Run portfolio monetization backtest
python scripts/backtest/phase2b_monetization.py

# Validate baseline (100% turnover stress test)
python scripts/validation/phase2b_validate_baseline.py
```

## Strategy Specification

### Model (FROZEN - DO NOT CHANGE)

- **Architecture**: 2-layer LSTM (128 hidden units)
- **Features**: 14 core features (returns, volatility, price structure, trend, volume, market)
- **Sequence Length**: 90 days
- **Loss Function**: Combined rank-regression (70% rank + 30% Huber)
- **Temperature**: 0.05 (sharp rankings)
- **Checkpoint**: `models/checkpoints/lstm_phase2a_temp0.05_best.pth`

### Portfolio Construction

- **Universe**: 189 stocks
- **Long/Short**: Top/Bottom 20% by predicted return
- **Weights**: Equal-weighted within buckets
- **Position Smoothing**: EWMA with α=0.15
  - `pos_t = 0.85 * pos_{t-1} + 0.15 * target_t`
- **Rebalance**: Daily
- **Dollar Neutrality**: Enforced

### Performance Metrics

**Best Configuration** (Position Smoothing Only):
- Net Sharpe: 2.20
- Gross Sharpe: 2.47
- Turnover: 22% daily
- Max Drawdown: < -6%

**Baseline Validation** (100% daily turnover):
- Net Sharpe: 1.32
- Confirms alpha is real and robust

## Project Structure

```
stock-data/
├── config/                 # Configuration
│   └── config.yaml        # Model and backtest settings (FROZEN)
│
├── scripts/               # Executable scripts
│   ├── training/         # Phase 1 & 2A training (DO NOT rerun)
│   ├── backtest/         # Portfolio backtesting
│   └── validation/       # Alpha validation tests
│
├── src/                  # Core library
│   ├── data/            # Feature engineering (14 features)
│   ├── models/          # LSTM + rank loss implementation
│   └── utils/           # Config and logging
│
├── models/               # Saved checkpoints
│   └── checkpoints/
│       └── lstm_phase2a_temp0.05_best.pth  # Production model
│
├── docs/                 # Documentation
│   ├── FINAL_RESULTS.md         # Complete project summary ⭐
│   ├── PHASE1_README.md         # Rank loss implementation
│   ├── PHASE2A_VALIDATED.md     # Temperature tuning
│   └── PHASE2B_BUGFIXES.md      # Portfolio engineering fixes
│
├── data/                 # Data storage
│   ├── raw/             # Stock price data
│   └── processed/       # Features and results
│
└── logs/                 # Execution logs
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure.

## Installation Guide

### 1. System Requirements

**GPU (Highly Recommended)**:
- NVIDIA GPU with CUDA support
- RTX 5090: Requires PyTorch 2.7+ with CUDA 12.8
- RTX 40 series: PyTorch with CUDA 12.4
- Older GPUs: PyTorch with CUDA 11.8

**Memory**:
- 8GB+ GPU memory recommended
- 16GB+ system RAM

### 2. Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install PyTorch

**For RTX 5090 (CUDA 12.8 required)**:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**For RTX 40 series (CUDA 12.4)**:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For older GPUs (CUDA 11.8)**:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU setup**:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 4. Download Data

The project uses Yahoo Finance data (via `yfinance`). Data is downloaded automatically during preprocessing.

## Usage

### DO NOT Retrain the Model

The model is **FROZEN** and in production state. Retraining will destroy the validated alpha.

**DO NOT**:
- Add more features
- Change model architecture
- Retrain with different hyperparameters
- Tune temperature again

### Running Backtests

**Test portfolio configurations**:
```bash
python scripts/backtest/phase2b_monetization.py
```

This tests 5 configurations:
1. Baseline (no engineering)
2. Rank filter only
3. Position smoothing only ⭐ (best: Net Sharpe 2.20)
4. Cross-sectional z-score only
5. All techniques combined

**Validate baseline**:
```bash
python scripts/validation/phase2b_validate_baseline.py
```

Forces 100% daily rebalance to prove alpha is real (should show Net Sharpe ~1.32).

### Analyzing Results

Results are saved to:
- `data/processed/phase2b_monetization_results.json` - All configurations
- Logs in `logs/` directory

View with:
```python
import json
with open('data/processed/phase2b_monetization_results.json') as f:
    results = json.load(f)
    for config in results:
        print(f"{config['name']}: Net Sharpe {config['sharpe_net']:.2f}")
```

## Configuration

**Config file**: [config/config.yaml](config/config.yaml)

**FROZEN Settings** (DO NOT CHANGE):
```yaml
model:
  sequence_length: 90
  hidden_size: 128
  num_layers: 2

  loss:
    regression: "rank"
    rank_temperature: 0.05
    rank_weight: 0.7
    huber_delta: 0.05
```

**Portfolio Settings** (Can adjust for Phase 3):
```yaml
backtest:
  long_pct: 0.2           # Top 20%
  short_pct: 0.2          # Bottom 20%
  transaction_cost_bps: 5.0
```

## Key Lessons Learned

1. **Simplicity wins**: 14 features > 40 features
2. **Loss function matters**: Rank loss > MSE for trading
3. **Costs kill**: 120% turnover destroyed 0.71 gross Sharpe
4. **Temperature is critical**: 0.05 vs 1.0 determines success/failure
5. **Portfolio engineering ≠ ML**: Smoothing increased Sharpe 67%
6. **Validate everything**: Baseline stress test caught accounting assumptions
7. **Professional rigor**: Systematic experimentation prevents premature celebration

## Documentation

**Must Read**:
- [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) - Complete journey and results ⭐
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed project structure

**Phase Documentation**:
- [docs/PHASE1_README.md](docs/PHASE1_README.md) - Rank loss implementation
- [docs/PHASE1_SUMMARY.md](docs/PHASE1_SUMMARY.md) - Phase 1 results
- [docs/PHASE2A_README.md](docs/PHASE2A_README.md) - Temperature tuning
- [docs/PHASE2A_VALIDATED.md](docs/PHASE2A_VALIDATED.md) - Temperature results
- [docs/PHASE2B_README.md](docs/PHASE2B_README.md) - Portfolio engineering
- [docs/PHASE2B_BUGFIXES.md](docs/PHASE2B_BUGFIXES.md) - Critical bug fixes

## Next Phase: Phase 3 - Risk Management & Deployment

**Current Status**: Phase 2B complete, strategy validated

**Planned Work**:
1. ✅ Volatility targeting (risk normalization)
2. ✅ Sector neutrality (reduce factor exposure)
3. ✅ Sub-period analysis (rolling Sharpe, IC stability)
4. ✅ Capacity estimation (how much capital?)
5. ✅ Slippage modeling (realistic execution)
6. ✅ Paper trading setup

**What NOT to do**:
- ❌ Add more features
- ❌ Try different architectures
- ❌ Retrain the model
- ❌ Tune more hyperparameters

The ML work is DONE. Focus is now on risk management and deployment.

## Troubleshooting

### RTX 5090 CUDA Error

**Error**: `CUDA error: no kernel image is available for execution on the device`

**Solution**: RTX 5090 requires PyTorch 2.7+ with CUDA 12.8:
```bash
pip uninstall -y torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Out of GPU Memory

Reduce batch size in config:
```yaml
model:
  training:
    batch_size: 16  # Reduce from 32
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### Data Download Fails

The project uses Yahoo Finance via `yfinance`. If downloads fail:
- Check internet connection
- Verify ticker symbols are valid
- Try reducing number of tickers in config

## Performance Characteristics

### Signal Type

**Medium-horizon cross-sectional ranking alpha**:
- NOT daily momentum
- NOT trend-following
- IS relative regime estimation
- Persists over 2-5 days (hence why smoothing helps)

### Institutional Quality

**Strengths**:
- Net Sharpe > 2.0 (institutional quality)
- Low turnover (22% executable at scale)
- Dollar-neutral (market beta ~0)
- Reasonable drawdown (<-6%)

**Validated**:
- Survives 100% daily turnover (Net Sharpe 1.32)
- No data leakage or lookahead bias
- Consistent Gross Sharpe ~2.5 across all backtests

## ChatGPT's Critical Contributions

This project benefited from ChatGPT's professional quant expertise:

1. **Diagnosed over-complexity**: Stopped adding 7 new features
2. **Focused on leverage**: Temperature tuning (highest signal-to-effort ratio)
3. **Added professional metrics**: IC, IC IR, Rank Autocorrelation
4. **Identified 4 critical bugs** in portfolio engineering logic
5. **Validated baseline**: Forced stress test proving alpha is real
6. **Reframed mindset**: ML → Portfolio Engineering transition

**Key quote**:
> "You are no longer searching for alpha. You are engineering execution. That is a huge transition."

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lstm_stock_trading_2025,
  title={LSTM Cross-Sectional Stock Trading Strategy},
  author={Your Name},
  year={2025},
  description={Institutional-quality LSTM trading strategy with rank loss and portfolio engineering},
  url={https://github.com/yourusername/stock-data}
}
```

## License

MIT License - See LICENSE file for details

## Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss. Always do your own research before making investment decisions.

---

**Achievement Unlocked**: Production-grade quant trading strategy ✅

For questions or collaboration, open an issue on GitHub.
