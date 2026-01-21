# Getting Started

**New User Onboarding for Stock Trading System**

## Prerequisites

### Required Knowledge
- Python programming (intermediate level)
- Basic machine learning concepts
- Understanding of stock markets and trading
- Git/GitHub basics

### System Requirements
- Python 3.8+ (3.12 recommended)
- 8GB+ RAM
- Windows/Linux/Mac OS
- Internet connection for data downloads

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd stock-data
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (if needed)
# Most settings have sensible defaults
```

### 5. Verify Installation
```bash
# Check if Python can import key modules
python -c "import torch; import pandas; import qdrant_client; print('✓ All dependencies installed')"
```

## Understanding the Project

### What's Already Done

This project is **NOT** a tutorial or starter template. It's a **complete, working trading system** that has been:

1. ✅ Trained on historical data
2. ✅ Backtested across multiple strategies
3. ✅ Optimized for risk and returns
4. ✅ Validated with rigorous analysis
5. ✅ Ready for paper trading

### Key Files to Read First

1. **[README.md](../../README.md)** - Project homepage (5 min)
2. **[STRATEGY_DEFINITION.md](../../STRATEGY_DEFINITION.md)** - Frozen strategy spec (10 min)
3. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - High-level overview (10 min)
4. **[docs/FINAL_RESULTS.md](../FINAL_RESULTS.md)** - Phase 0-2 journey (15 min)

### Important Directories

```
data/
├── raw/                      # Downloaded stock data
├── processed/
│   ├── phase1_predictions.parquet    # ML predictions (pre-computed)
│   └── phase4/                       # Paper trading results

models/
└── best_lstm_model.pth      # Trained LSTM model

scripts/
├── paper_trading/           # Phase 4 scripts
└── automation/              # Automated daily trading

docs/
├── guides/                  # User guides (you are here)
├── phase_results/           # Phase 0-3 results
└── technical/               # Technical documentation
```

## Quick Start Options

### Option A: Run Paper Trading (Recommended for New Users)

This is the safest way to understand the system without risking anything.

```bash
# Run a single day of paper trading
python scripts/paper_trading/run_daily_paper_trading.py

# View results
cat data/processed/phase4/phase4_paper_trading_summary.json

# View performance report
python scripts/paper_trading/phase4_performance_tracker.py
cat reports/phase4/phase4_performance_report.txt
```

**What this does**: Simulates trading for one historical day (2025-04-01 onwards), saves results, and shows performance metrics.

### Option B: Explore Historical Results

```bash
# View Phase 2B backtest results
python scripts/phase2b/phase2b_runner.py --strategy S2

# This will show you:
# - Portfolio construction
# - Daily P&L
# - Risk metrics
# - Transaction costs
```

### Option C: Setup Automation (For Ongoing Paper Trading)

If you want to run paper trading daily automatically:

1. Read [AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md)
2. Install Qdrant:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
3. Setup daily task:
   ```powershell
   # Windows (as Administrator)
   .\scripts\automation\setup_daily_task.ps1
   ```

## Understanding the Data

### Pre-computed Predictions

The key file is `data/processed/phase1_predictions.parquet`:

```python
import pandas as pd

# Load predictions
df = pd.read_parquet('data/processed/phase1_predictions.parquet')

# Columns:
# - date: Trading date
# - ticker: Stock ticker
# - y_pred_reg: Predicted next-day return
# - y_true_reg: Actual next-day return
# - close: Closing price
# - ... (14 engineered features)

# View sample
print(df.head())
```

This file contains predictions for **2025-04-01 to 2025-10-31** (159 trading days).

### Why Pre-computed?

The LSTM model has already been trained and predictions generated. This allows you to:
- Focus on strategy validation (not model training)
- Run paper trading faster (no inference overhead)
- Ensure reproducibility

## Common First Tasks

### Task 1: Understand the Strategy

1. Read [STRATEGY_DEFINITION.md](../../STRATEGY_DEFINITION.md)
2. Look at [docs/technical/PORTFOLIO_LOGIC.md](../technical/PORTFOLIO_LOGIC.md)
3. Run a backtest:
   ```bash
   python scripts/phase2b/phase2b_runner.py --strategy S2
   ```

### Task 2: Run Your First Paper Trade

1. Check that predictions exist:
   ```bash
   python -c "import pandas as pd; df=pd.read_parquet('data/processed/phase1_predictions.parquet'); print(f'Loaded {len(df)} rows')"
   ```

2. Run one day:
   ```bash
   python scripts/paper_trading/run_daily_paper_trading.py
   ```

3. View results:
   ```bash
   cat data/processed/phase4/phase4_paper_trading_summary.json
   ```

### Task 3: Explore the Model

```python
import torch
from src.models.lstm_model import StockLSTM

# Load model
model = StockLSTM(input_size=14, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load('models/best_lstm_model.pth'))
model.eval()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Model architecture:\n{model}")
```

### Task 4: Query Paper Trading Results (if automation running)

```bash
# View latest recommendations
python scripts/automation/query_qdrant.py --type recommendations --limit 10

# View trading results
python scripts/automation/query_qdrant.py --type results --limit 5

# View performance
python scripts/automation/query_qdrant.py --type performance
```

## Next Steps

### To Start Daily Paper Trading

Follow [START_TOMORROW.md](../../START_TOMORROW.md) for a step-by-step Day 1 guide.

### To Understand the Full System

1. **Technical Docs**: [docs/technical/](../technical/)
   - [ARCHITECTURE.md](../technical/ARCHITECTURE.md) - System design
   - [FEATURES.md](../technical/FEATURES.md) - Feature engineering
   - [MODEL_DETAILS.md](../technical/MODEL_DETAILS.md) - LSTM specs

2. **Phase Results**: [docs/phase_results/](../phase_results/)
   - See how each phase improved the strategy
   - Understand key decisions and trade-offs

3. **Daily Operations**: [DAILY_WORKFLOW.md](DAILY_WORKFLOW.md)
   - Daily, weekly, monthly procedures
   - Performance monitoring
   - When to intervene

### To Modify the System

**⚠️ WARNING**: The strategy is frozen at v2.0.0. Any modifications will deviate from validated backtest results.

If you still want to experiment:
1. Create a new branch
2. Read [docs/technical/ARCHITECTURE.md](../technical/ARCHITECTURE.md) to understand system structure
3. Make changes in isolation
4. Re-run full backtests to validate
5. Do NOT merge to main without full validation

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the virtual environment
# Windows
venv\Scripts\activate

# Install missing packages
pip install -r requirements.txt
```

### "File not found" errors
```bash
# Make sure you're in the project root
cd /path/to/stock-data

# Check if predictions file exists
ls -l data/processed/phase1_predictions.parquet
```

### Qdrant connection errors
```bash
# Make sure Qdrant is running
docker ps | grep qdrant

# Start Qdrant if not running
docker run -p 6333:6333 qdrant/qdrant
```

For more issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## Additional Resources

- **[INDEX.md](../../INDEX.md)**: Quick reference to all documentation
- **[PROJECT_ORGANIZATION.md](../../PROJECT_ORGANIZATION.md)**: Complete navigation guide
- **[AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md)**: Setup automated trading

## Getting Help

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Search existing documentation in [docs/](../)
3. Review relevant phase results
4. Check code comments in [src/](../../src/)

---

**Welcome to the project!** Take your time understanding the system before running paper trades or making modifications.

**Last Updated**: January 20, 2026
