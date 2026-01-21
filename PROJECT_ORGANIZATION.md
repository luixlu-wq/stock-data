# Project Organization Guide

**Last Updated**: January 20, 2026
**Current Phase**: Phase 4 - Paper Trading
**Status**: Production Ready ✅

---

## 📁 Project Structure

```
stock-data/
│
├── 📖 QUICK START FILES (Read These First)
│   ├── README.md                           ⭐ Project overview & quick start
│   ├── STRATEGY_DEFINITION.md              ⭐ Frozen strategy specification (v2.0.0)
│   ├── START_TOMORROW.md                   🚀 Your Day 1 launch guide
│   └── PROJECT_ORGANIZATION.md             📋 This file - navigation guide
│
├── 📚 DOCUMENTATION (docs/)
│   │
│   ├── guides/                             User Guides
│   │   ├── PROJECT_OVERVIEW.md            ⭐ High-level project overview
│   │   ├── GETTING_STARTED.md             ⭐ New user onboarding
│   │   ├── AUTOMATION_GUIDE.md            🤖 Automated trading setup
│   │   ├── PHASE4_GUIDE.md                📊 Paper trading comprehensive guide
│   │   ├── DAILY_WORKFLOW.md              📅 Daily/weekly/monthly procedures
│   │   └── TROUBLESHOOTING.md             🔧 Common issues & solutions
│   │
│   ├── phase_results/                      Phase Completion Documents
│   │   ├── PHASE0_BASELINE.md             Phase 0: Initial experiments
│   │   ├── PHASE1_TRAINING.md             Phase 1: LSTM with rank loss
│   │   ├── PHASE2A_TEMPERATURE.md         Phase 2A: Temperature tuning
│   │   ├── PHASE2B_PORTFOLIO.md           Phase 2B: Portfolio engineering
│   │   └── PHASE3_RISK_ANALYSIS.md        Phase 3: Risk validation (complete)
│   │
│   └── technical/                          Technical Documentation
│       ├── ARCHITECTURE.md                System architecture & design
│       ├── FEATURES.md                    Feature engineering details
│       ├── MODEL_DETAILS.md               LSTM model specifications
│       ├── PORTFOLIO_LOGIC.md             Portfolio construction rules
│       └── RISK_FRAMEWORK.md              Risk management system
│
├── 🔧 SCRIPTS (scripts/)
│   │
│   ├── automation/                         Automated Trading
│   │   ├── daily_paper_trading_qdrant.py  Main automation script
│   │   ├── query_qdrant.py                Database query tool
│   │   ├── setup_daily_task.ps1           Windows scheduler setup (PowerShell)
│   │   └── setup_daily_task.bat           Windows scheduler setup (Batch)
│   │
│   ├── paper_trading/                      Paper Trading
│   │   ├── run_daily_paper_trading.py     ⭐ Daily manual execution
│   │   ├── phase4_paper_trading_runner.py Core paper trading engine
│   │   ├── phase4_performance_tracker.py  Performance monitoring
│   │   └── phase4_daily_pipeline.py       Live data pipeline (future)
│   │
│   ├── portfolio/                          Portfolio Construction
│   │   ├── phase3_3_portfolio_comparison.py
│   │   ├── phase3_4_short_salvage.py
│   │   └── phase3_5_risk_management.py
│   │
│   ├── risk/                               Risk Analysis
│   │   └── phase3_2b_factor_exposure.py
│   │
│   ├── training/                           Model Training (Complete - DO NOT RERUN)
│   │   ├── phase1_train.py
│   │   └── phase2a_temperature_experiment.py
│   │
│   ├── backtest/                           Backtesting
│   │   └── phase2b_monetization.py
│   │
│   └── validation/                         Validation
│       └── phase2b_validate_baseline.py
│
├── 💾 DATA (data/)
│   │
│   ├── processed/                          Processed Datasets
│   │   ├── phase1_predictions.parquet      ⭐ Historical predictions (188 days)
│   │   ├── phase4/                        Paper trading results
│   │   │   ├── phase4_paper_trading_daily.parquet
│   │   │   ├── phase4_paper_trading_summary.json
│   │   │   └── paper_trading_progress.json
│   │   ├── phase3_*.json                  Phase 3 results
│   │   └── s2_daily_pnl_history.parquet   S2 strategy daily PnL
│   │
│   └── raw/                                Raw Data (empty - auto-downloaded)
│
├── 📊 REPORTS (reports/)
│   │
│   └── phase4/                             Phase 4 Reports
│       ├── phase4_performance_report.txt   Latest performance summary
│       └── phase4_performance_plots.png    6-panel visualization
│
├── 📝 LOGS (logs/)
│   │
│   ├── automation/                         Automation Logs
│   │   └── daily_automation_*.log          Daily automation execution logs
│   │
│   └── paper_trading/                      Paper Trading Logs
│       ├── JOURNAL.md                      ⭐ Your daily trading journal
│       ├── WEEKLY_SUMMARY.md               ⭐ Your weekly summaries
│       └── paper_trading_*.log             Execution logs
│
├── 🧠 MODELS (models/)
│   │
│   └── checkpoints/                        Model Checkpoints
│       └── lstm_phase2a_temp0.05_best.pth  ⭐ FROZEN production model
│
├── 📦 SOURCE CODE (src/)
│   │
│   ├── data/                               Data Processing
│   │   └── preprocessor_v2.py              Feature engineering (14 features)
│   │
│   ├── models/                             Model Definitions
│   │   └── lstm_model.py                   2-layer LSTM architecture
│   │
│   └── utils/                              Utilities
│       └── config_loader.py                Configuration management
│
└── ⚙️ CONFIGURATION (config/)
    └── config.yaml                         ⭐ Main configuration (FROZEN)
```

---

## 🎯 Navigation by Task

### I Want To...

#### Start Paper Trading
1. Read: [`START_TOMORROW.md`](START_TOMORROW.md)
2. Read: [`docs/guides/DAILY_WORKFLOW.md`](docs/guides/DAILY_WORKFLOW.md)
3. Run: `python scripts/paper_trading/run_daily_paper_trading.py`
4. Update: `logs/paper_trading/JOURNAL.md`

#### Setup Automation
1. Read: [`docs/guides/AUTOMATION_GUIDE.md`](docs/guides/AUTOMATION_GUIDE.md)
2. Start Qdrant: `docker run -d -p 6333:6333 qdrant/qdrant`
3. Run: `.\scripts\automation\setup_daily_task.ps1` (as Administrator)
4. Test: `python scripts/automation/daily_paper_trading_qdrant.py`

#### Understand the Strategy
1. Read: [`STRATEGY_DEFINITION.md`](STRATEGY_DEFINITION.md)
2. Read: [`docs/technical/ARCHITECTURE.md`](docs/technical/ARCHITECTURE.md)
3. Read: [`docs/technical/FEATURES.md`](docs/technical/FEATURES.md)

#### Review Performance
1. Weekly: `python scripts/paper_trading/phase4_performance_tracker.py`
2. View report: `reports/phase4/phase4_performance_report.txt`
3. View plots: `reports/phase4/phase4_performance_plots.png`
4. Query Qdrant: `python scripts/automation/query_qdrant.py --type performance`

#### Query Stock Recommendations
1. View latest: `python scripts/automation/query_qdrant.py --type recommendations`
2. Specific date: `python scripts/automation/query_qdrant.py --date 2025-04-01`
3. Similar stocks: `python scripts/automation/query_qdrant.py --search AAPL`

#### Understand What Happened (History)
1. Phase 0-2: Read `docs/FINAL_RESULTS.md`
2. Phase 3: Read `docs/phase_results/PHASE3_RISK_ANALYSIS.md`
3. Phase 4: Read `PHASE4_GUIDE.md`

#### Troubleshoot Issues
1. Read: `docs/guides/TROUBLESHOOTING.md`
2. Check logs: `logs/automation/` or `logs/paper_trading/`
3. Verify Qdrant: http://localhost:6333/dashboard
4. Test manually: Run scripts with `--help` flag

---

## 📋 Document Categories

### 🟢 User-Facing (Start Here)
| Document | Purpose | When to Read |
|----------|---------|--------------|
| `README.md` | Project overview | First time |
| `STRATEGY_DEFINITION.md` | Strategy spec | Before starting |
| `START_TOMORROW.md` | Day 1 guide | Day before launch |
| `docs/guides/AUTOMATION_GUIDE.md` | Setup automation | One-time setup |
| `docs/guides/DAILY_WORKFLOW.md` | Daily procedures | Every day |

### 🟡 Reference (As Needed)
| Document | Purpose | When to Read |
|----------|---------|--------------|
| `PHASE4_GUIDE.md` | Comprehensive P4 guide | When confused |
| `docs/phase_results/PHASE3_*.md` | Phase 3 results | Understanding validation |
| `docs/technical/*.md` | Technical details | Deep dive |

### 🔴 Historical (Archive)
| Document | Purpose | When to Read |
|----------|---------|--------------|
| `docs/FINAL_RESULTS.md` | Phase 0-2 journey | Understanding history |
| `PHASE1_*.md`, `PHASE2_*.md` | Old phase docs | Research only |
| `CLEANUP_SUMMARY.md` | Old cleanup notes | Rarely |

---

## 🗂️ File Organization Rules

### ✅ Keep These Files (Critical)
- `STRATEGY_DEFINITION.md` - Strategy spec (v2.0.0)
- `START_TOMORROW.md` - Launch guide
- `PHASE4_*.md` - Phase 4 guides
- `scripts/automation/*` - Automation scripts
- `scripts/paper_trading/*` - Paper trading scripts
- `logs/paper_trading/JOURNAL.md` - Your journal
- `models/checkpoints/lstm_phase2a_temp0.05_best.pth` - Production model
- `data/processed/phase1_predictions.parquet` - Historical predictions

### 📦 Archive These Files (Move to `archive/`)
- `CLEANUP_SUMMARY.md`
- Old phase docs (if consolidated)
- Experimental scripts no longer used

### 🔥 Can Delete (If Needed)
- `*.pyc` files
- `__pycache__` directories
- Old log files (after 90 days)
- Test outputs

---

## 📊 Data Flow

```
Historical Data
    ↓
phase1_predictions.parquet (188 days)
    ↓
run_daily_paper_trading.py
    ↓
phase4_paper_trading_daily.parquet (cumulative)
    ↓
phase4_performance_tracker.py
    ↓
reports/phase4/*.txt, *.png
    ↓
Qdrant Database (if automation enabled)
```

---

## 🚀 Quick Reference Commands

### Daily Operations
```bash
# Manual paper trading
python scripts/paper_trading/run_daily_paper_trading.py

# Automated (runs at 4:15 PM via Task Scheduler)
# Nothing to do - it runs automatically!

# Query today's results
python scripts/automation/query_qdrant.py --type recommendations --limit 10
```

### Weekly Review
```bash
# Generate performance report
python scripts/paper_trading/phase4_performance_tracker.py

# View report
cat reports/phase4/phase4_performance_report.txt

# Query Qdrant
python scripts/automation/query_qdrant.py --type performance
```

### Setup (One-Time)
```bash
# Start Qdrant
docker run -d -p 6333:6333 --name qdrant-paper-trading qdrant/qdrant

# Setup automation
.\scripts\automation\setup_daily_task.ps1  # Run as Administrator

# Test automation
python scripts/automation/daily_paper_trading_qdrant.py
```

---

## 📍 Where to Find Things

### "Where are my stock recommendations?"
**Answer**: Qdrant database (if automation enabled)
- Query: `python scripts/automation/query_qdrant.py --type recommendations`
- Or: Check Qdrant dashboard at http://localhost:6333/dashboard

### "Where is my trading history?"
**Answer**: `data/processed/phase4/phase4_paper_trading_daily.parquet`
- View: Load with pandas
- Or: Use performance tracker script

### "Where are my performance metrics?"
**Answer**: `data/processed/phase4/phase4_paper_trading_summary.json`
- View: `cat data/processed/phase4/phase4_paper_trading_summary.json`
- Or: `python scripts/automation/query_qdrant.py --type performance`

### "Where do I write my daily notes?"
**Answer**: `logs/paper_trading/JOURNAL.md`
- Update after each day's trading
- Template provided in the file

### "Where are the automation logs?"
**Answer**: `logs/automation/daily_automation_*.log`
- One file per day
- Check if automation failed

### "Where is the production model?"
**Answer**: `models/checkpoints/lstm_phase2a_temp0.05_best.pth`
- FROZEN - do not retrain
- Used by all paper trading scripts

---

## 🎯 Common Workflows

### Workflow 1: Daily Paper Trading (Manual)
```
1. Open terminal (4:15 PM EST)
2. cd c:\Users\luixj\AI\stock-data
3. venv\Scripts\activate
4. python scripts/paper_trading/run_daily_paper_trading.py
5. Review output
6. Update logs/paper_trading/JOURNAL.md
7. Done! (10-15 minutes)
```

### Workflow 2: Weekly Performance Review
```
1. Friday end of day
2. python scripts/paper_trading/phase4_performance_tracker.py
3. Open reports/phase4/phase4_performance_report.txt
4. Open reports/phase4/phase4_performance_plots.png
5. Check: Sharpe > 1.0? MaxDD < -10%? KS < 15%?
6. Update logs/paper_trading/WEEKLY_SUMMARY.md
7. Done! (30 minutes)
```

### Workflow 3: Query Qdrant for Insights
```
1. View latest recommendations
   python scripts/automation/query_qdrant.py --type recommendations

2. View trading results
   python scripts/automation/query_qdrant.py --type results

3. Search similar to AAPL
   python scripts/automation/query_qdrant.py --search AAPL

4. View specific date
   python scripts/automation/query_qdrant.py --date 2025-04-01
```

---

## 🔍 Troubleshooting Quick Links

| Issue | Check | Solution Doc |
|-------|-------|--------------|
| Automation not running | Task Scheduler | `docs/guides/AUTOMATION_GUIDE.md` |
| Qdrant connection failed | Docker | `docs/guides/TROUBLESHOOTING.md` |
| Script errors | Logs directory | `docs/guides/TROUBLESHOOTING.md` |
| Missing dependencies | pip install | `README.md` Installation section |
| Data not found | data/processed/ | Re-run prediction generation |

---

## 📈 Current Status

**Phase**: 4 - Paper Trading (Automated)
**Days Completed**: 3 / 60+
**Sharpe**: 1.22 ✅
**MaxDD**: -4.72% ✅
**Status**: On Track ✅

**Next Milestone**: Day 20 checkpoint (approximately Feb 15, 2026)

---

## 🎓 Learning Path

### Beginner (New to Project)
1. `README.md` - Project overview
2. `docs/guides/PROJECT_OVERVIEW.md` - Understand goals
3. `STRATEGY_DEFINITION.md` - Strategy basics
4. `START_TOMORROW.md` - Get started

### Intermediate (Running Paper Trading)
1. `docs/guides/DAILY_WORKFLOW.md` - Daily procedures
2. `PHASE4_GUIDE.md` - Comprehensive guide
3. `docs/guides/AUTOMATION_GUIDE.md` - Automation setup
4. `logs/paper_trading/JOURNAL.md` - Track your progress

### Advanced (Understanding Internals)
1. `docs/technical/ARCHITECTURE.md` - System design
2. `docs/technical/FEATURES.md` - Feature engineering
3. `docs/technical/MODEL_DETAILS.md` - LSTM details
4. `docs/phase_results/PHASE3_*.md` - Validation process

---

## 🗄️ Archive Policy

### Keep Forever
- Strategy definition
- Production model
- Phase 3+ results
- Your trading journal

### Archive After 90 Days
- Old logs (automation, paper trading)
- Test outputs
- Temporary analysis files

### Can Delete
- Python cache (`__pycache__`)
- Temporary files (`*.tmp`)
- Old error logs (after resolving)

---

## 📞 Getting Help

1. **Check Documentation**:
   - Start with `docs/guides/TROUBLESHOOTING.md`
   - Search this file (Ctrl+F)

2. **Check Logs**:
   - `logs/automation/` for automation issues
   - `logs/paper_trading/` for paper trading issues

3. **Common Issues**:
   - See `docs/guides/TROUBLESHOOTING.md`

4. **Contact**:
   - Open GitHub issue
   - Check Qdrant dashboard: http://localhost:6333/dashboard

---

## ✅ Quick Health Check

Run this checklist to verify everything is working:

```bash
# 1. Virtual environment active?
which python  # Should show venv path

# 2. Qdrant running?
curl http://localhost:6333/  # Should return JSON

# 3. Model exists?
ls models/checkpoints/lstm_phase2a_temp0.05_best.pth

# 4. Data exists?
ls data/processed/phase1_predictions.parquet

# 5. Automation task exists?
Get-ScheduledTask -TaskName "DailyPaperTrading"  # PowerShell

# 6. Can run paper trading?
python scripts/paper_trading/run_daily_paper_trading.py --help
```

All checks pass? ✅ You're ready!

---

**Last Updated**: January 20, 2026
**Maintainer**: Project Owner
**Version**: 2.0

For updates to this guide, check git history or project README.
