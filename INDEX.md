# 📚 Project Documentation Index

**Quick navigation to all project documentation**

---

## 🚀 START HERE

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[README.md](README.md)** | Project overview & quick start | 5 min |
| **[STRATEGY_DEFINITION.md](STRATEGY_DEFINITION.md)** | Frozen strategy specification (v2.0.0) | 10 min |
| **[START_TOMORROW.md](START_TOMORROW.md)** | Your Day 1 launch guide | 5 min |
| **[PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md)** | Complete navigation guide | 15 min |

---

## 📖 User Guides (docs/guides/)

### Getting Started
- **[PROJECT_OVERVIEW.md](docs/guides/PROJECT_OVERVIEW.md)** - High-level project overview
- **[GETTING_STARTED.md](docs/guides/GETTING_STARTED.md)** - New user onboarding
- **[TROUBLESHOOTING.md](docs/guides/TROUBLESHOOTING.md)** - Common issues & solutions

### Daily Operations
- **[PHASE4_DAILY_WORKFLOW.md](docs/guides/PHASE4_DAILY_WORKFLOW.md)** - Daily/weekly/monthly procedures
- **[AUTOMATION_GUIDE.md](docs/guides/AUTOMATION_GUIDE.md)** - Automated trading setup
- **[PHASE4_GUIDE.md](docs/guides/PHASE4_GUIDE.md)** - Comprehensive paper trading guide
- **[PHASE4_CHECKLIST.md](docs/guides/PHASE4_CHECKLIST.md)** - Launch & progress checklist
- **[PHASE4_LAUNCH_SUMMARY.md](docs/guides/PHASE4_LAUNCH_SUMMARY.md)** - Implementation summary
- **[PHASE4_SIMPLIFIED_START.md](docs/guides/PHASE4_SIMPLIFIED_START.md)** - Why historical replay

---

## 📊 Phase Results (docs/phase_results/)

| Phase | Document | Status | Key Result |
|-------|----------|--------|------------|
| **Phase 0** | [PHASE0_BASELINE.md](docs/phase_results/PHASE0_BASELINE.md) | ✅ Complete | Baseline: Sharpe 0.42 |
| **Phase 1** | [PHASE1_TRAINING.md](docs/phase_results/PHASE1_TRAINING.md) | ✅ Complete | LSTM: Sharpe 2.53 |
| **Phase 2A** | [PHASE2A_TEMPERATURE.md](docs/phase_results/PHASE2A_TEMPERATURE.md) | ✅ Complete | Temp 0.05 optimal |
| **Phase 2B** | [PHASE2B_PORTFOLIO.md](docs/phase_results/PHASE2B_PORTFOLIO.md) | ✅ Complete | Net Sharpe 2.20 |
| **Phase 3** | [PHASE3_RISK_ANALYSIS.md](docs/phase_results/PHASE3_RISK_ANALYSIS.md) | ✅ Complete | GREEN LIGHT |

---

## 🔧 Technical Documentation (docs/technical/)

- **[ARCHITECTURE.md](docs/technical/ARCHITECTURE.md)** - System architecture & design
- **[FEATURES.md](docs/technical/FEATURES.md)** - Feature engineering (14 core features)
- **[MODEL_DETAILS.md](docs/technical/MODEL_DETAILS.md)** - LSTM model specifications
- **[PORTFOLIO_LOGIC.md](docs/technical/PORTFOLIO_LOGIC.md)** - Portfolio construction rules
- **[RISK_FRAMEWORK.md](docs/technical/RISK_FRAMEWORK.md)** - Risk management system

---

## 📝 Scripts & Automation

| Script | Purpose |
|--------|---------|
| `scripts/automation/daily_paper_trading_qdrant.py` | Main automation script |
| `scripts/automation/query_qdrant.py` | Database queries |
| `scripts/automation/setup_daily_task.ps1` | Windows Task Scheduler setup |
| `scripts/automation/setup_daily_task.bat` | Alternative batch setup |
| `scripts/paper_trading/run_daily_paper_trading.py` | Simple daily runner |
| `scripts/paper_trading/phase4_performance_tracker.py` | Performance reporting |

---

## 📊 Historical Documents

### Legacy Phase Docs (Reference Only)
- [FINAL_RESULTS.md](docs/FINAL_RESULTS.md) - Phase 0-2 complete journey
- [PHASE1_README.md](docs/phase_results/PHASE1_README.md) - Phase 1 original docs
- [PHASE1_SUMMARY.md](docs/phase_results/PHASE1_SUMMARY.md) - Phase 1 summary
- [PHASE2A_README.md](docs/phase_results/PHASE2A_README.md) - Phase 2A docs
- [PHASE2A_VALIDATED.md](docs/phase_results/PHASE2A_VALIDATED.md) - Phase 2A results
- [PHASE2B_README.md](docs/phase_results/PHASE2B_README.md) - Phase 2B docs
- [PHASE2B_BUGFIXES.md](docs/phase_results/PHASE2B_BUGFIXES.md) - Critical fixes
- [PHASE3_COMPLETE.md](docs/phase_results/PHASE3_COMPLETE.md) - Phase 3 overview
- [PHASE3_2_FINDINGS.md](docs/phase_results/PHASE3_2_FINDINGS.md) - Risk decomposition
- [PHASE3_3_FIXES.md](docs/phase_results/PHASE3_3_FIXES.md) - Cost calculation fix
- [PHASE3_FINAL_DECISION.md](docs/phase_results/PHASE3_FINAL_DECISION.md) - Deployment decision
- [PHASE3_EXECUTION_GUIDE.md](PHASE3_EXECUTION_GUIDE.md) - Phase 3 guide (root directory)

### Misc Legacy
- [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - Old cleanup notes
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Old structure doc

---

## 🎯 By Task

### "I want to start paper trading"
1. [START_TOMORROW.md](START_TOMORROW.md)
2. [docs/guides/PHASE4_DAILY_WORKFLOW.md](docs/guides/PHASE4_DAILY_WORKFLOW.md)
3. [docs/guides/PHASE4_CHECKLIST.md](docs/guides/PHASE4_CHECKLIST.md)

### "I want to setup automation"
1. [docs/guides/AUTOMATION_GUIDE.md](docs/guides/AUTOMATION_GUIDE.md)
2. Run: `scripts/automation/setup_daily_task.ps1`

### "I want to understand the strategy"
1. [STRATEGY_DEFINITION.md](STRATEGY_DEFINITION.md)
2. [docs/technical/ARCHITECTURE.md](docs/technical/ARCHITECTURE.md)
3. [docs/technical/FEATURES.md](docs/technical/FEATURES.md)

### "I want to review performance"
1. [docs/guides/PHASE4_GUIDE.md](docs/guides/PHASE4_GUIDE.md) - Section: Performance Monitoring
2. Run: `python scripts/paper_trading/phase4_performance_tracker.py`
3. View: `reports/phase4/phase4_performance_report.txt`

### "I want to understand the history"
1. [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md) - Phase 0-2 journey
2. [docs/phase_results/PHASE3_RISK_ANALYSIS.md](docs/phase_results/PHASE3_RISK_ANALYSIS.md) - Phase 3
3. [docs/guides/PHASE4_LAUNCH_SUMMARY.md](docs/guides/PHASE4_LAUNCH_SUMMARY.md) - Phase 4 start

---

## 📂 File Locations

### Your Active Files
```
logs/paper_trading/
├── JOURNAL.md                  ⭐ Your daily notes
└── WEEKLY_SUMMARY.md           ⭐ Your weekly summaries
```

### Results Files
```
data/processed/phase4/
├── phase4_paper_trading_daily.parquet      Trading history
├── phase4_paper_trading_summary.json       Summary metrics
└── paper_trading_progress.json             Progress tracker
```

### Reports
```
reports/phase4/
├── phase4_performance_report.txt           Latest report
└── phase4_performance_plots.png            Visualizations
```

---

## 🔍 Search Tips

### Find by Topic
- **Automation**: Search "automation" in `AUTOMATION_GUIDE.md`
- **Strategy Spec**: `STRATEGY_DEFINITION.md`
- **Daily Workflow**: `PHASE4_DAILY_WORKFLOW.md`
- **Troubleshooting**: `docs/guides/TROUBLESHOOTING.md`

### Find by Phase
- **Phase 0-2**: `docs/FINAL_RESULTS.md`
- **Phase 3**: `docs/phase_results/PHASE3_RISK_ANALYSIS.md`
- **Phase 4**: `docs/guides/PHASE4_GUIDE.md`

### Find by File Type
- **Guides**: `docs/guides/*.md`
- **Phase Results**: `docs/phase_results/*.md`
- **Technical**: `docs/technical/*.md`

---

## 📊 Document Status

| Status | Meaning | Documents |
|--------|---------|-----------|
| ⭐ **Active** | Read regularly | START_TOMORROW, DAILY_WORKFLOW, JOURNAL |
| ✅ **Complete** | Reference only | Phase 0-3 results |
| 🔄 **In Progress** | Being updated | PHASE4_GUIDE, performance reports |
| 📦 **Archive** | Historical | Old phase docs |

---

## 🎓 Reading Order

### First Time Users
1. README.md
2. STRATEGY_DEFINITION.md
3. docs/guides/PROJECT_OVERVIEW.md
4. docs/guides/GETTING_STARTED.md
5. START_TOMORROW.md

### Before Starting Paper Trading
1. STRATEGY_DEFINITION.md
2. docs/guides/PHASE4_GUIDE.md
3. docs/guides/PHASE4_DAILY_WORKFLOW.md
4. docs/guides/PHASE4_CHECKLIST.md
5. START_TOMORROW.md

### Want Deep Understanding
1. docs/FINAL_RESULTS.md (Phase 0-2 journey)
2. docs/phase_results/PHASE3_RISK_ANALYSIS.md
3. docs/technical/ARCHITECTURE.md
4. docs/technical/FEATURES.md
5. docs/technical/MODEL_DETAILS.md

---

## 📞 Quick Links

| Need | Document |
|------|----------|
| **Emergency Help** | docs/guides/TROUBLESHOOTING.md |
| **Daily Commands** | PROJECT_ORGANIZATION.md (Quick Reference) |
| **Strategy Spec** | STRATEGY_DEFINITION.md |
| **Performance Check** | reports/phase4/phase4_performance_report.txt |
| **Database Queries** | AUTOMATION_GUIDE.md (Query section) |

---

## ✅ Documentation Health Check

All these files should exist:

**Critical**:
- ✅ README.md
- ✅ STRATEGY_DEFINITION.md
- ✅ START_TOMORROW.md
- ✅ PROJECT_ORGANIZATION.md

**Guides**:
- ✅ docs/guides/PROJECT_OVERVIEW.md
- ✅ docs/guides/GETTING_STARTED.md
- ✅ docs/guides/AUTOMATION_GUIDE.md
- ✅ docs/guides/PHASE4_GUIDE.md
- ✅ docs/guides/PHASE4_DAILY_WORKFLOW.md
- ✅ docs/guides/TROUBLESHOOTING.md

**Technical Docs**:
- ✅ docs/technical/ARCHITECTURE.md
- ✅ docs/technical/FEATURES.md
- ✅ docs/technical/MODEL_DETAILS.md
- ✅ docs/technical/PORTFOLIO_LOGIC.md
- ✅ docs/technical/RISK_FRAMEWORK.md

**Phase Results**:
- ✅ docs/phase_results/PHASE0_BASELINE.md
- ✅ docs/phase_results/PHASE1_TRAINING.md
- ✅ docs/phase_results/PHASE2A_TEMPERATURE.md
- ✅ docs/phase_results/PHASE2B_PORTFOLIO.md
- ✅ docs/phase_results/PHASE3_RISK_ANALYSIS.md

**Your Files**:
- ✅ logs/paper_trading/JOURNAL.md
- ✅ logs/paper_trading/WEEKLY_SUMMARY.md

**Results**:
- ✅ data/processed/phase4/phase4_paper_trading_summary.json
- ✅ reports/phase4/phase4_performance_report.txt

---

## 🔄 Last Updated

**Date**: January 20, 2026
**Phase**: 4 - Paper Trading
**Status**: Automation Enabled ✅

---

**Can't find what you need?**
1. Search this file (Ctrl+F)
2. Check [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md)
3. Read [docs/guides/TROUBLESHOOTING.md](docs/guides/TROUBLESHOOTING.md)
