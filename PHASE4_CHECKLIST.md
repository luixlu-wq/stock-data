# Phase 4 Paper Trading - Start Checklist

Use this checklist to prepare for and launch paper trading.

---

## Pre-Launch Setup (Do Once)

### Technical Setup

- [ ] Python environment activated and tested
  ```bash
  cd c:\Users\luixj\AI\stock-data
  venv\Scripts\activate
  python --version  # Should show Python 3.x
  ```

- [ ] All dependencies installed
  ```bash
  pip install matplotlib yfinance pandas numpy torch scikit-learn pyarrow
  ```

- [ ] Model file exists
  ```bash
  ls models/checkpoints/lstm_phase2a_temp0.05_best.pth
  # Should show the file
  ```

- [ ] Directories created
  ```bash
  mkdir -p logs/paper_trading
  mkdir -p data/paper_trading/daily
  mkdir -p reports/paper_trading/daily
  ```

- [ ] Scripts tested (dry run)
  ```bash
  python scripts/paper_trading/phase4_paper_trading_runner.py --start-date 2025-12-01 --end-date 2025-12-01
  # Should complete without errors
  ```

- [ ] Performance tracker tested
  ```bash
  python scripts/paper_trading/phase4_performance_tracker.py
  # Should generate report and plots
  ```

### Documentation Setup

- [ ] Read [PHASE4_GUIDE.md](PHASE4_GUIDE.md) completely
- [ ] Read [PHASE4_DAILY_WORKFLOW.md](PHASE4_DAILY_WORKFLOW.md) completely
- [ ] Understand success criteria (Sharpe > 1.0, MaxDD < -10%)
- [ ] Understand red flags (when to stop)
- [ ] Journal template ready: `logs/paper_trading/JOURNAL.md`
- [ ] Weekly summary template ready: `logs/paper_trading/WEEKLY_SUMMARY.md`

### Calendar Setup

- [ ] Set start date: _______________
- [ ] Calculate target end date (60 trading days): _______________
- [ ] Set daily reminder: 4:15 PM EST (after market close)
- [ ] Set weekly review: Every Friday end-of-day
- [ ] Set monthly review: Last trading day of each month
- [ ] Block out holidays (market closed days)

### Mental Preparation

- [ ] Understand this will take 3+ months
- [ ] Commit to running EVERY trading day
- [ ] Commit to NO parameter changes
- [ ] Commit to NO model retraining
- [ ] Commit to NO feature engineering
- [ ] Understand that some days will lose money (normal)
- [ ] Understand that Sharpe > 1.0 is the goal, not 100% win rate

---

## Day 1 Launch

### Before Market Open (9:00 AM EST)

- [ ] Confirm market is open today
- [ ] Review strategy parameters one last time:
  - K = 38 (top/bottom stocks)
  - Long exposure: 65%
  - Short exposure: 35%
  - Short filter: y_pred < 0
  - Vol target: 8% annual
  - EWMA alpha: 0.15

- [ ] Set reminder for 4:15 PM execution

### After Market Close (4:15 PM EST)

- [ ] Run paper trading for Day 1
  ```bash
  cd c:\Users\luixj\AI\stock-data
  venv\Scripts\activate

  # Replace with actual date
  python scripts/paper_trading/phase4_paper_trading_runner.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD
  ```

- [ ] Review results in terminal output
- [ ] Check for any errors
- [ ] Note key metrics:
  - PnL Net: _______
  - PnL Scaled: _______
  - Turnover: _______ (expect ~100% on day 1)
  - Kill Switches: _______ (expect 0)

- [ ] Log results in `logs/paper_trading/JOURNAL.md`

- [ ] Check files created:
  ```bash
  ls data/processed/phase4/
  # Should see phase4_paper_trading_daily.parquet and phase4_paper_trading_summary.json
  ```

- [ ] Celebrate! Day 1 complete! 🎉

---

## Daily Routine (Days 2-60+)

### Every Trading Day After Market Close

- [ ] Run paper trading script for today's date
- [ ] Review terminal output
- [ ] Check for errors or warnings
- [ ] Note if any kill switches triggered
- [ ] Update journal with today's entry
- [ ] Quick sanity check: Does performance look reasonable?

**Time commitment**: 10-15 minutes per day

---

## Weekly Routine (Every Friday)

### End of Week Review

- [ ] Run performance tracker
  ```bash
  python scripts/paper_trading/phase4_performance_tracker.py
  ```

- [ ] Review report: `reports/phase4/phase4_performance_report.txt`

- [ ] Review plots: `reports/phase4/phase4_performance_plots.png`

- [ ] Check key metrics:
  - [ ] Sharpe > 1.0? ✅/❌
  - [ ] MaxDD < -10%? ✅/❌
  - [ ] Kill switches < 15%? ✅/❌
  - [ ] Any RED alerts? ✅/❌

- [ ] Update weekly summary: `logs/paper_trading/WEEKLY_SUMMARY.md`

- [ ] Decision: Continue / Investigate / Halt

**Time commitment**: 30 minutes per week

---

## Monthly Routine (End of Each Month)

### Monthly Deep Dive

- [ ] Generate comprehensive performance report
- [ ] Review all 6 performance plots in detail
- [ ] Analyze Sharpe progression over time
- [ ] Review drawdown periods and recoveries
- [ ] Check long vs short attribution
- [ ] Verify turnover and costs are stable
- [ ] Review kill switch history and justifications
- [ ] Update monthly summary document
- [ ] Compare actual vs backtest expectations
- [ ] Make go/no-go assessment
- [ ] Plan for next month

**Time commitment**: 1 hour per month

---

## After 60 Trading Days

### Final Decision Checklist

- [ ] Total trading days: _____ (must be 60+)

- [ ] Final metrics calculated:
  - Sharpe: _____ (must be > 1.0)
  - Annual Return: _____ %
  - Volatility: _____ %
  - Max Drawdown: _____ % (must be > -10%)
  - Kill Switch %: _____ % (must be < 15%)

- [ ] All success criteria met?
  - [ ] Sharpe > 1.0 ✅/❌
  - [ ] MaxDD < -10% ✅/❌
  - [ ] No systematic issues ✅/❌
  - [ ] Kill switches < 15% ✅/❌
  - [ ] Costs < 7 bps ✅/❌
  - [ ] Performance stable (not degrading) ✅/❌

- [ ] Generate final report and archive results

- [ ] Update STRATEGY_DEFINITION.md with Phase 4 results

### If ALL criteria met (GREEN LIGHT):

- [ ] Prepare for Phase 5 (live deployment)
- [ ] Start with 10% capital
- [ ] Follow phased deployment plan
- [ ] 🎉 Congratulations! Strategy validated!

### If criteria NOT met (RED LIGHT):

- [ ] Document what went wrong
- [ ] Investigate root cause
- [ ] Decide: Fix / Extend / Abandon
- [ ] DO NOT proceed to live trading
- [ ] DO NOT tweak parameters

---

## Emergency Stop Conditions

**HALT immediately if any of these occur:**

- [ ] Sharpe < 0.5 for 2 consecutive weeks
- [ ] MaxDD < -15% at any point
- [ ] Kill switches > 25% of days
- [ ] Systematic data or execution errors
- [ ] Costs consistently > 10 bps

**Action**: Stop paper trading, investigate, document findings

---

## Common Questions

**Q: I missed a day. What should I do?**
A: Note it in your journal. One missed day is OK. Multiple missed days may invalidate the test.

**Q: Results are worse than backtest. Should I stop?**
A: Not necessarily. As long as Sharpe > 1.0, continue. Paper trading is about validation, not perfection.

**Q: Can I run multiple days at once to catch up?**
A: For paper trading (simulation), yes. But in true live paper trading, you must do it daily.

**Q: Kill switch triggered. Is that bad?**
A: Not necessarily. Check if it was justified. < 15% event rate is acceptable.

**Q: Should I adjust parameters if performance is lower than expected?**
A: NO. Strategy is frozen. If it's not working, that's valuable information.

---

## Support and Resources

- **PHASE4_GUIDE.md** - Comprehensive guide
- **PHASE4_DAILY_WORKFLOW.md** - Detailed daily procedures
- **STRATEGY_DEFINITION.md v2.0.0** - Frozen strategy specification
- **Historical simulation results** - `reports/phase4/` for reference

---

## Final Reminder

**Paper trading is about VALIDATION, not OPTIMIZATION.**

- Follow the process
- Track everything
- Be patient
- Let the data decide
- Don't skip steps

**Good luck! 🚀**

Once you've completed this checklist and run for 60+ days with Sharpe > 1.0, you'll be ready for Phase 5 (live deployment)!
