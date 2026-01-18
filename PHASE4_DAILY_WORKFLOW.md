# Phase 4: Daily Paper Trading Workflow

**START DATE**: [To be determined - when you're ready to begin]

**DURATION**: 60+ trading days (approximately 3 months)

**GOAL**: Validate S2_FilterNegative strategy with real market data before live deployment

---

## Prerequisites Setup (One-Time)

Before starting paper trading, complete these setup steps:

### 1. Verify Python Environment

```bash
cd c:\Users\luixj\AI\stock-data
venv\Scripts\activate

# Install any missing dependencies
pip install matplotlib yfinance pandas numpy torch scikit-learn pyarrow
```

### 2. Verify Model Files

```bash
# Check that the production model exists
ls models/checkpoints/lstm_phase2a_temp0.05_best.pth
```

### 3. Create Daily Log Directory

```bash
mkdir -p logs/paper_trading
mkdir -p data/paper_trading/daily
mkdir -p reports/paper_trading/daily
```

### 4. Test the System (Dry Run)

```bash
# Run a single day from historical data to verify everything works
python scripts/paper_trading/phase4_paper_trading_runner.py --start-date 2025-12-01 --end-date 2025-12-01
```

If this completes successfully, you're ready to start!

---

## Daily Workflow (Every Trading Day)

### Morning Routine (Before Market Open)

**Time**: 9:00 AM EST

**Duration**: 5 minutes

1. **Check market status**:
   - Is the market open today? (no weekends, no holidays)
   - Any major news that might affect strategy?

2. **Review yesterday's results** (if not first day):
   ```bash
   # View yesterday's summary
   tail -20 logs/paper_trading/paper_trading_$(date -d yesterday +%Y%m%d).log
   ```

3. **Prepare for today**:
   - Set a reminder for end-of-day execution (4:15 PM EST)

### End-of-Day Execution (After Market Close)

**Time**: 4:15 PM EST (15 minutes after market close)

**Duration**: 10-15 minutes

**CRITICAL**: This is when you run the daily paper trading simulation.

#### Step 1: Fetch Today's Data and Generate Predictions

Since we're doing paper trading (not live trading), we need to:
1. Fetch today's market data
2. Run the model to generate predictions
3. Simulate what orders we would have placed

**For now, we'll use the historical simulation approach** since we already have predictions from Phase 1. Once you're ready for true live paper trading, you'll need to:
- Set up daily data fetching
- Run model inference
- Generate new predictions daily

For the current approach (using historical data as proxy):

```bash
# Navigate to project directory
cd c:\Users\luixj\AI\stock-data
venv\Scripts\activate

# Run paper trading for a specific date range
# Replace DATE with actual date
python scripts/paper_trading/phase4_paper_trading_runner.py --start-date 2025-12-30 --end-date 2025-12-30
```

#### Step 2: Review Today's Results

Check the log output for:

- **PnL Net**: Today's net profit/loss
- **PnL Scaled**: Vol-targeted profit/loss
- **Turnover**: Portfolio turnover percentage
- **Vol Scale**: Volatility scaling factor
- **Kill Switches**: Any triggered? (ks1, ks2, ks3)

**Example output**:
```
PnL Net: 0.0012 | Scaled: 0.0018 | Turnover: 17.23% | Vol Scale: 1.45
```

**What to look for**:
- ✅ PnL positive or negative (both are normal)
- ✅ Turnover 15-25% (typical range)
- ✅ Vol scale 0.5-2.0 (within bounds)
- ⚠️ Kill switch triggered (note in journal)

#### Step 3: Update Your Trading Journal

Create a file: `logs/paper_trading/JOURNAL.md`

Add an entry for today:

```markdown
## 2026-01-XX

**Market Conditions**: [Normal / Volatile / Trending Up / Trending Down]

**Results**:
- PnL Net: X.XX%
- PnL Scaled: X.XX%
- Turnover: XX.X%
- Kill Switches: [None / KS1 / KS2 / KS3]

**Observations**:
- [Any notable market events?]
- [Did performance match expectations?]
- [Any execution concerns?]

**Concerns**: [None / List any concerns]

**Action Items**: [None / List any follow-ups]
```

#### Step 4: Quick Health Check

Run these commands to check overall health:

```bash
# Count total trading days so far
python -c "import pandas as pd; df = pd.read_parquet('data/processed/phase4/phase4_paper_trading_daily.parquet'); print(f'Total days: {len(df)}')"

# Check latest summary metrics
cat data/processed/phase4/phase4_paper_trading_summary.json
```

**Decision Point**: Are we still on track?
- Sharpe > 1.0? ✅
- MaxDD < -10%? ✅
- No systematic issues? ✅

If YES to all → Continue
If NO to any → Investigate before continuing

---

## Weekly Review (Every Friday)

**Time**: End of day Friday

**Duration**: 30 minutes

### Step 1: Generate Performance Report

```bash
python scripts/paper_trading/phase4_performance_tracker.py
```

This creates:
- `reports/phase4/phase4_performance_report.txt`
- `reports/phase4/phase4_performance_plots.png`

### Step 2: Review Key Metrics

Open the report and check:

1. **Sharpe Ratio**:
   - Current: X.XX
   - Target: > 1.0
   - Status: ✅/❌

2. **Weekly PnL**:
   - This week: X.XX%
   - Cumulative: X.XX%

3. **Kill Switches**:
   - This week: X events
   - Total: X events (X.X%)
   - Acceptable: < 15%

4. **Alerts**:
   - Any RED alerts? → STOP and investigate
   - Any YELLOW alerts? → Monitor closely

### Step 3: Compare vs Backtest

Review the comparison table:
- Are metrics within 20% of expectations?
- Any systematic deviations?
- Sharpe still > 1.0?

### Step 4: Update Weekly Log

Add to `logs/paper_trading/WEEKLY_SUMMARY.md`:

```markdown
## Week of 2026-01-XX

**Days Traded**: X
**Total Days**: XX

**Performance**:
- Weekly Sharpe: X.XX
- Weekly Return: X.XX%
- Cumulative Sharpe: X.XX
- Cumulative Return: X.XX%

**Kill Switches**: X events (X.X% of days)

**Status**: ✅ On Track / ⚠️ Review Needed / 🚨 Issues Detected

**Key Observations**:
- [What went well?]
- [What didn't go well?]
- [Any concerns?]

**Action Items**:
- [ ] [Any follow-up tasks]
```

### Step 5: Decision Point

**After 4 weeks (~20 trading days)**:
- If Sharpe > 1.0 and no issues → Continue to 60 days
- If Sharpe < 1.0 → Investigate root cause
- If systematic issues → HALT and review

**After 8 weeks (~40 trading days)**:
- If Sharpe > 1.0 consistently → Continue to 60 days
- If deteriorating → Investigate or abort

**After 12 weeks (~60 trading days)**:
- If Sharpe > 1.0 → GREEN LIGHT for Phase 5
- If Sharpe < 1.0 → DO NOT proceed to live

---

## Monthly Review (End of Each Month)

**Time**: Last trading day of the month

**Duration**: 1 hour

### Step 1: Comprehensive Performance Analysis

```bash
# Generate full report
python scripts/paper_trading/phase4_performance_tracker.py

# Review plots
open reports/phase4/phase4_performance_plots.png
```

### Step 2: Deep Dive Analysis

Review each metric in detail:

1. **Sharpe Progression**:
   - Month 1: X.XX
   - Month 2: X.XX
   - Month 3: X.XX
   - Trend: Stable / Improving / Degrading

2. **Drawdown Analysis**:
   - Largest DD this month: -X.XX%
   - When did it occur?
   - How long to recover?

3. **Long vs Short**:
   - Long Sharpe: X.XX
   - Short Sharpe: X.XX
   - Is short filter working?

4. **Turnover & Costs**:
   - Avg turnover: XX.X%
   - Avg cost: X.X bps
   - Within budget?

5. **Kill Switches**:
   - KS1 events: X
   - KS2 events: X
   - KS3 events: X
   - Total: X (X.X%)
   - Are they justified?

### Step 3: Update Monthly Summary

Create `logs/paper_trading/MONTHLY_SUMMARY.md`:

```markdown
## Month X: [Month Name]

**Trading Days**: XX
**Total Days to Date**: XXX

**Performance Summary**:
- Monthly Sharpe: X.XX
- Monthly Return: X.XX%
- Monthly MaxDD: -X.XX%
- Monthly Volatility: X.XX%

**Cumulative Performance**:
- Sharpe to Date: X.XX
- Total Return: X.XX%
- Max Drawdown: -X.XX%
- Kill Switch %: X.X%

**Comparison to Backtest**:
| Metric | Backtest | Actual | Diff |
|--------|----------|--------|------|
| Sharpe | 1.29 | X.XX | +/-X.XX |
| Return | 11.95% | X.XX% | +/-X.XX% |
| Vol | 9.27% | X.XX% | +/-X.XX% |
| MaxDD | -5.21% | -X.XX% | +/-X.XX% |

**Key Insights**:
- [What did we learn this month?]
- [Any surprises?]
- [Strategy working as expected?]

**Concerns**:
- [List any concerns]

**Go/No-Go Assessment**:
- [ ] Sharpe > 1.0? ✅/❌
- [ ] MaxDD < -10%? ✅/❌
- [ ] No systematic issues? ✅/❌
- [ ] Costs within budget? ✅/❌

**Decision**:
- ✅ Continue to Phase 5
- ⚠️ Continue monitoring
- 🚨 HALT - investigate issues
```

### Step 4: Prepare for Next Month

- Review market calendar (holidays, etc.)
- Any strategy adjustments needed? (Answer: NO - strategy is frozen)
- Set goals for next month

---

## Red Flags to Watch For

### STOP IMMEDIATELY if:

1. **Sharpe < 0.5 for 2 consecutive weeks**
   - Action: HALT paper trading
   - Investigate: Why is performance so bad?
   - Consider: Strategy may be broken

2. **MaxDD < -15%**
   - Action: HALT paper trading
   - Investigate: Risk management failure
   - Review: Kill switch logic

3. **Kill switches triggering > 25% of days**
   - Action: Review kill switch thresholds
   - Investigate: Are they too tight? Or is strategy broken?

4. **Systematic execution issues**
   - Data quality problems
   - Model failures
   - Feature calculation errors

5. **Costs consistently > 10 bps**
   - Action: Review turnover
   - Investigate: Why so high?

### YELLOW FLAGS (Monitor Closely):

1. Sharpe 0.8-1.0 for multiple weeks
2. MaxDD -8% to -10%
3. Kill switches 15-25% of days
4. Costs 6-8 bps
5. Significant deviation from backtest (>30%)

---

## After 60 Days: Final Decision

### Success Criteria (ALL must be met):

- ✅ Sharpe > 1.0 (preferably > 1.15)
- ✅ MaxDD < -10%
- ✅ No systematic execution issues
- ✅ Kill switches < 15% of days
- ✅ Costs < 7 bps
- ✅ Stable performance (not deteriorating)

### If ALL criteria met → Proceed to Phase 5

Create final report:

```bash
# Generate final report
python scripts/paper_trading/phase4_performance_tracker.py

# Archive results
cp -r data/processed/phase4 data/archives/phase4_complete_$(date +%Y%m%d)
cp -r reports/phase4 reports/archives/phase4_complete_$(date +%Y%m%d)
```

Update [STRATEGY_DEFINITION.md](STRATEGY_DEFINITION.md):
- Change status to "PAPER TRADING COMPLETE - READY FOR LIVE"
- Add Phase 4 results section

**Next step**: Begin Phase 5 with 10% capital

### If criteria NOT met → DO NOT PROCEED

Document why:
- Which criteria failed?
- What went wrong?
- Is it fixable?
- Should we abandon the strategy?

**Options**:
1. Investigate and fix (if possible)
2. Extend paper trading (another 30-60 days)
3. Abandon strategy (if fundamentally broken)

**DO NOT**:
- ❌ Tweak parameters to "fix" it
- ❌ Retrain the model
- ❌ Add new features
- ❌ Proceed to live anyway

---

## Quick Reference Commands

```bash
# Daily execution
python scripts/paper_trading/phase4_paper_trading_runner.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD

# Generate performance report
python scripts/paper_trading/phase4_performance_tracker.py

# View latest summary
cat data/processed/phase4/phase4_paper_trading_summary.json

# Count trading days
python -c "import pandas as pd; df = pd.read_parquet('data/processed/phase4/phase4_paper_trading_daily.parquet'); print(f'Days: {len(df)}, Sharpe: {df[\"pnl_scaled\"].mean() / df[\"pnl_scaled\"].std() * (252**0.5):.2f}')"

# View recent daily results
python -c "import pandas as pd; df = pd.read_parquet('data/processed/phase4/phase4_paper_trading_daily.parquet'); print(df.tail(5)[['date', 'pnl_net', 'pnl_scaled', 'turnover', 'ks_triggered']])"
```

---

## Important Reminders

1. **Run EVERY trading day** - No skipping days
2. **Run AFTER market close** - Need full day's data
3. **Keep detailed journal** - Document everything
4. **NO parameter changes** - Strategy is frozen
5. **Monitor Sharpe > 1.0** - Primary success metric
6. **Watch kill switches** - Should be < 15% of days
7. **Compare vs backtest** - Weekly reviews
8. **Be patient** - Need 60+ days for statistical significance

---

## When to Reach Out for Help

Contact me (or your advisor) if:

- Sharpe drops below 0.8
- MaxDD exceeds -12%
- Systematic issues detected
- Kill switches > 20% of days
- Unsure how to interpret results
- Major market event (crash, etc.)

**Remember**: Paper trading is cheap. Real losses are expensive. Take your time and do it right!

---

**Good luck with your paper trading! 🚀**

Track everything, stay disciplined, and let the data tell you when you're ready for live deployment.
