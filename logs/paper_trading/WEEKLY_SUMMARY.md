# Phase 4 Paper Trading - Weekly Summaries

**Strategy**: S2_FilterNegative (v2.0.0)

**Target**: 60+ trading days (~12 weeks)

---

## Template (Copy for each week)

```markdown
## Week X: [Start Date] to [End Date]

**Days Traded This Week**: X
**Total Days to Date**: XX / 60+

**Performance This Week**:
- Weekly PnL: X.XX%
- Daily Avg PnL: X.XX%
- Best Day: X.XX% (Date)
- Worst Day: -X.XX% (Date)
- Weekly Sharpe: X.XX

**Cumulative Performance**:
- Total PnL: X.XX%
- Cumulative Sharpe: X.XX
- Max Drawdown: -X.XX%
- Current Drawdown: -X.XX%

**Execution Metrics**:
- Avg Turnover: XX.X%
- Avg Cost: X.X bps
- Avg Vol Scale: X.XX

**Kill Switches**:
- This Week: X events
- Total to Date: X events (X.X% of days)
- KS1 (3-sigma loss): X
- KS2 (8% DD): X
- KS3 (Sharpe < 0): X

**Portfolio Attribution**:
- Long Sharpe: X.XX
- Short Sharpe: X.XX
- Short Filter: XX avg shorts/day (from 38 candidates)

**Status**:
- ✅ On Track (Sharpe > 1.0, no issues)
- ⚠️ Review Needed (Sharpe 0.8-1.0 or yellow flags)
- 🚨 Issues Detected (Sharpe < 0.8 or red flags)

**Key Observations**:
- [What went well this week?]
- [What didn't go well?]
- [Any notable market conditions?]
- [Strategy behavior as expected?]

**Concerns**:
- [List any concerns, or "None"]

**Action Items**:
- [ ] [Any follow-up tasks]
- [ ] [Items to investigate]

**Comparison to Backtest** (if enough data):
- Expected Sharpe: 1.29 | Actual: X.XX | Diff: X.XX
- Expected Return: 11.95% annual | Actual: X.XX% | Diff: X.XX%
- Expected Vol: 9.27% | Actual: X.XX% | Diff: X.XX%

**Decision**:
- [Continue / Investigate / Halt]

---
```

## Example Week

### Week 1: 2026-01-18 to 2026-01-24

**Days Traded This Week**: 5
**Total Days to Date**: 5 / 60+

**Performance This Week**:
- Weekly PnL: 0.45%
- Daily Avg PnL: 0.09%
- Best Day: 0.27% (2026-01-18)
- Worst Day: -0.15% (2026-01-21)
- Weekly Sharpe: 1.15

**Cumulative Performance**:
- Total PnL: 0.45%
- Cumulative Sharpe: 1.15
- Max Drawdown: -0.08%
- Current Drawdown: 0%

**Execution Metrics**:
- Avg Turnover: 18.2%
- Avg Cost: 0.5 bps
- Avg Vol Scale: 1.02

**Kill Switches**:
- This Week: 0 events
- Total to Date: 0 events (0% of days)
- KS1 (3-sigma loss): 0
- KS2 (8% DD): 0
- KS3 (Sharpe < 0): 0

**Portfolio Attribution**:
- Long Sharpe: 0.85
- Short Sharpe: 0.45
- Short Filter: 18 avg shorts/day (from 38 candidates)

**Status**: ✅ On Track

**Key Observations**:
- Strong start to paper trading
- Sharpe above 1.0 target
- Short filter working well (only ~18 shorts qualifying per day)
- Turnover in expected range (15-20%)
- No kill switches triggered
- Both long and short sides contributing

**Concerns**: None - too early to draw conclusions, but good start

**Action Items**:
- [ ] Continue daily execution
- [ ] Monitor if pattern continues next week

**Comparison to Backtest**:
- Too early (need 20+ days for meaningful comparison)

**Decision**: Continue

---

## Start your weekly summaries below this line

---
