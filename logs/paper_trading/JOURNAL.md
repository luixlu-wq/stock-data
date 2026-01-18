# Phase 4 Paper Trading Journal

**Start Date**: [To be filled when you start]

**Target End Date**: [Start + 60 trading days]

**Strategy**: S2_FilterNegative (v2.0.0)

---

## Daily Log

### Template (Copy for each day)

```markdown
## YYYY-MM-DD (Day X of 60+)

**Market Conditions**: [Normal / Volatile / Trending Up / Trending Down / Other]

**Results**:
- PnL Net: X.XXXX (X.XX%)
- PnL Scaled: X.XXXX (X.XX%)
- Long PnL: X.XXXX
- Short PnL: X.XXXX
- Turnover: XX.X%
- Vol Scale: X.XX
- Kill Switches: [None / KS1 / KS2 / KS3 / Multiple]

**Portfolio Stats**:
- Num Longs: XX
- Num Shorts: XX
- Gross Exposure: 100%
- Net Exposure: ~30%

**Observations**:
- [What happened today?]
- [Any unusual market moves?]
- [Strategy behavior as expected?]
- [Long vs short performance?]

**Concerns**: [None / List any]

**Action Items**: [None / List any]

---
```

## Example Entry

### 2026-01-18 (Day 1 of 60+)

**Market Conditions**: Normal - Mixed trading, moderate volatility

**Results**:
- PnL Net: 0.0027 (0.27%)
- PnL Scaled: 0.0027 (0.27%)
- Long PnL: 0.0018
- Short PnL: 0.0009
- Turnover: 100.00% (first day - full position establishment)
- Vol Scale: 1.00 (not enough history yet)
- Kill Switches: None

**Portfolio Stats**:
- Num Longs: 38
- Num Shorts: 15 (filtered from 38 candidates)
- Gross Exposure: 100%
- Net Exposure: 30%

**Observations**:
- First day of paper trading went smoothly
- Short filter working - only 15 shorts qualified (needed y_pred < 0)
- Both long and short sides contributed positively
- 100% turnover expected on day 1 (establishing positions)

**Concerns**: None - everything as expected

**Action Items**:
- Continue daily execution
- Monitor turnover tomorrow (should drop to ~15-20%)

---

## Start your daily entries below this line

---
