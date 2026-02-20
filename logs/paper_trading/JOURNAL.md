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

## 2026-02-20 (Day 197 target, pre-filled)

**Run Intent**: Process next tradable historical date if available; otherwise record clean no-action.

**Starting State (from progress file)**:
- Days completed: 196
- Last historical date processed: 2026-02-18
- Last calendar run date: 2026-02-19

**Pre-Run Plan**:
1. Refresh predictions first:
   ```bash
   python scripts/paper_trading/run_2026_paper_trading.py --sim-start 2026-02-19 --sim-end auto
   ```
2. Execute daily runner:
   ```bash
   python scripts/paper_trading/run_daily_paper_trading.py
   ```
3. If runner reports no new tradable date, mark today as no-action and do not force manual date.

**Carry-Forward Context (latest completed historical day: 2026-02-18)**:
- PnL Net: -0.0028 (-0.28%)
- PnL Scaled: -0.0051 (-0.51%)
- Long PnL: -0.0040
- Short PnL: 0.0013
- Turnover: 8.78%
- Vol Scale: 1.85
- Kill Switches: None (KS1/KS2/KS3 all false)
- Num Longs: 107
- Num Shorts: 82

**Cumulative Baseline Before 2026-02-20 Run**:
- Scaled Sharpe: 0.93
- Annual Return (Scaled): 8.27%
- Volatility (Scaled): 8.87%
- Max Drawdown: -4.72%
- KS Events: 15 (7.65% of days)

**Post-Run Fill-In (complete after execution)**:
- Execution outcome: [Success with new day / No action taken / Failed]
- Processed historical date: [YYYY-MM-DD or N/A]
- Today PnL Net: [ ]
- Today PnL Scaled: [ ]
- Today Turnover: [ ]
- Today Vol Scale: [ ]
- Kill switch status: [ ]
- Updated cumulative Sharpe: [ ]
- Updated days completed: [ ]

**Observations**:
- [ ]

**Concerns**:
- [ ]

**Next-Day Action**:
- [Run normal daily pipeline / Refresh predictions first / No action expected]

---

## Daily Operations Checklist (Effective 2026-02-20)

Use this checklist before writing each daily entry.

Optional helper to auto-generate the next entry:

```bash
python scripts/paper_trading/generate_journal_entry.py
```

### 1. Pre-Run Checks (3-5 minutes)

- [ ] Confirm date/time and session:
  - Calendar date: `YYYY-MM-DD`
  - Run window: after market close (4:15 PM ET or later)
- [ ] Confirm required files exist:
  - `data/processed/phase4/predictions_combined.parquet`
  - `data/processed/phase4/paper_trading_progress.json`
  - `data/processed/phase4/phase4_paper_trading_daily.parquet`
- [ ] Check latest processed historical date in progress file.
- [ ] If no new tradable date is expected, note that a no-action run is acceptable.

### 2. Data Refresh Step (Run first when needed)

```bash
python scripts/paper_trading/run_2026_paper_trading.py --sim-start <next-date> --sim-end auto
```

- [ ] Run when predictions are stale or when `run_daily_paper_trading.py` reports no new tradable date.
- [ ] Record refresh result (new max date, success/failure).

### 3. Daily Paper Run

```bash
python scripts/paper_trading/run_daily_paper_trading.py
```

- [ ] Record whether execution was:
  - [ ] Success with new day processed
  - [ ] No action taken (no new tradable date)
  - [ ] Failed (capture error)

### 4. Post-Run Metrics Capture

- [ ] PnL Net / PnL Scaled
- [ ] Turnover / Vol Scale
- [ ] Long PnL / Short PnL
- [ ] Kill Switch status (KS1/KS2/KS3)
- [ ] Updated cumulative:
  - Scaled Sharpe
  - Max Drawdown
  - KS event %
  - Days completed

### 5. Risk & Regime Notes

- [ ] Was today a low-conviction signal day? (e.g., unusually compressed prediction dispersion)
- [ ] Any unusual one-sided behavior (all/near-all predictions one sign)?
- [ ] Any system/data anomalies?

### 6. Escalation Rules (Document immediately if triggered)

- [ ] Trigger review if scaled Sharpe deteriorates materially over recent 10-day window.
- [ ] Trigger review if drawdown approaches/breaches internal warning levels.
- [ ] Trigger review if kill-switch frequency rises unexpectedly.
- [ ] Trigger review if prediction regime is abnormal for 3+ sessions.

### 7. End-of-Day Journal Completion

- [ ] Fill daily entry using template above.
- [ ] Add explicit next-day action:
  - `Run normal daily pipeline`
  - `Refresh predictions first`
  - `No action expected unless new tradable date appears`

---

## 2026-02-21 (Day 197 target, auto-generated)

**Run Intent**: Process next tradable historical date if available; otherwise record clean no-action.

**Starting State (from progress file)**:
- Days completed: 196
- Last historical date processed: 2026-02-18
- Last calendar run date: 2026-02-19

**Pre-Run Plan**:
1. Refresh predictions first:
   ```bash
   python scripts/paper_trading/run_2026_paper_trading.py --sim-start 2026-02-19 --sim-end auto
   ```
2. Execute daily runner:
   ```bash
   python scripts/paper_trading/run_daily_paper_trading.py
   ```
3. If runner reports no new tradable date, mark today as no-action and do not force manual date.

**Carry-Forward Context (latest completed historical day: 2026-02-18)**:
- PnL Net: -0.0028 (-0.28%)
- PnL Scaled: -0.0051 (-0.51%)
- Long PnL: -0.0040
- Short PnL: 0.0013
- Turnover: 8.78%
- Vol Scale: 1.85
- Kill Switches: None
- Num Longs: 107
- Num Shorts: 82

**Cumulative Baseline Before 2026-02-21 Run**:
- Scaled Sharpe: 0.93
- Annual Return (Scaled): 8.27%
- Volatility (Scaled): 8.87%
- Max Drawdown: -4.72%
- KS Events: 15 (7.65% of days)

**Post-Run Fill-In (complete after execution)**:
- Execution outcome: [Success with new day / No action taken / Failed]
- Processed historical date: [YYYY-MM-DD or N/A]
- Today PnL Net: [ ]
- Today PnL Scaled: [ ]
- Today Turnover: [ ]
- Today Vol Scale: [ ]
- Kill switch status: [ ]
- Updated cumulative Sharpe: [ ]
- Updated days completed: [ ]

**Observations**:
- [ ]

**Concerns**:
- [ ]

**Next-Day Action**:
- [Run normal daily pipeline / Refresh predictions first / No action expected]

---
