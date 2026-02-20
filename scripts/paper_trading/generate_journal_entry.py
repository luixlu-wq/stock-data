"""
Generate and append a pre-filled daily journal block for paper trading.

Default behavior:
  - Target date = tomorrow (calendar date + 1)
  - Append block to logs/paper_trading/JOURNAL.md

Usage:
  python scripts/paper_trading/generate_journal_entry.py
  python scripts/paper_trading/generate_journal_entry.py --date 2026-02-20
  python scripts/paper_trading/generate_journal_entry.py --dry-run
  python scripts/paper_trading/generate_journal_entry.py --force
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE4_DIR = PROJECT_ROOT / "data" / "processed" / "phase4"

PROGRESS_FILE = PHASE4_DIR / "paper_trading_progress.json"
SUMMARY_FILE = PHASE4_DIR / "phase4_paper_trading_summary.json"
DAILY_FILE = PHASE4_DIR / "phase4_paper_trading_daily.parquet"
JOURNAL_FILE = PROJECT_ROOT / "logs" / "paper_trading" / "JOURNAL.md"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_num(v: float) -> str:
    return f"{v:.4f}"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_latest_daily(path: Path) -> dict:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"No rows in daily parquet: {path}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    row = df.iloc[-1]
    return {k: row[k] for k in row.index}


def _next_historical_date(last_historical_date: str | None) -> str:
    if not last_historical_date:
        return "2025-04-01"
    dt = pd.to_datetime(last_historical_date, errors="coerce")
    if pd.isna(dt):
        return "2025-04-01"
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


def _target_day_label(days_completed: int) -> int:
    return int(days_completed) + 1


def _date_exists(journal_text: str, target_date: str) -> bool:
    pattern = re.compile(rf"^##\s+{re.escape(target_date)}\b", re.MULTILINE)
    return bool(pattern.search(journal_text))


def build_block(
    target_date: str,
    progress: dict,
    summary: dict,
    latest_daily: dict,
) -> str:
    day_label = _target_day_label(progress.get("days_completed", 0))
    next_hist = _next_historical_date(progress.get("last_historical_date"))

    latest_date = str(latest_daily["date"])
    ks1 = bool(latest_daily.get("ks1", False))
    ks2 = bool(latest_daily.get("ks2", False))
    ks3 = bool(latest_daily.get("ks3", False))
    ks_text = "None" if not (ks1 or ks2 or ks3) else f"KS1={ks1}, KS2={ks2}, KS3={ks3}"

    return (
        f"\n## {target_date} (Day {day_label} target, auto-generated)\n\n"
        f"**Run Intent**: Process next tradable historical date if available; otherwise record clean no-action.\n\n"
        f"**Starting State (from progress file)**:\n"
        f"- Days completed: {progress.get('days_completed', 'N/A')}\n"
        f"- Last historical date processed: {progress.get('last_historical_date', 'N/A')}\n"
        f"- Last calendar run date: {progress.get('last_calendar_date', 'N/A')}\n\n"
        f"**Pre-Run Plan**:\n"
        f"1. Refresh predictions first:\n"
        f"   ```bash\n"
        f"   python scripts/paper_trading/run_2026_paper_trading.py --sim-start {next_hist} --sim-end auto\n"
        f"   ```\n"
        f"2. Execute daily runner:\n"
        f"   ```bash\n"
        f"   python scripts/paper_trading/run_daily_paper_trading.py\n"
        f"   ```\n"
        f"3. If runner reports no new tradable date, mark today as no-action and do not force manual date.\n\n"
        f"**Carry-Forward Context (latest completed historical day: {latest_date})**:\n"
        f"- PnL Net: {_fmt_num(float(latest_daily['pnl_net']))} ({_fmt_pct(float(latest_daily['pnl_net']))})\n"
        f"- PnL Scaled: {_fmt_num(float(latest_daily['pnl_scaled']))} ({_fmt_pct(float(latest_daily['pnl_scaled']))})\n"
        f"- Long PnL: {_fmt_num(float(latest_daily['long_pnl']))}\n"
        f"- Short PnL: {_fmt_num(float(latest_daily['short_pnl']))}\n"
        f"- Turnover: {_fmt_pct(float(latest_daily['turnover']))}\n"
        f"- Vol Scale: {float(latest_daily['vol_scale']):.2f}\n"
        f"- Kill Switches: {ks_text}\n"
        f"- Num Longs: {int(latest_daily['num_longs'])}\n"
        f"- Num Shorts: {int(latest_daily['num_shorts'])}\n\n"
        f"**Cumulative Baseline Before {target_date} Run**:\n"
        f"- Scaled Sharpe: {float(summary['sharpe_scaled']):.2f}\n"
        f"- Annual Return (Scaled): {_fmt_pct(float(summary['annual_return_scaled']))}\n"
        f"- Volatility (Scaled): {_fmt_pct(float(summary['vol_scaled']))}\n"
        f"- Max Drawdown: {_fmt_pct(float(summary['max_dd']))}\n"
        f"- KS Events: {int(summary['ks_events'])} ({float(summary['ks_pct']):.2f}% of days)\n\n"
        f"**Post-Run Fill-In (complete after execution)**:\n"
        f"- Execution outcome: [Success with new day / No action taken / Failed]\n"
        f"- Processed historical date: [YYYY-MM-DD or N/A]\n"
        f"- Today PnL Net: [ ]\n"
        f"- Today PnL Scaled: [ ]\n"
        f"- Today Turnover: [ ]\n"
        f"- Today Vol Scale: [ ]\n"
        f"- Kill switch status: [ ]\n"
        f"- Updated cumulative Sharpe: [ ]\n"
        f"- Updated days completed: [ ]\n\n"
        f"**Observations**:\n"
        f"- [ ]\n\n"
        f"**Concerns**:\n"
        f"- [ ]\n\n"
        f"**Next-Day Action**:\n"
        f"- [Run normal daily pipeline / Refresh predictions first / No action expected]\n\n"
        f"---\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and append a pre-filled paper-trading journal entry.")
    parser.add_argument("--date", type=str, default=None, help="Target calendar date YYYY-MM-DD (default: tomorrow).")
    parser.add_argument("--journal", type=str, default=str(JOURNAL_FILE), help="Path to JOURNAL.md")
    parser.add_argument("--dry-run", action="store_true", help="Print block only; do not write file.")
    parser.add_argument("--force", action="store_true", help="Append even if target date entry already exists.")
    args = parser.parse_args()

    target_date = args.date or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    journal_path = Path(args.journal)

    for required in [PROGRESS_FILE, SUMMARY_FILE, DAILY_FILE, journal_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    progress = _load_json(PROGRESS_FILE)
    summary = _load_json(SUMMARY_FILE)
    latest_daily = _load_latest_daily(DAILY_FILE)

    with open(journal_path, "r", encoding="utf-8") as f:
        journal_text = f.read()

    if _date_exists(journal_text, target_date) and not args.force:
        print(f"Entry for {target_date} already exists in {journal_path}. Use --force to append anyway.")
        return

    block = build_block(target_date, progress, summary, latest_daily)

    if args.dry_run:
        print(block)
        return

    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(block)

    print(f"Appended journal entry for {target_date} to {journal_path}")


if __name__ == "__main__":
    main()

