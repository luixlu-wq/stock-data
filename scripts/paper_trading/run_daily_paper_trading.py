"""
Phase 4: Daily Paper Trading Runner (Simplified)

Simple wrapper to run paper trading with historical data mapped to today's date.

Usage:
    # Automatic: Uses today's date and maps to next historical date
    python scripts/paper_trading/run_daily_paper_trading.py

    # Manual: Specify which historical date to use
    python scripts/paper_trading/run_daily_paper_trading.py --historical-date 2025-04-01
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.paper_trading.phase4_paper_trading_runner import PaperTradingRunner

# Mapping: Your calendar date → Historical date
# This allows you to "replay" historical data as if it's happening in real-time
HISTORICAL_START_DATE = "2025-04-01"
YOUR_START_DATE = "2026-01-19"
PHASE1_PREDICTIONS = Path("data/processed/phase1_predictions.parquet")
COMBINED_PREDICTIONS = Path("data/processed/phase4/predictions_combined.parquet")


def resolve_predictions_path(cli_path: str | None) -> Path:
    """Resolve predictions file path, preferring combined 2025+2026 predictions."""
    if cli_path:
        return Path(cli_path)
    if COMBINED_PREDICTIONS.exists():
        return COMBINED_PREDICTIONS
    return PHASE1_PREDICTIONS


def _normalize_dates(date_series: pd.Series) -> pd.Series:
    return pd.to_datetime(date_series, errors='coerce').dt.tz_localize(None).dt.strftime('%Y-%m-%d')


def load_available_dates(predictions_path: Path) -> tuple[list[str], list[str]]:
    """
    Load date ranges from predictions.

    Returns:
        (all_dates, tradable_dates)
        tradable_dates excludes dates where y_true_reg is entirely missing.
    """
    try:
        df = pd.read_parquet(predictions_path, columns=['date', 'y_true_reg'])
    except Exception:
        df = pd.read_parquet(predictions_path, columns=['date'])

    df = df.copy()
    df['date'] = _normalize_dates(df['date'])
    df = df.dropna(subset=['date'])

    all_dates = sorted(df['date'].unique().tolist())
    if not all_dates:
        raise ValueError(f"No valid dates found in predictions file: {predictions_path}")

    if 'y_true_reg' in df.columns:
        by_date_has_truth = df.groupby('date')['y_true_reg'].apply(lambda s: s.notna().any())
        tradable_dates = sorted(by_date_has_truth[by_date_has_truth].index.tolist())
    else:
        tradable_dates = all_dates

    if not tradable_dates:
        raise ValueError(f"No tradable dates (with y_true_reg) found in predictions file: {predictions_path}")

    return all_dates, tradable_dates


def build_refresh_command(max_date: str) -> str:
    """Generate command to refresh predictions beyond the current max date."""
    next_start = (pd.to_datetime(max_date) + timedelta(days=1)).strftime('%Y-%m-%d')
    return (
        f"python scripts/paper_trading/run_2026_paper_trading.py "
        f"--sim-start {next_start} --sim-end auto"
    )


def get_next_historical_date(progress_file: Path, predictions_path: Path) -> tuple[str | None, str, str, str]:
    """
    Get the next historical date to run based on progress.
    Skips weekends/holidays by looking at actual available dates in predictions.

    Returns:
        (next_date_or_none, min_tradable_date, max_tradable_date, max_raw_date)
    """
    all_dates, tradable_dates = load_available_dates(predictions_path)
    min_date = tradable_dates[0]
    max_date = tradable_dates[-1]
    max_raw_date = all_dates[-1]

    if not progress_file.exists():
        return min_date, min_date, max_date, max_raw_date

    with open(progress_file, 'r') as f:
        progress = json.load(f)

    last_date = progress.get('last_historical_date')
    if not last_date:
        return min_date, min_date, max_date, max_raw_date

    # Find the next date after last_date (strip timezone for comparison)
    last_dt = pd.to_datetime(last_date, errors='coerce')
    if pd.isna(last_dt):
        return min_date, min_date, max_date, max_raw_date
    last_dt = last_dt.tz_localize(None)
    max_dt = pd.to_datetime(max_date)
    if last_dt > max_dt:
        last_dt = max_dt

    for d in tradable_dates:
        d_naive = pd.to_datetime(d)
        if d_naive > last_dt:
            return d_naive.strftime('%Y-%m-%d'), min_date, max_date, max_raw_date

    return None, min_date, max_date, max_raw_date


def update_progress(progress_file: Path, historical_date: str, calendar_date: str):
    """
    Update progress tracker.
    """
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {
            'start_calendar_date': YOUR_START_DATE,
            'start_historical_date': HISTORICAL_START_DATE,
            'days_completed': 0
        }

    previous_date = progress.get('last_historical_date')
    progress['last_calendar_date'] = calendar_date
    progress['last_historical_date'] = historical_date
    if previous_date != historical_date:
        progress['days_completed'] = progress.get('days_completed', 0) + 1

    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Daily Paper Trading Runner')
    parser.add_argument('--historical-date', type=str, default=None,
                       help='Historical date to use (YYYY-MM-DD). If not provided, auto-increments.')
    parser.add_argument('--predictions', type=str,
                       default=None,
                       help='Path to predictions file (default: combined 2025+2026 if available, else phase1)')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed/phase4',
                       help='Output directory')

    args = parser.parse_args()

    # Progress tracking file
    progress_file = Path("data/processed/phase4/paper_trading_progress.json")
    predictions_path = resolve_predictions_path(args.predictions)

    if not predictions_path.exists():
        print(f"ERROR: Predictions file not found: {predictions_path}")
        sys.exit(1)

    all_dates, available_dates = load_available_dates(predictions_path)
    min_available = available_dates[0]
    max_available = available_dates[-1]
    max_raw_date = all_dates[-1]

    # Determine which historical date to run
    if args.historical_date:
        historical_date = args.historical_date
        if historical_date not in set(available_dates):
            print(f"ERROR: No data for date {historical_date} in {predictions_path}")
            print(f"Available tradable range: {min_available} to {max_available}")
            if max_raw_date > max_available:
                print(f"Note: newest raw prediction date is {max_raw_date}, but it has no y_true_reg yet.")
            sys.exit(1)
    else:
        historical_date, min_available, max_available, max_raw_date = get_next_historical_date(
            progress_file, predictions_path
        )
        if historical_date is None:
            today = datetime.now().strftime('%Y-%m-%d')
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            print("="*60)
            print("DAILY PAPER TRADING")
            print("="*60)
            print(f"Calendar Date (Today): {today}")
            print("No new historical trading date available.")
            print(f"Predictions: {predictions_path}")
            print(f"Available tradable range: {min_available} to {max_available}")
            if max_raw_date > max_available:
                print(f"Newest raw prediction date: {max_raw_date} (not tradable yet, y_true_reg missing)")
            if max_available < yesterday:
                print("Predictions look stale. Refresh command:")
                print(f"  {build_refresh_command(max_available)}")
            print("No action taken.")
            print("="*60)
            return

    # Today's calendar date
    calendar_date = datetime.now().strftime('%Y-%m-%d')

    print("="*60)
    print(f"DAILY PAPER TRADING")
    print("="*60)
    print(f"Calendar Date (Today): {calendar_date}")
    print(f"Historical Date (Simulation): {historical_date}")
    print("="*60)

    # Run paper trading for this historical date
    runner = PaperTradingRunner(
        predictions_path=str(predictions_path),
        output_dir=args.output_dir
    )

    result = runner.run_single_day(historical_date)

    if result:
        print("\n" + "="*60)
        print("TODAY'S RESULTS")
        print("="*60)
        print(f"PnL Net: {result['pnl_net']:.4f} ({result['pnl_net']*100:.2f}%)")
        print(f"PnL Scaled (Vol-Targeted): {result['pnl_scaled']:.4f} ({result['pnl_scaled']*100:.2f}%)")
        print(f"Long PnL: {result['long_pnl']:.4f}")
        print(f"Short PnL: {result['short_pnl']:.4f}")
        print(f"Turnover: {result['turnover']:.2%}")
        print(f"Vol Scale: {result['vol_scale']:.2f}")
        print(f"Cost: {result['cost']:.6f} ({result['cost']*10000:.2f} bps)")
        print(f"\nPortfolio:")
        print(f"  Longs: {result['num_longs']}")
        print(f"  Shorts: {result['num_shorts']}")
        print(f"\nKill Switches:")
        print(f"  KS1 (3-sigma loss): {'TRIGGERED' if result['ks1'] else 'OK'}")
        print(f"  KS2 (8% DD): {'TRIGGERED' if result['ks2'] else 'OK'}")
        print(f"  KS3 (Sharpe < 0): {'TRIGGERED' if result['ks3'] else 'OK'}")

        if result['ks_triggered']:
            print("\nWARNING: Kill switch triggered!")

        # Update progress
        update_progress(progress_file, historical_date, calendar_date)

        # Load and display cumulative stats
        try:
            with open(Path(args.output_dir) / "phase4_paper_trading_summary.json", 'r') as f:
                summary = json.load(f)

            print(f"\n" + "="*60)
            print(f"CUMULATIVE PERFORMANCE ({summary['num_days']} days)")
            print("="*60)
            print(f"Vol-Targeted Sharpe: {summary['sharpe_scaled']:.2f} (Target: > 1.0)")
            print(f"Annual Return: {summary['annual_return_scaled']:.2%}")
            print(f"Volatility: {summary['vol_scaled']:.2%}")
            print(f"Max Drawdown: {summary['max_dd']:.2%}")
            print(f"Kill Switch Events: {summary['ks_events']} ({summary['ks_pct']:.1f}% of days)")

            # Deployment gate check
            print(f"\nDeployment Gates:")
            sharpe_ok = summary['sharpe_scaled'] > 1.0
            dd_ok = summary['max_dd'] > -0.10
            ks_ok = summary['ks_pct'] < 15.0

            print(f"  Sharpe > 1.0: {'PASS' if sharpe_ok else 'FAIL'}")
            print(f"  MaxDD < -10%: {'PASS' if dd_ok else 'FAIL'}")
            print(f"  Kill Switches < 15%: {'PASS' if ks_ok else 'FAIL'}")

            if sharpe_ok and dd_ok and ks_ok:
                print(f"\nON TRACK for live deployment.")
            else:
                print(f"\nReview needed - some gates not met")

        except Exception as e:
            print(f"\nCould not load cumulative stats: {e}")

        print("="*60)

        # Show next steps
        print("\nNext Steps:")
        print("1. Update your journal: logs/paper_trading/JOURNAL.md")
        print("2. Tomorrow: Run this script again (it will auto-increment)")
        print("3. Friday: Run performance tracker for weekly review")

    else:
        print(f"\nERROR: Failed to run paper trading for {historical_date}")
        print("Check if this date exists in the historical data")
        sys.exit(1)


if __name__ == "__main__":
    main()
