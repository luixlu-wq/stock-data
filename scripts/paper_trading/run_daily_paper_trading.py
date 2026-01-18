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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.paper_trading.phase4_paper_trading_runner import PaperTradingRunner

# Mapping: Your calendar date → Historical date
# This allows you to "replay" historical data as if it's happening in real-time
HISTORICAL_START_DATE = "2025-04-01"
YOUR_START_DATE = "2026-01-19"


def get_next_historical_date(progress_file: Path) -> str:
    """
    Get the next historical date to run based on progress.
    """
    if not progress_file.exists():
        return HISTORICAL_START_DATE

    with open(progress_file, 'r') as f:
        progress = json.load(f)

    last_date = progress.get('last_historical_date', HISTORICAL_START_DATE)

    # Find next date in the historical data
    from datetime import timedelta
    import pandas as pd

    last_dt = pd.to_datetime(last_date)
    next_dt = last_dt + timedelta(days=1)

    # Keep incrementing until we find a trading day
    # (for simplicity, just increment - the runner will handle missing data)
    return next_dt.strftime('%Y-%m-%d')


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

    progress['last_calendar_date'] = calendar_date
    progress['last_historical_date'] = historical_date
    progress['days_completed'] = progress.get('days_completed', 0) + 1

    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Daily Paper Trading Runner')
    parser.add_argument('--historical-date', type=str, default=None,
                       help='Historical date to use (YYYY-MM-DD). If not provided, auto-increments.')
    parser.add_argument('--predictions', type=str,
                       default='data/processed/phase1_predictions.parquet',
                       help='Path to predictions file')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed/phase4',
                       help='Output directory')

    args = parser.parse_args()

    # Progress tracking file
    progress_file = Path("data/processed/phase4/paper_trading_progress.json")

    # Determine which historical date to run
    if args.historical_date:
        historical_date = args.historical_date
    else:
        historical_date = get_next_historical_date(progress_file)

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
        predictions_path=args.predictions,
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
            print("\n⚠️  WARNING: Kill switch triggered!")

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

            print(f"  Sharpe > 1.0: {'✅ PASS' if sharpe_ok else '❌ FAIL'}")
            print(f"  MaxDD < -10%: {'✅ PASS' if dd_ok else '❌ FAIL'}")
            print(f"  Kill Switches < 15%: {'✅ PASS' if ks_ok else '❌ FAIL'}")

            if sharpe_ok and dd_ok and ks_ok:
                print(f"\n✅ ON TRACK for live deployment!")
            else:
                print(f"\n⚠️  Review needed - some gates not met")

        except Exception as e:
            print(f"\nCould not load cumulative stats: {e}")

        print("="*60)

        # Show next steps
        print("\nNext Steps:")
        print("1. Update your journal: logs/paper_trading/JOURNAL.md")
        print("2. Tomorrow: Run this script again (it will auto-increment)")
        print("3. Friday: Run performance tracker for weekly review")

    else:
        print(f"\n❌ ERROR: Failed to run paper trading for {historical_date}")
        print("Check if this date exists in the historical data")
        sys.exit(1)


if __name__ == "__main__":
    main()
