"""
Phase 4: Paper Trading Daily Runner

This script simulates daily production execution for paper trading validation.
Runs the full S2_FilterNegative strategy pipeline and logs results for comparison
against backtest expectations.

Production Configuration (STRATEGY_DEFINITION.md v2.0.0):
- Portfolio: S2_FilterNegative (130/70 with y_pred < 0 filter)
- Vol Targeting: 8% annual (20-day lookback)
- Kill Switches: 3-sigma loss, 8% DD, Sharpe < 0
- Expected Sharpe: 1.29

Usage:
    python scripts/paper_trading/phase4_paper_trading_runner.py --date 2025-01-18
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Setup logging
log_dir = Path("logs/paper_trading")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants from STRATEGY_DEFINITION.md v2.0.0
K = 38  # Top/bottom K stocks
LONG_EXPOSURE = 0.65  # 130% of base
SHORT_EXPOSURE = 0.35  # 70% of base
ALPHA = 0.15  # EWMA smoothing parameter
TARGET_VOL = 0.08  # 8% annual volatility target
VOL_LOOKBACK = 20  # 20-day rolling vol
TRANSACTION_COST_BPS = 5  # 5 bps per trade


class PaperTradingRunner:
    """
    Runs the S2_FilterNegative strategy for one trading day in paper trading mode.
    """

    def __init__(self, predictions_path: str, output_dir: str):
        self.predictions_path = predictions_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load historical predictions
        logger.info(f"Loading predictions from {predictions_path}")
        self.predictions_df = pd.read_parquet(predictions_path)
        self.dates = sorted(self.predictions_df['date'].unique())

        # Initialize state
        self.current_positions = None
        self.position_history = []
        self.pnl_history = []
        self.vol_scale_history = []
        self.kill_switch_events = []

    def construct_s2_portfolio(self, day_preds: pd.DataFrame) -> dict:
        """
        Construct S2_FilterNegative portfolio for one day.

        Critical logic:
        - Long: Top K stocks (equal-weighted)
        - Short: Bottom K stocks FILTERED BY y_pred < 0 (equal-weighted)
        """
        # Sort by predicted return
        day_sorted = day_preds.sort_values('y_pred_reg', ascending=False)

        # Long positions: Top K
        longs = day_sorted.head(K).copy()
        long_weight = LONG_EXPOSURE / len(longs)

        # Short positions: Bottom K FILTERED by negative predictions
        # This is the CRITICAL filter that transforms -1.69 Sharpe → +0.29 Sharpe
        shorts_candidates = day_sorted.tail(K)
        shorts = shorts_candidates[shorts_candidates['y_pred_reg'] < 0].copy()

        if len(shorts) == 0:
            logger.warning("No negative predictions for shorts - skipping short positions")
            short_weight = 0
        else:
            short_weight = -SHORT_EXPOSURE / len(shorts)

        # Construct target positions
        target_positions = {}

        for ticker in longs['ticker']:
            target_positions[ticker] = long_weight

        for ticker in shorts['ticker']:
            target_positions[ticker] = short_weight

        logger.info(f"Portfolio construction: {len(longs)} longs, {len(shorts)} shorts (from {len(shorts_candidates)} candidates)")

        return target_positions

    def apply_ewma_smoothing(self, target_positions: dict) -> dict:
        """
        Apply EWMA position smoothing to reduce turnover.

        Formula: pos_t = (1 - α) × pos_{t-1} + α × target_t
        """
        if self.current_positions is None:
            # First day: use target positions directly
            smoothed = target_positions
        else:
            smoothed = {}
            all_tickers = set(target_positions.keys()) | set(self.current_positions.keys())

            for ticker in all_tickers:
                prev_pos = self.current_positions.get(ticker, 0.0)
                target_pos = target_positions.get(ticker, 0.0)
                smoothed[ticker] = (1 - ALPHA) * prev_pos + ALPHA * target_pos

        return smoothed

    def calculate_pnl(self, positions: dict, returns: pd.DataFrame) -> dict:
        """
        Calculate PnL, turnover, and costs for one day.
        """
        # Get returns for this day
        returns_dict = dict(zip(returns['ticker'], returns['y_true_reg']))

        # Calculate gross PnL (before costs)
        pnl_gross = 0.0
        long_pnl = 0.0
        short_pnl = 0.0

        for ticker, weight in positions.items():
            if ticker in returns_dict:
                ret = returns_dict[ticker]
                contribution = weight * ret
                pnl_gross += contribution

                if weight > 0:
                    long_pnl += contribution
                else:
                    short_pnl += contribution

        # Calculate turnover and costs
        if self.current_positions is None:
            # First day: full turnover
            turnover = sum(abs(w) for w in positions.values())
        else:
            traded = sum(abs(positions.get(t, 0) - self.current_positions.get(t, 0))
                        for t in set(positions.keys()) | set(self.current_positions.keys()))
            turnover = traded / (LONG_EXPOSURE + SHORT_EXPOSURE)  # Normalize by gross exposure

        # Cost calculation (CRITICAL: divide by 2 to avoid double-counting)
        cost = (turnover * (LONG_EXPOSURE + SHORT_EXPOSURE) / 2.0) * (TRANSACTION_COST_BPS / 10000)

        pnl_net = pnl_gross - cost

        return {
            'pnl_gross': pnl_gross,
            'pnl_net': pnl_net,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'cost': cost,
            'turnover': turnover
        }

    def apply_vol_targeting(self) -> float:
        """
        Calculate volatility scaling factor.

        Returns scaling factor in [0.5, 2.0]
        """
        if len(self.pnl_history) < 10:
            # Not enough history: use 1.0 (no scaling)
            return 1.0

        # Calculate rolling volatility (annualized)
        recent_pnl = [p['pnl_net'] for p in self.pnl_history[-VOL_LOOKBACK:]]
        rolling_vol = np.std(recent_pnl) * np.sqrt(252)

        if rolling_vol < 1e-6:
            return 1.0

        # Calculate scaling factor, capped at [0.5, 2.0]
        scale = TARGET_VOL / rolling_vol
        scale = np.clip(scale, 0.5, 2.0)

        logger.info(f"Vol targeting: rolling_vol={rolling_vol:.4f}, scale={scale:.4f}")

        return scale

    def check_kill_switches(self, date: str) -> dict:
        """
        Check all three kill switches.

        Returns dict with kill switch status.
        """
        ks_status = {
            'ks1_daily_loss': False,
            'ks2_dd_breach': False,
            'ks3_sharpe_negative': False,
            'any_triggered': False
        }

        if len(self.pnl_history) < 10:
            return ks_status

        recent_pnl = [p['pnl_net'] for p in self.pnl_history]

        # KS1: Daily loss > 3-sigma
        mean_pnl = np.mean(recent_pnl)
        std_pnl = np.std(recent_pnl)
        three_sigma = mean_pnl - 3 * std_pnl

        if recent_pnl[-1] < three_sigma:
            ks_status['ks1_daily_loss'] = True
            logger.warning(f"KS1 TRIGGERED: Daily loss {recent_pnl[-1]:.4f} < 3-sigma {three_sigma:.4f}")

        # KS2: Rolling 5-day drawdown > 8%
        if len(recent_pnl) >= 5:
            cum_returns = np.cumsum(recent_pnl[-5:])
            rolling_max = np.maximum.accumulate(cum_returns)
            rolling_dd = cum_returns - rolling_max

            if rolling_dd[-1] < -0.08:
                ks_status['ks2_dd_breach'] = True
                logger.warning(f"KS2 TRIGGERED: 5-day DD {rolling_dd[-1]:.4f} > 8%")

        # KS3: Rolling 60-day Sharpe < 0
        if len(recent_pnl) >= 60:
            rolling_mean = np.mean(recent_pnl[-60:])
            rolling_std = np.std(recent_pnl[-60:])

            if rolling_std > 1e-6:
                rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

                if rolling_sharpe < 0:
                    ks_status['ks3_sharpe_negative'] = True
                    logger.warning(f"KS3 TRIGGERED: 60-day Sharpe {rolling_sharpe:.4f} < 0")

        ks_status['any_triggered'] = any([ks_status['ks1_daily_loss'],
                                          ks_status['ks2_dd_breach'],
                                          ks_status['ks3_sharpe_negative']])

        if ks_status['any_triggered']:
            self.kill_switch_events.append({
                'date': date,
                'ks1': ks_status['ks1_daily_loss'],
                'ks2': ks_status['ks2_dd_breach'],
                'ks3': ks_status['ks3_sharpe_negative']
            })

        return ks_status

    def run_single_day(self, date: str) -> dict:
        """
        Run the full strategy pipeline for one trading day.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running paper trading for {date}")
        logger.info(f"{'='*60}")

        # Get predictions and actuals for this day
        day_data = self.predictions_df[self.predictions_df['date'] == date].copy()

        if len(day_data) == 0:
            logger.error(f"No data for date {date}")
            return None

        # 1. Portfolio Construction (S2 Logic)
        target_positions = self.construct_s2_portfolio(day_data)

        # 2. EWMA Position Smoothing
        smoothed_positions = self.apply_ewma_smoothing(target_positions)

        # 3. Calculate PnL (before vol targeting)
        pnl_metrics = self.calculate_pnl(smoothed_positions, day_data)
        self.pnl_history.append(pnl_metrics)

        # 4. Vol Targeting
        vol_scale = self.apply_vol_targeting()
        self.vol_scale_history.append(vol_scale)

        # 5. Check Kill Switches
        ks_status = self.check_kill_switches(date)

        # 6. Apply vol scaling to PnL (post-hoc for tracking)
        pnl_scaled = pnl_metrics['pnl_net'] * vol_scale

        # Update current positions for next day
        self.current_positions = smoothed_positions
        self.position_history.append({
            'date': date,
            'positions': smoothed_positions.copy()
        })

        # Compile daily results
        result = {
            'date': date,
            'num_longs': sum(1 for w in smoothed_positions.values() if w > 0),
            'num_shorts': sum(1 for w in smoothed_positions.values() if w < 0),
            'pnl_gross': pnl_metrics['pnl_gross'],
            'pnl_net': pnl_metrics['pnl_net'],
            'pnl_scaled': pnl_scaled,
            'long_pnl': pnl_metrics['long_pnl'],
            'short_pnl': pnl_metrics['short_pnl'],
            'cost': pnl_metrics['cost'],
            'turnover': pnl_metrics['turnover'],
            'vol_scale': vol_scale,
            'ks1': ks_status['ks1_daily_loss'],
            'ks2': ks_status['ks2_dd_breach'],
            'ks3': ks_status['ks3_sharpe_negative'],
            'ks_triggered': ks_status['any_triggered']
        }

        logger.info(f"PnL Net: {pnl_metrics['pnl_net']:.4f} | Scaled: {pnl_scaled:.4f} | "
                   f"Turnover: {pnl_metrics['turnover']:.2%} | Vol Scale: {vol_scale:.2f}")

        return result

    def run_backtest_simulation(self, start_date: str = None, end_date: str = None):
        """
        Run paper trading simulation over historical data.
        This simulates what Phase 4 will do in real-time.
        """
        dates_to_run = self.dates

        if start_date:
            dates_to_run = [d for d in dates_to_run if d >= start_date]
        if end_date:
            dates_to_run = [d for d in dates_to_run if d <= end_date]

        logger.info(f"Running paper trading simulation for {len(dates_to_run)} days")
        logger.info(f"Date range: {dates_to_run[0]} to {dates_to_run[-1]}")

        daily_results = []

        for date in dates_to_run:
            result = self.run_single_day(date)
            if result:
                daily_results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(daily_results)

        # Calculate summary metrics
        summary = self.calculate_summary_metrics(results_df)

        # Save results
        self.save_results(results_df, summary)

        return results_df, summary

    def calculate_summary_metrics(self, results_df: pd.DataFrame) -> dict:
        """
        Calculate summary performance metrics.
        """
        # Raw metrics (no vol targeting)
        pnl_net = results_df['pnl_net'].values
        sharpe_raw = (np.mean(pnl_net) / np.std(pnl_net)) * np.sqrt(252) if np.std(pnl_net) > 0 else 0
        annual_return_raw = np.mean(pnl_net) * 252
        vol_raw = np.std(pnl_net) * np.sqrt(252)

        # Scaled metrics (with vol targeting)
        pnl_scaled = results_df['pnl_scaled'].values
        sharpe_scaled = (np.mean(pnl_scaled) / np.std(pnl_scaled)) * np.sqrt(252) if np.std(pnl_scaled) > 0 else 0
        annual_return_scaled = np.mean(pnl_scaled) * 252
        vol_scaled = np.std(pnl_scaled) * np.sqrt(252)

        # Drawdown
        cum_returns = np.cumsum(pnl_net)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - running_max
        max_dd = np.min(drawdown)

        # Long/Short attribution
        long_sharpe = (np.mean(results_df['long_pnl']) / np.std(results_df['long_pnl'])) * np.sqrt(252) if np.std(results_df['long_pnl']) > 0 else 0
        short_sharpe = (np.mean(results_df['short_pnl']) / np.std(results_df['short_pnl'])) * np.sqrt(252) if np.std(results_df['short_pnl']) > 0 else 0

        # Kill switches
        ks_events = results_df[results_df['ks_triggered']].shape[0]
        ks_pct = (ks_events / len(results_df)) * 100

        summary = {
            'num_days': len(results_df),
            'sharpe_raw': sharpe_raw,
            'annual_return_raw': annual_return_raw,
            'vol_raw': vol_raw,
            'sharpe_scaled': sharpe_scaled,
            'annual_return_scaled': annual_return_scaled,
            'vol_scaled': vol_scaled,
            'max_dd': max_dd,
            'avg_turnover': results_df['turnover'].mean(),
            'avg_cost_bps': results_df['cost'].mean() * 10000,
            'long_sharpe': long_sharpe,
            'short_sharpe': short_sharpe,
            'ks_events': int(ks_events),
            'ks_pct': ks_pct
        }

        return summary

    def save_results(self, results_df: pd.DataFrame, summary: dict):
        """
        Save paper trading results to disk.
        """
        # Save daily results
        results_file = self.output_dir / "phase4_paper_trading_daily.parquet"
        results_df.to_parquet(results_file)
        logger.info(f"Saved daily results to {results_file}")

        # Save summary metrics
        summary_file = self.output_dir / "phase4_paper_trading_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PAPER TRADING SUMMARY")
        logger.info("="*60)
        logger.info(f"Trading Days: {summary['num_days']}")
        logger.info(f"\nRAW PERFORMANCE (no vol targeting):")
        logger.info(f"  Sharpe Ratio: {summary['sharpe_raw']:.2f}")
        logger.info(f"  Annual Return: {summary['annual_return_raw']:.2%}")
        logger.info(f"  Volatility: {summary['vol_raw']:.2%}")
        logger.info(f"\nVOL-TARGETED PERFORMANCE:")
        logger.info(f"  Sharpe Ratio: {summary['sharpe_scaled']:.2f}")
        logger.info(f"  Annual Return: {summary['annual_return_scaled']:.2%}")
        logger.info(f"  Volatility: {summary['vol_scaled']:.2%}")
        logger.info(f"\nRISK METRICS:")
        logger.info(f"  Max Drawdown: {summary['max_dd']:.2%}")
        logger.info(f"  Long Sharpe: {summary['long_sharpe']:.2f}")
        logger.info(f"  Short Sharpe: {summary['short_sharpe']:.2f}")
        logger.info(f"\nEXECUTION:")
        logger.info(f"  Avg Turnover: {summary['avg_turnover']:.2%}")
        logger.info(f"  Avg Cost: {summary['avg_cost_bps']:.1f} bps")
        logger.info(f"\nKILL SWITCHES:")
        logger.info(f"  Events: {summary['ks_events']} ({summary['ks_pct']:.1f}% of days)")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Paper Trading Runner')
    parser.add_argument('--predictions', type=str,
                       default='data/processed/phase1_predictions.parquet',
                       help='Path to predictions parquet file')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed/phase4',
                       help='Output directory for results')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for simulation (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for simulation (YYYY-MM-DD)')

    args = parser.parse_args()

    # Run paper trading simulation
    runner = PaperTradingRunner(
        predictions_path=args.predictions,
        output_dir=args.output_dir
    )

    results_df, summary = runner.run_backtest_simulation(
        start_date=args.start_date,
        end_date=args.end_date
    )

    logger.info("\n✅ Paper trading simulation complete!")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
