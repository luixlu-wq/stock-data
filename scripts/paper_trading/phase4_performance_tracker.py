"""
Phase 4: Performance Tracker

Compares paper trading performance against Phase 3.5 backtest expectations.
Tracks key metrics and generates alerts when performance deviates significantly.

Success Criteria (from PHASE3_EXECUTION_GUIDE.md):
- Paper trading Sharpe > 1.0
- Costs within 5-7 bps
- No systematic execution issues
- Kill switches working as designed

Usage:
    python scripts/paper_trading/phase4_performance_tracker.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected performance from Phase 3.5 (STRATEGY_DEFINITION.md v2.0.0)
EXPECTED_METRICS = {
    'sharpe_scaled': 1.29,
    'annual_return_scaled': 0.1195,
    'vol_scaled': 0.0927,
    'max_dd': -0.0521,
    'long_sharpe': 0.92,
    'short_sharpe': 0.29,
    'avg_turnover': 0.22,  # Estimated from Phase 2B
    'ks_pct': 8.0
}

# Alert thresholds
ALERT_THRESHOLDS = {
    'sharpe_min': 1.0,  # RED: Sharpe < 1.0
    'sharpe_warning': 1.15,  # YELLOW: Sharpe < 1.15
    'dd_max': -0.10,  # RED: MaxDD < -10%
    'dd_warning': -0.08,  # YELLOW: MaxDD < -8%
    'cost_max_bps': 7.0,  # RED: Avg cost > 7 bps
    'cost_warning_bps': 6.0,  # YELLOW: Avg cost > 6 bps
    'ks_pct_max': 15.0  # RED: Kill switches > 15% of days
}


class PerformanceTracker:
    """
    Tracks paper trading performance and compares against backtest expectations.
    """

    def __init__(self, paper_trading_dir: str, backtest_results_path: str = None):
        self.paper_trading_dir = Path(paper_trading_dir)
        self.backtest_results_path = backtest_results_path

        # Load paper trading results
        self.paper_daily = self.load_paper_trading_results()
        self.paper_summary = self.load_paper_trading_summary()

        # Load backtest results if available
        self.backtest_summary = self.load_backtest_results() if backtest_results_path else EXPECTED_METRICS

    def load_paper_trading_results(self) -> pd.DataFrame:
        """Load daily paper trading results."""
        daily_file = self.paper_trading_dir / "phase4_paper_trading_daily.parquet"

        if not daily_file.exists():
            raise FileNotFoundError(f"Paper trading results not found: {daily_file}")

        df = pd.read_parquet(daily_file)
        logger.info(f"Loaded {len(df)} days of paper trading results")
        return df

    def load_paper_trading_summary(self) -> dict:
        """Load paper trading summary metrics."""
        summary_file = self.paper_trading_dir / "phase4_paper_trading_summary.json"

        if not summary_file.exists():
            raise FileNotFoundError(f"Paper trading summary not found: {summary_file}")

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        logger.info("Loaded paper trading summary")
        return summary

    def load_backtest_results(self) -> dict:
        """Load Phase 3.5 backtest results."""
        with open(self.backtest_results_path, 'r') as f:
            backtest = json.load(f)

        logger.info(f"Loaded backtest results from {self.backtest_results_path}")
        return backtest

    def compare_metrics(self) -> pd.DataFrame:
        """
        Compare paper trading vs backtest metrics.
        """
        comparison = []

        metrics_to_compare = [
            ('sharpe_scaled', 'Sharpe (Vol-Targeted)', '{:.2f}'),
            ('annual_return_scaled', 'Annual Return', '{:.2%}'),
            ('vol_scaled', 'Volatility', '{:.2%}'),
            ('max_dd', 'Max Drawdown', '{:.2%}'),
            ('long_sharpe', 'Long Sharpe', '{:.2f}'),
            ('short_sharpe', 'Short Sharpe', '{:.2f}'),
            ('avg_turnover', 'Avg Turnover', '{:.2%}'),
            ('ks_pct', 'Kill Switch %', '{:.1f}%')
        ]

        for metric_key, metric_name, fmt in metrics_to_compare:
            backtest_val = self.backtest_summary.get(metric_key, np.nan)
            paper_val = self.paper_summary.get(metric_key, np.nan)

            if not np.isnan(backtest_val) and not np.isnan(paper_val):
                diff = paper_val - backtest_val
                diff_pct = (diff / abs(backtest_val)) * 100 if backtest_val != 0 else np.nan
            else:
                diff = np.nan
                diff_pct = np.nan

            comparison.append({
                'Metric': metric_name,
                'Backtest': backtest_val,
                'Paper': paper_val,
                'Diff': diff,
                'Diff %': diff_pct
            })

        df = pd.DataFrame(comparison)
        return df

    def generate_alerts(self) -> List[Dict]:
        """
        Generate alerts for metrics outside acceptable ranges.
        """
        alerts = []

        # Sharpe ratio check
        sharpe = self.paper_summary['sharpe_scaled']
        if sharpe < ALERT_THRESHOLDS['sharpe_min']:
            alerts.append({
                'level': 'RED',
                'metric': 'Sharpe Ratio',
                'value': sharpe,
                'threshold': ALERT_THRESHOLDS['sharpe_min'],
                'message': f"Sharpe {sharpe:.2f} < {ALERT_THRESHOLDS['sharpe_min']} - BELOW DEPLOYMENT GATE"
            })
        elif sharpe < ALERT_THRESHOLDS['sharpe_warning']:
            alerts.append({
                'level': 'YELLOW',
                'metric': 'Sharpe Ratio',
                'value': sharpe,
                'threshold': ALERT_THRESHOLDS['sharpe_warning'],
                'message': f"Sharpe {sharpe:.2f} < {ALERT_THRESHOLDS['sharpe_warning']} - Below expected"
            })

        # Drawdown check
        max_dd = self.paper_summary['max_dd']
        if max_dd < ALERT_THRESHOLDS['dd_max']:
            alerts.append({
                'level': 'RED',
                'metric': 'Max Drawdown',
                'value': max_dd,
                'threshold': ALERT_THRESHOLDS['dd_max'],
                'message': f"MaxDD {max_dd:.2%} > {-ALERT_THRESHOLDS['dd_max']:.2%} - EXCESSIVE DRAWDOWN"
            })
        elif max_dd < ALERT_THRESHOLDS['dd_warning']:
            alerts.append({
                'level': 'YELLOW',
                'metric': 'Max Drawdown',
                'value': max_dd,
                'threshold': ALERT_THRESHOLDS['dd_warning'],
                'message': f"MaxDD {max_dd:.2%} > {-ALERT_THRESHOLDS['dd_warning']:.2%} - Higher than expected"
            })

        # Cost check
        avg_cost_bps = self.paper_summary['avg_cost_bps']
        if avg_cost_bps > ALERT_THRESHOLDS['cost_max_bps']:
            alerts.append({
                'level': 'RED',
                'metric': 'Avg Cost',
                'value': avg_cost_bps,
                'threshold': ALERT_THRESHOLDS['cost_max_bps'],
                'message': f"Avg cost {avg_cost_bps:.1f} bps > {ALERT_THRESHOLDS['cost_max_bps']} bps - COSTS TOO HIGH"
            })
        elif avg_cost_bps > ALERT_THRESHOLDS['cost_warning_bps']:
            alerts.append({
                'level': 'YELLOW',
                'metric': 'Avg Cost',
                'value': avg_cost_bps,
                'threshold': ALERT_THRESHOLDS['cost_warning_bps'],
                'message': f"Avg cost {avg_cost_bps:.1f} bps > {ALERT_THRESHOLDS['cost_warning_bps']} bps - Higher than expected"
            })

        # Kill switch check
        ks_pct = self.paper_summary['ks_pct']
        if ks_pct > ALERT_THRESHOLDS['ks_pct_max']:
            alerts.append({
                'level': 'RED',
                'metric': 'Kill Switch %',
                'value': ks_pct,
                'threshold': ALERT_THRESHOLDS['ks_pct_max'],
                'message': f"Kill switches {ks_pct:.1f}% > {ALERT_THRESHOLDS['ks_pct_max']}% - TOO FREQUENT"
            })

        return alerts

    def plot_performance(self, output_dir: Path):
        """
        Generate performance visualization plots.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate cumulative returns
        cum_returns_raw = np.cumsum(self.paper_daily['pnl_net'].values)
        cum_returns_scaled = np.cumsum(self.paper_daily['pnl_scaled'].values)

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Phase 4: Paper Trading Performance', fontsize=16, fontweight='bold')

        # 1. Cumulative Returns
        ax = axes[0, 0]
        ax.plot(cum_returns_raw, label='Raw', linewidth=2)
        ax.plot(cum_returns_scaled, label='Vol-Targeted', linewidth=2, linestyle='--')
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Cumulative PnL')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Daily PnL
        ax = axes[0, 1]
        ax.bar(range(len(self.paper_daily)), self.paper_daily['pnl_net'].values, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Daily PnL (Net)')
        ax.set_ylabel('Daily PnL')
        ax.grid(True, alpha=0.3)

        # 3. Rolling Sharpe (60-day)
        ax = axes[1, 0]
        if len(self.paper_daily) >= 60:
            rolling_sharpe = []
            for i in range(60, len(self.paper_daily) + 1):
                window_pnl = self.paper_daily['pnl_net'].values[i-60:i]
                sharpe = (np.mean(window_pnl) / np.std(window_pnl)) * np.sqrt(252) if np.std(window_pnl) > 0 else 0
                rolling_sharpe.append(sharpe)

            ax.plot(range(60, len(self.paper_daily) + 1), rolling_sharpe, linewidth=2)
            ax.axhline(y=1.0, color='green', linestyle='--', label='Target (1.0)')
            ax.axhline(y=1.29, color='blue', linestyle='--', label='Expected (1.29)')
            ax.set_title('Rolling 60-Day Sharpe')
            ax.set_ylabel('Sharpe Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Turnover
        ax = axes[1, 1]
        ax.plot(self.paper_daily['turnover'].values, linewidth=2)
        ax.axhline(y=0.22, color='red', linestyle='--', label='Expected (22%)')
        ax.set_title('Daily Turnover')
        ax.set_ylabel('Turnover')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Long vs Short PnL
        ax = axes[2, 0]
        cum_long = np.cumsum(self.paper_daily['long_pnl'].values)
        cum_short = np.cumsum(self.paper_daily['short_pnl'].values)
        ax.plot(cum_long, label='Long', linewidth=2)
        ax.plot(cum_short, label='Short', linewidth=2)
        ax.set_title('Cumulative Long/Short PnL')
        ax.set_ylabel('Cumulative PnL')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Vol Targeting Scale
        ax = axes[2, 1]
        ax.plot(self.paper_daily['vol_scale'].values, linewidth=2)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Volatility Scaling Factor')
        ax.set_ylabel('Scale Factor')
        ax.set_ylim([0, 2.5])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        plot_file = output_dir / "phase4_performance_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance plots to {plot_file}")
        plt.close()

    def generate_report(self) -> str:
        """
        Generate comprehensive performance report.
        """
        report = []
        report.append("="*80)
        report.append("PHASE 4: PAPER TRADING PERFORMANCE REPORT")
        report.append("="*80)
        report.append("")

        # Summary
        report.append(f"Trading Days: {self.paper_summary['num_days']}")
        report.append(f"Date Range: {self.paper_daily['date'].min()} to {self.paper_daily['date'].max()}")
        report.append("")

        # Comparison table
        report.append("PERFORMANCE COMPARISON (Paper vs Backtest)")
        report.append("-"*80)

        comparison_df = self.compare_metrics()
        for _, row in comparison_df.iterrows():
            backtest_str = f"{row['Backtest']:.4f}" if not pd.isna(row['Backtest']) else "N/A"
            paper_str = f"{row['Paper']:.4f}" if not pd.isna(row['Paper']) else "N/A"
            diff_str = f"{row['Diff %']:+.1f}%" if not pd.isna(row['Diff %']) else "N/A"

            report.append(f"{row['Metric']:<25} | Backtest: {backtest_str:>8} | Paper: {paper_str:>8} | Diff: {diff_str:>8}")

        report.append("")

        # Alerts
        alerts = self.generate_alerts()
        if alerts:
            report.append("ALERTS")
            report.append("-"*80)
            for alert in alerts:
                emoji = "🚨" if alert['level'] == 'RED' else "⚠️"
                report.append(f"{emoji} [{alert['level']}] {alert['message']}")
            report.append("")
        else:
            report.append("✅ No alerts - all metrics within acceptable ranges")
            report.append("")

        # Deployment Decision
        report.append("DEPLOYMENT DECISION")
        report.append("-"*80)

        sharpe = self.paper_summary['sharpe_scaled']
        max_dd = self.paper_summary['max_dd']
        ks_pct = self.paper_summary['ks_pct']

        red_alerts = [a for a in alerts if a['level'] == 'RED']

        if red_alerts:
            report.append("🚨 DO NOT PROCEED TO LIVE TRADING")
            report.append("   Critical metrics outside acceptable ranges:")
            for alert in red_alerts:
                report.append(f"   - {alert['message']}")
        elif sharpe >= 1.0 and max_dd > -0.10 and ks_pct < 15.0:
            report.append("✅ GREEN LIGHT FOR LIVE DEPLOYMENT")
            report.append("   All success criteria met:")
            report.append(f"   - Sharpe {sharpe:.2f} > 1.0 ✅")
            report.append(f"   - MaxDD {max_dd:.2%} > -10% ✅")
            report.append(f"   - Kill switches {ks_pct:.1f}% < 15% ✅")
        else:
            report.append("⚠️  YELLOW FLAG - Review before proceeding")
            report.append("   Some metrics below expectations but within tolerance")

        report.append("")
        report.append("="*80)

        return "\n".join(report)

    def save_report(self, output_dir: Path):
        """
        Save performance report to file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        report_text = self.generate_report()

        # Save to file
        report_file = output_dir / "phase4_performance_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Saved performance report to {report_file}")

        # Print to console
        print("\n" + report_text)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Phase 4 Performance Tracker')
    parser.add_argument('--paper-dir', type=str,
                       default='data/processed/phase4',
                       help='Directory with paper trading results')
    parser.add_argument('--backtest', type=str,
                       default='data/processed/phase3_5_risk_management_results.json',
                       help='Path to Phase 3.5 backtest results')
    parser.add_argument('--output-dir', type=str,
                       default='reports/phase4',
                       help='Output directory for reports and plots')

    args = parser.parse_args()

    # Create tracker
    tracker = PerformanceTracker(
        paper_trading_dir=args.paper_dir,
        backtest_results_path=args.backtest
    )

    # Generate plots
    tracker.plot_performance(Path(args.output_dir))

    # Generate and save report
    tracker.save_report(Path(args.output_dir))

    logger.info("\n✅ Performance tracking complete!")


if __name__ == "__main__":
    main()
