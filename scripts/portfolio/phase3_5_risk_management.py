"""
PHASE 3.5: RISK MANAGEMENT & VOL TARGETING

Apply institutional-grade risk controls to S2_FilterNegative:
1. Volatility targeting (8% annual)
2. Kill switches (3-sigma loss, DD threshold, Sharpe collapse)
3. Production-ready metrics

Input: S2 daily PnL history from Phase 3.4
Output: Final production metrics with risk controls
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

TRADING_DAYS = 252


class Phase35RiskManagement:
    """Apply vol targeting and kill switches to S2_FilterNegative."""

    def __init__(
        self,
        pnl_history: pd.DataFrame,
        target_vol: float = 0.08,
        vol_lookback: int = 20,
        logger=None
    ):
        """
        Initialize risk management system.

        Args:
            pnl_history: DataFrame with daily PnL history
            target_vol: Target annual volatility (default 8%)
            vol_lookback: Rolling window for vol calculation (default 20 days)
            logger: Logger instance
        """
        self.df = pnl_history.copy()
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.logger = logger or setup_logger(__name__)

    def apply_vol_targeting(self) -> pd.DataFrame:
        """
        Apply volatility targeting to PnL series.

        Returns:
            DataFrame with vol-targeted PnL and scaling factors
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("VOLATILITY TARGETING")
        self.logger.info("="*70)
        self.logger.info("")
        self.logger.info(f"Target Volatility: {self.target_vol*100:.1f}% annual")
        self.logger.info(f"Lookback Window: {self.vol_lookback} days")
        self.logger.info("")

        # Calculate rolling volatility (annualized)
        self.df['rolling_vol'] = self.df['pnl'].rolling(
            window=self.vol_lookback,
            min_periods=10
        ).std() * np.sqrt(TRADING_DAYS)

        # Fill initial NaN with overall vol
        overall_vol = self.df['pnl'].std() * np.sqrt(TRADING_DAYS)
        self.df['rolling_vol'] = self.df['rolling_vol'].fillna(overall_vol)

        # Calculate scaling factor
        # scale = target_vol / realized_vol
        # Capped at [0.5, 2.0] to prevent extreme leverage/deleverage
        self.df['vol_scale'] = (self.target_vol / self.df['rolling_vol']).clip(0.5, 2.0)

        # Apply scaling to PnL
        self.df['pnl_scaled'] = self.df['pnl'] * self.df['vol_scale']

        # Also scale long/short attribution
        self.df['long_pnl_scaled'] = self.df['long_pnl'] * self.df['vol_scale']
        self.df['short_pnl_scaled'] = self.df['short_pnl'] * self.df['vol_scale']

        self.logger.info("Vol Targeting Applied:")
        self.logger.info(f"  Raw Vol: {overall_vol*100:.2f}%")
        self.logger.info(f"  Target Vol: {self.target_vol*100:.2f}%")
        self.logger.info(f"  Avg Scale Factor: {self.df['vol_scale'].mean():.2f}x")
        self.logger.info(f"  Scale Range: [{self.df['vol_scale'].min():.2f}x, {self.df['vol_scale'].max():.2f}x]")
        self.logger.info("")

        return self.df

    def apply_kill_switches(self) -> pd.DataFrame:
        """
        Apply 3 kill switches to identify risk events.

        Kill Switch 1: Daily loss > 3-sigma
        Kill Switch 2: Rolling 5-day DD > 8%
        Kill Switch 3: Rolling 60-day Sharpe < 0

        Returns:
            DataFrame with kill switch flags
        """
        self.logger.info("="*70)
        self.logger.info("KILL SWITCHES")
        self.logger.info("="*70)
        self.logger.info("")

        # Use scaled PnL for kill switches
        pnl = self.df['pnl_scaled']

        # Kill Switch 1: Daily loss > 3-sigma
        mean_pnl = pnl.mean()
        std_pnl = pnl.std()
        three_sigma = mean_pnl - 3 * std_pnl

        self.df['ks1_daily_loss'] = pnl < three_sigma

        ks1_events = self.df['ks1_daily_loss'].sum()
        self.logger.info(f"KS1: Daily Loss > 3-Sigma")
        self.logger.info(f"  Threshold: {three_sigma*100:.3f}%")
        self.logger.info(f"  Events: {ks1_events}")
        if ks1_events > 0:
            worst_day = pnl.min()
            self.logger.info(f"  Worst Day: {worst_day*100:.3f}%")
        self.logger.info("")

        # Kill Switch 2: Rolling 5-day drawdown > 8%
        cum_returns = pnl.cumsum()
        rolling_max = cum_returns.rolling(window=5, min_periods=1).max()
        rolling_dd = cum_returns - rolling_max

        self.df['ks2_dd_breach'] = rolling_dd < -0.08

        ks2_events = self.df['ks2_dd_breach'].sum()
        self.logger.info(f"KS2: Rolling 5-Day DD > 8%")
        self.logger.info(f"  Threshold: -8.00%")
        self.logger.info(f"  Events: {ks2_events}")
        if ks2_events > 0:
            worst_dd = rolling_dd.min()
            self.logger.info(f"  Worst 5-Day DD: {worst_dd*100:.2f}%")
        self.logger.info("")

        # Kill Switch 3: Rolling 60-day Sharpe < 0
        rolling_mean = pnl.rolling(window=60, min_periods=30).mean()
        rolling_std = pnl.rolling(window=60, min_periods=30).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS)

        self.df['ks3_sharpe_negative'] = rolling_sharpe < 0

        ks3_events = self.df['ks3_sharpe_negative'].sum()
        self.logger.info(f"KS3: Rolling 60-Day Sharpe < 0")
        self.logger.info(f"  Threshold: 0.00")
        self.logger.info(f"  Events: {ks3_events}")
        if ks3_events > 0:
            min_sharpe = rolling_sharpe.min()
            self.logger.info(f"  Worst 60-Day Sharpe: {min_sharpe:.2f}")
        self.logger.info("")

        # Any kill switch triggered
        self.df['any_kill_switch'] = (
            self.df['ks1_daily_loss'] |
            self.df['ks2_dd_breach'] |
            self.df['ks3_sharpe_negative']
        )

        total_ks_events = self.df['any_kill_switch'].sum()
        self.logger.info(f"TOTAL KILL SWITCH EVENTS: {total_ks_events}")
        self.logger.info(f"  ({total_ks_events / len(self.df) * 100:.1f}% of trading days)")
        self.logger.info("")

        return self.df

    def calculate_final_metrics(self) -> dict:
        """
        Calculate final production metrics.

        Returns:
            Dict with comprehensive performance metrics
        """
        self.logger.info("="*70)
        self.logger.info("FINAL PRODUCTION METRICS")
        self.logger.info("="*70)
        self.logger.info("")

        # Raw metrics (before vol targeting)
        pnl_raw = self.df['pnl']
        sharpe_raw = pnl_raw.mean() / pnl_raw.std() * np.sqrt(TRADING_DAYS) if pnl_raw.std() > 0 else 0
        vol_raw = pnl_raw.std() * np.sqrt(TRADING_DAYS)

        cum_raw = pnl_raw.cumsum()
        running_max_raw = cum_raw.expanding().max()
        dd_raw = cum_raw - running_max_raw
        max_dd_raw = dd_raw.min()

        # Scaled metrics (after vol targeting)
        pnl_scaled = self.df['pnl_scaled']
        sharpe_scaled = pnl_scaled.mean() / pnl_scaled.std() * np.sqrt(TRADING_DAYS) if pnl_scaled.std() > 0 else 0
        vol_scaled = pnl_scaled.std() * np.sqrt(TRADING_DAYS)

        cum_scaled = pnl_scaled.cumsum()
        running_max_scaled = cum_scaled.expanding().max()
        dd_scaled = cum_scaled - running_max_scaled
        max_dd_scaled = dd_scaled.min()

        # Attribution
        long_sharpe = (self.df['long_pnl_scaled'].mean() / self.df['long_pnl_scaled'].std() * np.sqrt(TRADING_DAYS)
                      if self.df['long_pnl_scaled'].std() > 0 else 0)
        short_sharpe = (self.df['short_pnl_scaled'].mean() / self.df['short_pnl_scaled'].std() * np.sqrt(TRADING_DAYS)
                       if self.df['short_pnl_scaled'].std() > 0 else 0)

        metrics = {
            # Raw (before vol targeting)
            'raw_sharpe': float(sharpe_raw),
            'raw_annual_return': float(pnl_raw.mean() * TRADING_DAYS),
            'raw_vol': float(vol_raw),
            'raw_max_dd': float(max_dd_raw),
            'raw_avg_turnover': float(self.df['turnover'].mean()),

            # Scaled (after vol targeting)
            'scaled_sharpe': float(sharpe_scaled),
            'scaled_annual_return': float(pnl_scaled.mean() * TRADING_DAYS),
            'scaled_vol': float(vol_scaled),
            'scaled_max_dd': float(max_dd_scaled),
            'scaled_avg_turnover': float(self.df['turnover'].mean()),  # Turnover unchanged

            # Attribution
            'long_sharpe': float(long_sharpe),
            'short_sharpe': float(short_sharpe),

            # Vol targeting
            'target_vol': self.target_vol,
            'vol_lookback': self.vol_lookback,
            'avg_scale_factor': float(self.df['vol_scale'].mean()),

            # Kill switches
            'ks1_events': int(self.df['ks1_daily_loss'].sum()),
            'ks2_events': int(self.df['ks2_dd_breach'].sum()),
            'ks3_events': int(self.df['ks3_sharpe_negative'].sum()),
            'total_ks_events': int(self.df['any_kill_switch'].sum()),
            'ks_event_pct': float(self.df['any_kill_switch'].sum() / len(self.df) * 100)
        }

        # Display
        self.logger.info("RAW METRICS (No Vol Targeting):")
        self.logger.info(f"  Sharpe: {metrics['raw_sharpe']:.2f}")
        self.logger.info(f"  Annual Return: {metrics['raw_annual_return']*100:.2f}%")
        self.logger.info(f"  Volatility: {metrics['raw_vol']*100:.2f}%")
        self.logger.info(f"  Max DD: {metrics['raw_max_dd']*100:.2f}%")
        self.logger.info(f"  Avg Turnover: {metrics['raw_avg_turnover']*100:.1f}%")
        self.logger.info("")

        self.logger.info("SCALED METRICS (Vol-Targeted @ 8%):")
        self.logger.info(f"  Sharpe: {metrics['scaled_sharpe']:.2f}")
        self.logger.info(f"  Annual Return: {metrics['scaled_annual_return']*100:.2f}%")
        self.logger.info(f"  Volatility: {metrics['scaled_vol']*100:.2f}%")
        self.logger.info(f"  Max DD: {metrics['scaled_max_dd']*100:.2f}%")
        self.logger.info(f"  Avg Scale Factor: {metrics['avg_scale_factor']:.2f}x")
        self.logger.info("")

        self.logger.info("ATTRIBUTION:")
        self.logger.info(f"  Long Sharpe: {metrics['long_sharpe']:.2f}")
        self.logger.info(f"  Short Sharpe: {metrics['short_sharpe']:.2f}")
        self.logger.info("")

        self.logger.info("KILL SWITCHES:")
        self.logger.info(f"  KS1 (3-Sigma): {metrics['ks1_events']} events")
        self.logger.info(f"  KS2 (DD 8%): {metrics['ks2_events']} events")
        self.logger.info(f"  KS3 (Sharpe<0): {metrics['ks3_events']} events")
        self.logger.info(f"  Total: {metrics['total_ks_events']} events ({metrics['ks_event_pct']:.1f}% of days)")
        self.logger.info("")

        return metrics

    def run(self) -> dict:
        """
        Run complete risk management pipeline.

        Returns:
            Final production metrics
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("PHASE 3.5: RISK MANAGEMENT & VOL TARGETING")
        self.logger.info("="*70)
        self.logger.info("")
        self.logger.info("Applying institutional-grade risk controls to S2_FilterNegative")
        self.logger.info("")

        # Step 1: Vol targeting
        self.apply_vol_targeting()

        # Step 2: Kill switches
        self.apply_kill_switches()

        # Step 3: Final metrics
        metrics = self.calculate_final_metrics()

        return metrics


def main():
    """Run Phase 3.5 risk management."""
    logger = setup_logger('phase3_5_risk', log_file='logs/phase3_5_risk_management.log', level='INFO')

    # Load S2 daily PnL history
    history_file = Path("data/processed/s2_daily_pnl_history.parquet")
    if not history_file.exists():
        logger.error(f"S2 history not found: {history_file}")
        logger.error("Run Phase 3.4 first to generate S2 daily PnL history")
        return

    logger.info(f"Loading S2 daily PnL history from {history_file}...")
    pnl_history = pd.read_parquet(history_file)
    logger.info(f"Loaded {len(pnl_history)} trading days")
    logger.info("")

    # Run risk management
    risk_mgmt = Phase35RiskManagement(
        pnl_history=pnl_history,
        target_vol=0.08,  # 8% annual
        vol_lookback=20,
        logger=logger
    )

    metrics = risk_mgmt.run()

    # Save results
    output_file = "data/processed/phase3_5_risk_management_results.json"

    import json
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Decision
    logger.info("\n" + "="*70)
    logger.info("DEPLOYMENT DECISION")
    logger.info("="*70)
    logger.info("")

    if metrics['scaled_sharpe'] >= 1.2:
        logger.info("STATUS: GREEN LIGHT")
        logger.info(f"  Vol-Targeted Sharpe: {metrics['scaled_sharpe']:.2f}")
        logger.info(f"  Target Vol Achieved: {metrics['scaled_vol']*100:.1f}%")
        logger.info(f"  Kill Switches: {metrics['ks_event_pct']:.1f}% of days")
        logger.info("")
        logger.info("RECOMMENDATION: Deploy S2_FilterNegative with 8% vol targeting")
        logger.info("  -> Proceed to paper trading (60+ days)")
        logger.info("  -> Start with 10% capital allocation")
    elif metrics['scaled_sharpe'] >= 0.8:
        logger.info("STATUS: YELLOW LIGHT")
        logger.info(f"  Vol-Targeted Sharpe: {metrics['scaled_sharpe']:.2f}")
        logger.info("  (Below ideal 1.2+ threshold)")
        logger.info("")
        logger.info("RECOMMENDATION: Cautious deployment")
        logger.info("  -> Extended paper trading (90+ days)")
        logger.info("  -> Start with 5% capital allocation")
    else:
        logger.info("STATUS: RED LIGHT")
        logger.info(f"  Vol-Targeted Sharpe: {metrics['scaled_sharpe']:.2f}")
        logger.info("  (Below minimum 0.8 threshold)")
        logger.info("")
        logger.info("RECOMMENDATION: Do not deploy")
        logger.info("  -> Investigate Sharpe collapse")
        logger.info("  -> Consider long-only alternative (P1)")

    logger.info("")
    logger.info("="*70)
    logger.info("PHASE 3.5 COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
