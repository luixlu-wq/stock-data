"""
PHASE 3.4: Capital Scaling & Kill Switches

This script implements:
1. Volatility targeting (8% annual target)
2. Kill switches:
   - Daily loss > 3-sigma -> flatten
   - Weekly drawdown > 8% -> halt
   - 60-day Net Sharpe < 0 -> disable
3. Position sizing rules
4. Real-time risk monitoring

This is what separates quants from gamblers.
"""
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger


class CapitalScaling:
    """Implement capital scaling with volatility targeting."""

    def __init__(
        self,
        target_vol: float = 0.08,  # 8% annual volatility
        vol_lookback: int = 60,     # Days to calculate volatility
        logger=None
    ):
        """
        Initialize capital scaling.

        Args:
            target_vol: Target annual volatility (e.g., 0.08 = 8%)
            vol_lookback: Days to look back for volatility calculation
            logger: Logger instance
        """
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.logger = logger or setup_logger(__name__)

    def calculate_position_scalar(self, returns_history: pd.Series) -> float:
        """
        Calculate position size scalar based on realized volatility.

        Returns:
            scalar: Position size multiplier (e.g., 1.5 means scale up 1.5x)
        """
        if len(returns_history) < 20:
            # Not enough data - use conservative 1.0x
            return 1.0

        # Calculate realized volatility (annualized)
        realized_vol = returns_history.std() * np.sqrt(252)

        if realized_vol < 1e-6:
            return 1.0

        # Scale positions to hit target volatility
        scalar = self.target_vol / realized_vol

        # Cap scalar at reasonable bounds (0.5x to 3.0x)
        scalar = np.clip(scalar, 0.5, 3.0)

        return float(scalar)

    def apply_volatility_targeting(
        self,
        pnl_df: pd.DataFrame,
        positions_history: List[Dict]
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Apply volatility targeting to historical PnL and positions.

        Args:
            pnl_df: DataFrame with daily PnL
            positions_history: List of {date, positions} dicts

        Returns:
            scaled_pnl_df: PnL with volatility targeting applied
            scaled_positions: Positions with scalars applied
        """
        scaled_pnl = []
        scaled_positions = []
        scalars = []

        for i, row in pnl_df.iterrows():
            # Get returns history up to this point
            if i >= self.vol_lookback:
                returns_history = pnl_df.iloc[i-self.vol_lookback:i]['pnl_net']
            else:
                returns_history = pnl_df.iloc[:i]['pnl_net']

            # Calculate scalar
            scalar = self.calculate_position_scalar(returns_history)
            scalars.append(scalar)

            # Scale PnL
            scaled_pnl.append({
                'date': row['date'],
                'pnl_gross': row['pnl_gross'] * scalar,
                'pnl_net': row['pnl_net'] * scalar,
                'pnl_long': row['pnl_long'] * scalar,
                'pnl_short': row['pnl_short'] * scalar,
                'turnover': row['turnover'],  # Turnover unchanged
                'cost': row['cost'] * scalar,  # Costs scale with position size
                'vol_scalar': scalar,
                'realized_vol': returns_history.std() * np.sqrt(252) if len(returns_history) > 0 else 0
            })

            # Scale positions
            if i < len(positions_history):
                pos_dict = positions_history[i]['positions']
                scaled_pos = {ticker: pos * scalar for ticker, pos in pos_dict.items()}
                scaled_positions.append({
                    'date': row['date'],
                    'positions': scaled_pos,
                    'vol_scalar': scalar
                })

        scaled_pnl_df = pd.DataFrame(scaled_pnl)
        return scaled_pnl_df, scaled_positions


class KillSwitches:
    """Implement mandatory kill switches for risk management."""

    def __init__(
        self,
        daily_loss_sigma: float = 3.0,      # Daily loss threshold (sigma)
        weekly_dd_threshold: float = 0.08,   # 8% weekly drawdown
        sharpe_window: int = 60,             # Days for Sharpe calculation
        sharpe_threshold: float = 0.0,       # Disable if Sharpe < 0
        logger=None
    ):
        """
        Initialize kill switches.

        Args:
            daily_loss_sigma: Number of standard deviations for daily loss kill
            weekly_dd_threshold: Maximum weekly drawdown before halting
            sharpe_window: Rolling window for Sharpe calculation
            sharpe_threshold: Minimum Sharpe to keep running
            logger: Logger instance
        """
        self.daily_loss_sigma = daily_loss_sigma
        self.weekly_dd_threshold = weekly_dd_threshold
        self.sharpe_window = sharpe_window
        self.sharpe_threshold = sharpe_threshold
        self.logger = logger or setup_logger(__name__)

        # Kill switch state
        self.is_halted = False
        self.halt_reason = None
        self.halt_date = None

    def check_daily_loss(self, pnl: float, historical_std: float) -> Tuple[bool, str]:
        """
        Check if daily loss exceeds threshold.

        Args:
            pnl: Today's PnL
            historical_std: Historical daily std dev

        Returns:
            (should_halt, reason)
        """
        threshold = -self.daily_loss_sigma * historical_std

        if pnl < threshold:
            reason = f"Daily loss {pnl:.4f} exceeds {self.daily_loss_sigma}-sigma threshold {threshold:.4f}"
            return True, reason

        return False, ""

    def check_weekly_drawdown(self, pnl_series: pd.Series) -> Tuple[bool, str]:
        """
        Check if weekly drawdown exceeds threshold.

        Args:
            pnl_series: Last 5 trading days of PnL

        Returns:
            (should_halt, reason)
        """
        if len(pnl_series) < 2:
            return False, ""

        # Calculate drawdown over past week
        cum_returns = pnl_series.cumsum()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns.iloc[-1] - running_max.iloc[-1])

        if drawdown < -self.weekly_dd_threshold:
            reason = f"Weekly drawdown {drawdown:.4f} exceeds threshold {-self.weekly_dd_threshold:.4f}"
            return True, reason

        return False, ""

    def check_rolling_sharpe(self, pnl_series: pd.Series) -> Tuple[bool, str]:
        """
        Check if rolling Sharpe has deteriorated.

        Args:
            pnl_series: Historical PnL series

        Returns:
            (should_halt, reason)
        """
        if len(pnl_series) < self.sharpe_window:
            return False, ""

        # Calculate rolling Sharpe
        recent_pnl = pnl_series.iloc[-self.sharpe_window:]
        sharpe = (recent_pnl.mean() / recent_pnl.std() * np.sqrt(252)
                 if recent_pnl.std() > 0 else 0)

        if sharpe < self.sharpe_threshold:
            reason = f"{self.sharpe_window}-day Sharpe {sharpe:.2f} below threshold {self.sharpe_threshold:.2f}"
            return True, reason

        return False, ""

    def evaluate(
        self,
        date: str,
        daily_pnl: float,
        pnl_history: pd.Series
    ) -> Dict:
        """
        Evaluate all kill switches.

        Args:
            date: Current date
            daily_pnl: Today's PnL
            pnl_history: Full PnL history up to today

        Returns:
            result dict with halt status and reason
        """
        if self.is_halted:
            return {
                'halt': True,
                'reason': f"Previously halted on {self.halt_date}: {self.halt_reason}",
                'trigger': 'previous_halt'
            }

        # Check 1: Daily loss
        if len(pnl_history) >= 20:
            historical_std = pnl_history.std()
            halt, reason = self.check_daily_loss(daily_pnl, historical_std)
            if halt:
                self.is_halted = True
                self.halt_reason = reason
                self.halt_date = date
                return {'halt': True, 'reason': reason, 'trigger': 'daily_loss'}

        # Check 2: Weekly drawdown
        if len(pnl_history) >= 5:
            halt, reason = self.check_weekly_drawdown(pnl_history.iloc[-5:])
            if halt:
                self.is_halted = True
                self.halt_reason = reason
                self.halt_date = date
                return {'halt': True, 'reason': reason, 'trigger': 'weekly_drawdown'}

        # Check 3: Rolling Sharpe
        if len(pnl_history) >= self.sharpe_window:
            halt, reason = self.check_rolling_sharpe(pnl_history)
            if halt:
                self.is_halted = True
                self.halt_reason = reason
                self.halt_date = date
                return {'halt': True, 'reason': reason, 'trigger': 'rolling_sharpe'}

        return {'halt': False, 'reason': '', 'trigger': None}


def run_with_capital_management(
    pnl_df: pd.DataFrame,
    positions_history: List[Dict],
    target_vol: float = 0.08,
    enable_kill_switches: bool = True,
    logger=None
) -> Dict:
    """
    Run backtest with capital scaling and kill switches.

    Args:
        pnl_df: Historical PnL DataFrame
        positions_history: Historical positions
        target_vol: Target annual volatility
        enable_kill_switches: Whether to enable kill switches
        logger: Logger instance

    Returns:
        results dict with scaled performance and kill switch events
    """
    logger = logger or setup_logger(__name__)

    logger.info("\n" + "="*70)
    logger.info("PHASE 3.4: CAPITAL SCALING & KILL SWITCHES")
    logger.info("="*70)
    logger.info("")
    logger.info(f"Target Volatility: {target_vol*100:.0f}%")
    logger.info(f"Kill Switches: {'ENABLED' if enable_kill_switches else 'DISABLED'}")
    logger.info("")

    # Apply volatility targeting
    scaler = CapitalScaling(target_vol=target_vol, logger=logger)
    scaled_pnl_df, scaled_positions = scaler.apply_volatility_targeting(
        pnl_df, positions_history
    )

    logger.info("VOLATILITY TARGETING APPLIED")
    logger.info(f"  Average scalar: {scaled_pnl_df['vol_scalar'].mean():.2f}x")
    logger.info(f"  Min scalar:     {scaled_pnl_df['vol_scalar'].min():.2f}x")
    logger.info(f"  Max scalar:     {scaled_pnl_df['vol_scalar'].max():.2f}x")
    logger.info("")

    # Initialize kill switches
    kill_switches = KillSwitches(logger=logger)
    kill_events = []

    # Simulate trading with kill switches
    final_pnl = []
    for i, row in scaled_pnl_df.iterrows():
        pnl_history = scaled_pnl_df.iloc[:i]['pnl_net']

        if enable_kill_switches:
            # Evaluate kill switches
            result = kill_switches.evaluate(
                row['date'],
                row['pnl_net'],
                pnl_history
            )

            if result['halt']:
                kill_events.append({
                    'date': row['date'],
                    'trigger': result['trigger'],
                    'reason': result['reason']
                })

                # After halt, PnL = 0 (flattened positions)
                final_pnl.append({
                    **row.to_dict(),
                    'pnl_gross': 0,
                    'pnl_net': 0,
                    'pnl_long': 0,
                    'pnl_short': 0,
                    'halted': True
                })
                continue

        final_pnl.append({
            **row.to_dict(),
            'halted': False
        })

    final_pnl_df = pd.DataFrame(final_pnl)

    # Calculate final metrics
    metrics = calculate_final_metrics(final_pnl_df, logger)

    # Report kill switch events
    if kill_events:
        logger.info("\n" + "="*70)
        logger.info("KILL SWITCH EVENTS")
        logger.info("="*70)
        for event in kill_events:
            logger.info(f"  {event['date']}: [{event['trigger'].upper()}] {event['reason']}")
    else:
        logger.info("\n[SUCCESS] No kill switches triggered")

    logger.info("")
    logger.info("="*70)
    logger.info("FINAL PERFORMANCE WITH CAPITAL MANAGEMENT")
    logger.info("="*70)
    logger.info(f"  Net Sharpe:          {metrics['sharpe_net']:.2f}")
    logger.info(f"  Annualized Return:   {metrics['annualized_return']*100:.2f}%")
    logger.info(f"  Realized Volatility: {metrics['realized_vol']*100:.2f}%")
    logger.info(f"  Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Days Halted:         {metrics['days_halted']} / {len(final_pnl_df)}")
    logger.info("")

    return {
        'final_pnl_df': final_pnl_df,
        'scaled_positions': scaled_positions,
        'kill_events': kill_events,
        'metrics': metrics
    }


def calculate_final_metrics(pnl_df: pd.DataFrame, logger=None) -> Dict:
    """Calculate final performance metrics."""
    # Filter out halted days for Sharpe calculation
    active_pnl = pnl_df[~pnl_df.get('halted', False)]['pnl_net']

    sharpe_net = (active_pnl.mean() / active_pnl.std() * np.sqrt(252)
                 if active_pnl.std() > 0 else 0)

    # Drawdown
    cum_returns = pnl_df['pnl_net'].cumsum()
    running_max = cum_returns.expanding().max()
    drawdown = cum_returns - running_max
    max_drawdown = drawdown.min()

    # Returns
    total_return = pnl_df['pnl_net'].sum()
    annualized_return = pnl_df['pnl_net'].mean() * 252

    # Realized volatility
    realized_vol = pnl_df['pnl_net'].std() * np.sqrt(252)

    # Days halted
    days_halted = pnl_df.get('halted', pd.Series([False]*len(pnl_df))).sum()

    return {
        'sharpe_net': float(sharpe_net),
        'annualized_return': float(annualized_return),
        'total_return': float(total_return),
        'max_drawdown': float(max_drawdown),
        'realized_vol': float(realized_vol),
        'days_halted': int(days_halted),
        'n_days': int(len(pnl_df))
    }


def main():
    """Run Phase 3.4 capital scaling and kill switches."""
    logger = setup_logger('phase3_4_capital', log_file='logs/phase3_4_capital_scaling.log', level='INFO')

    logger.info("="*70)
    logger.info("PHASE 3.4: CAPITAL SCALING & KILL SWITCHES")
    logger.info("="*70)
    logger.info("")

    # Load Phase 3.3 results (use best configuration)
    results_file = Path("data/processed/phase3_3_execution_stress_results.json")
    if not results_file.exists():
        logger.error(f"Phase 3.3 results not found: {results_file}")
        logger.error("Please run Phase 3.3 first")
        return

    # For now, we'll re-run the best config from Phase 3.3
    # In practice, you'd load the saved PnL from Phase 3.3
    logger.info("Re-running best configuration to get PnL history...")
    logger.info("(In production, this would load from Phase 3.3 results)")
    logger.info("")

    # Load predictions and run best config
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        logger.info("PyArrow installed. Please re-run.")
        return

    predictions_df = pd.read_parquet("data/processed/phase1_predictions.parquet")

    from scripts.stress_test.phase3_3_execution_reality import ExecutionStressTest
    from src.utils.config_loader import ConfigLoader

    config = ConfigLoader('config/config.yaml')
    tester = ExecutionStressTest(predictions_df, config)

    # Run best config (Long-biased 70/30) to get PnL and positions
    # This is a simplified version - in practice load from Phase 3.3
    logger.info("Generating PnL history from Long-Biased (70/30) configuration...")
    logger.info("")

    # Placeholder - would need to modify ExecutionStressTest to return positions
    # For now, demonstrate the capital management framework

    logger.info("="*70)
    logger.info("CAPITAL MANAGEMENT FRAMEWORK READY")
    logger.info("="*70)
    logger.info("")
    logger.info("Components implemented:")
    logger.info("  [x] Volatility Targeting (8% annual)")
    logger.info("  [x] Daily Loss Kill Switch (3-sigma)")
    logger.info("  [x] Weekly Drawdown Kill Switch (8%)")
    logger.info("  [x] Rolling Sharpe Kill Switch (60-day)")
    logger.info("")
    logger.info("To fully test:")
    logger.info("  1. Modify Phase 3.3 to save full PnL + positions history")
    logger.info("  2. Load that data here")
    logger.info("  3. Apply volatility targeting + kill switches")
    logger.info("  4. Compare performance")
    logger.info("")
    logger.info("="*70)


if __name__ == "__main__":
    main()
