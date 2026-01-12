"""
PHASE 0: Alpha Existence Test
Tests if your current LSTM has ANY predictive signal using cross-sectional ranking.

This is the CRITICAL first step - don't build anything else until this shows promise.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger


class CrossSectionalBacktest:
    """Cross-sectional long-short backtest with transaction costs."""

    def __init__(
        self,
        long_pct: float = 0.2,
        short_pct: float = 0.2,
        transaction_cost_bps: float = 5.0,
        logger=None
    ):
        """
        Initialize backtester.

        Args:
            long_pct: Percentage of stocks to long (top predictions)
            short_pct: Percentage of stocks to short (bottom predictions)
            transaction_cost_bps: Transaction cost in basis points (1 bp = 0.01%)
            logger: Logger instance
        """
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.transaction_cost = transaction_cost_bps / 10000  # Convert bps to decimal
        self.logger = logger or setup_logger('backtest')

        # Tracking
        self.daily_returns = []
        self.daily_positions = []
        self.turnover_history = []

    def run_backtest(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Run cross-sectional backtest on predictions.

        Args:
            predictions_df: DataFrame with columns [date, ticker, y_pred_reg, y_true_reg]

        Returns:
            Dictionary of performance metrics
        """
        self.logger.info("="*70)
        self.logger.info("PHASE 0: ALPHA EXISTENCE TEST")
        self.logger.info("="*70)
        self.logger.info(f"Testing {len(predictions_df)} predictions")
        self.logger.info(f"Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        self.logger.info(f"Unique tickers: {predictions_df['ticker'].nunique()}")
        self.logger.info(f"Long top {self.long_pct*100:.0f}%, Short bottom {self.short_pct*100:.0f}%")
        self.logger.info(f"Transaction cost: {self.transaction_cost*10000:.1f} bps per trade")
        self.logger.info("")

        # Sort by date
        df = predictions_df.sort_values('date').copy()

        # Get unique dates
        dates = sorted(df['date'].unique())

        self.logger.info(f"Simulating {len(dates)} trading days...")

        prev_positions = set()

        for date in dates:
            # Get all stocks for this date
            daily_data = df[df['date'] == date].copy()

            if len(daily_data) < 5:  # Need minimum stocks
                continue

            # Rank stocks by predicted return
            daily_data = daily_data.sort_values('y_pred_reg', ascending=False).reset_index(drop=True)

            # Calculate number of stocks to long/short
            n_stocks = len(daily_data)
            n_long = max(1, int(n_stocks * self.long_pct))
            n_short = max(1, int(n_stocks * self.short_pct))

            # Select positions
            long_stocks = set(daily_data.iloc[:n_long]['ticker'].tolist())
            short_stocks = set(daily_data.iloc[-n_short:]['ticker'].tolist())

            current_positions = long_stocks | short_stocks

            # Calculate turnover (% of portfolio changed)
            if prev_positions:
                turnover = len(current_positions.symmetric_difference(prev_positions)) / len(current_positions)
            else:
                turnover = 1.0  # First day, full turnover

            self.turnover_history.append(turnover)

            # Calculate returns for long and short positions
            long_returns = daily_data[daily_data['ticker'].isin(long_stocks)]['y_true_reg'].values
            short_returns = daily_data[daily_data['ticker'].isin(short_stocks)]['y_true_reg'].values

            # Portfolio return (equal-weighted, dollar-neutral)
            if len(long_returns) > 0 and len(short_returns) > 0:
                long_ret = np.mean(long_returns)
                short_ret = np.mean(short_returns)
                gross_return = (long_ret - short_ret) / 2  # Dollar-neutral

                # Apply transaction costs
                cost = turnover * self.transaction_cost
                net_return = gross_return - cost

                self.daily_returns.append({
                    'date': date,
                    'gross_return': gross_return,
                    'transaction_cost': cost,
                    'net_return': net_return,
                    'n_long': len(long_returns),
                    'n_short': len(short_returns),
                    'turnover': turnover
                })

            prev_positions = current_positions

        # Calculate metrics
        metrics = self._calculate_metrics()

        return metrics

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.daily_returns:
            self.logger.error("No returns to calculate metrics!")
            return {}

        df = pd.DataFrame(self.daily_returns)

        # Basic stats
        gross_returns = df['gross_return'].values
        net_returns = df['net_return'].values

        # Sharpe ratio (annualized, assuming 252 trading days)
        sharpe_gross = self._calculate_sharpe(gross_returns)
        sharpe_net = self._calculate_sharpe(net_returns)

        # Cumulative returns
        cum_gross = (1 + gross_returns).cumprod()
        cum_net = (1 + net_returns).cumprod()

        # Max drawdown
        dd_gross = self._calculate_max_drawdown(cum_gross)
        dd_net = self._calculate_max_drawdown(cum_net)

        # Turnover
        avg_turnover = np.mean(self.turnover_history)

        # Information Coefficient (mean correlation between prediction and actual)
        # This is a proxy - real IC needs rank correlation per day

        metrics = {
            'n_days': len(df),
            'sharpe_gross': sharpe_gross,
            'sharpe_net': sharpe_net,
            'annual_return_gross': np.mean(gross_returns) * 252,
            'annual_return_net': np.mean(net_returns) * 252,
            'annual_vol': np.std(net_returns) * np.sqrt(252),
            'max_drawdown_gross': dd_gross,
            'max_drawdown_net': dd_net,
            'avg_turnover': avg_turnover,
            'total_costs': df['transaction_cost'].sum(),
            'mean_daily_cost': df['transaction_cost'].mean(),
            'win_rate': (net_returns > 0).sum() / len(net_returns)
        }

        self._print_metrics(metrics)

        # Save detailed results
        self.results_df = df

        return metrics

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        return (mean_ret / std_ret) * np.sqrt(252)

    def _calculate_max_drawdown(self, cum_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return np.min(drawdown)

    def _print_metrics(self, metrics: Dict):
        """Print formatted metrics."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("BACKTEST RESULTS")
        self.logger.info("="*70)
        self.logger.info(f"Trading days: {metrics['n_days']}")
        self.logger.info("")
        self.logger.info("RETURNS (annualized):")
        self.logger.info(f"  Gross: {metrics['annual_return_gross']*100:>6.2f}%")
        self.logger.info(f"  Net:   {metrics['annual_return_net']*100:>6.2f}%")
        self.logger.info("")
        self.logger.info("SHARPE RATIO:")
        self.logger.info(f"  Gross: {metrics['sharpe_gross']:>6.2f}")
        self.logger.info(f"  Net:   {metrics['sharpe_net']:>6.2f}  ← THIS IS THE KEY METRIC")
        self.logger.info("")
        self.logger.info("RISK:")
        self.logger.info(f"  Volatility (annual): {metrics['annual_vol']*100:>6.2f}%")
        self.logger.info(f"  Max Drawdown (net):  {metrics['max_drawdown_net']*100:>6.2f}%")
        self.logger.info("")
        self.logger.info("TRADING:")
        self.logger.info(f"  Avg Turnover:      {metrics['avg_turnover']*100:>6.2f}%")
        self.logger.info(f"  Win Rate:          {metrics['win_rate']*100:>6.2f}%")
        self.logger.info(f"  Total Costs:       {metrics['total_costs']*100:>6.2f}%")
        self.logger.info(f"  Avg Daily Cost:    {metrics['mean_daily_cost']*10000:>6.2f} bps")
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("INTERPRETATION:")
        self.logger.info("="*70)

        sharpe = metrics['sharpe_net']
        if sharpe < 0:
            self.logger.info("❌ Sharpe < 0: NO SIGNAL - Model is worse than random")
            self.logger.info("   → Check data quality, feature leakage, or fundamental issues")
        elif sharpe < 0.3:
            self.logger.info("⚠️  Sharpe 0-0.3: NOISE - No reliable signal")
            self.logger.info("   → Don't proceed with complex models")
            self.logger.info("   → Revisit features and data")
        elif sharpe < 0.5:
            self.logger.info("✓  Sharpe 0.3-0.5: WEAK BUT REAL signal detected")
            self.logger.info("   → Proceed cautiously to Phase 1")
            self.logger.info("   → Focus on feature engineering and rank loss")
        elif sharpe < 1.0:
            self.logger.info("✓✓ Sharpe 0.5-1.0: GOOD signal detected")
            self.logger.info("   → Proceed to Phase 1 (rank loss + feature optimization)")
            self.logger.info("   → This is worth developing further")
        else:
            self.logger.info("✓✓✓ Sharpe > 1.0: EXCELLENT signal (rare!)")
            self.logger.info("   → Verify this is real (check for leakage)")
            self.logger.info("   → If legitimate, proceed to full development")

        self.logger.info("="*70)


def main():
    """Run Phase 0 backtest."""
    logger = setup_logger('phase0', log_file='logs/phase0_backtest.log')

    # Load predictions
    logger.info("Loading model predictions...")

    # Check which prediction files exist
    pred_files = {
        'multitask': 'data/processed/multitask_predictions.parquet',
        'regression': 'data/processed/regression_predictions.parquet',
        'classification': 'data/processed/classification_predictions.parquet'
    }

    # Find available prediction file
    pred_file = None
    model_type = None
    for mtype, fpath in pred_files.items():
        if Path(fpath).exists():
            pred_file = fpath
            model_type = mtype
            break

    if pred_file is None:
        logger.error("No prediction files found! Run evaluation first:")
        logger.error("  python main.py eval-multitask")
        return

    logger.info(f"Using {model_type} predictions from {pred_file}")

    # Load predictions
    df = pd.read_parquet(pred_file)

    logger.info(f"Loaded {len(df)} predictions")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Tickers: {sorted(df['ticker'].unique())}")

    # Create backtester
    backtester = CrossSectionalBacktest(
        long_pct=0.2,
        short_pct=0.2,
        transaction_cost_bps=5.0,
        logger=logger
    )

    # Run backtest
    metrics = backtester.run_backtest(df)

    # Save results
    output_file = 'data/processed/phase0_backtest_results.parquet'
    backtester.results_df.to_parquet(output_file)
    logger.info(f"\nDetailed results saved to: {output_file}")

    # Save metrics
    import json
    metrics_file = 'data/processed/phase0_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
