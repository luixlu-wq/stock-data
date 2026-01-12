"""
Phase 2B: Baseline Validation (ChatGPT's Critical Check)

ChatGPT identified a red flag: Baseline results are too good compared to Phase 2A.

This script creates a TRUE baseline that:
1. Forces full rebalance every day
2. NO position carry-forward
3. NO EWMA smoothing
4. NO dollar neutrality enforcement
5. NO rank filtering

This should match Phase 2A's poor performance (~0.1 Net Sharpe, 100% turnover).
If it doesn't, then Phase 2A had bugs or Phase 2B has implicit smoothing.
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessor_v2 import SimplifiedStockPreprocessor
from src.models.lstm_model import create_model
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class TrueBaselineBacktest:
    """
    ABSOLUTE MINIMAL backtest - no engineering whatsoever.

    Every day:
    1. Rank stocks
    2. Long top 20%, short bottom 20%
    3. Full rebalance (100% turnover every day in theory)
    4. Apply costs
    5. Calculate PnL
    6. RESET positions to empty
    """

    def __init__(self, long_pct=0.2, short_pct=0.2, transaction_cost_bps=5.0):
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.transaction_cost = transaction_cost_bps / 10000
        self.logger = setup_logger('true_baseline', 'logs/phase2b_true_baseline.log', 'INFO')

    def run_backtest(self, predictions_df: pd.DataFrame) -> dict:
        """Run TRUE baseline backtest."""

        self.logger.info("="*70)
        self.logger.info("TRUE BASELINE BACKTEST (ChatGPT's Validation)")
        self.logger.info("="*70)
        self.logger.info("NO smoothing, NO carry-forward, NO neutrality enforcement")
        self.logger.info("Full daily rebalance - positions reset every day")
        self.logger.info("="*70)
        self.logger.info("")

        df = predictions_df.copy()
        df = df.sort_values('date')
        dates = sorted(df['date'].unique())

        daily_pnl = []
        daily_turnover_list = []

        for date in dates:
            daily_data = df[df['date'] == date].copy()

            if len(daily_data) < 5:
                continue

            # Rank stocks
            daily_data = daily_data.sort_values('y_pred_reg', ascending=False).reset_index(drop=True)

            # Select positions
            n_stocks = len(daily_data)
            n_long = max(1, int(n_stocks * self.long_pct))
            n_short = max(1, int(n_stocks * self.short_pct))

            # Create positions dict
            positions = {}

            # Long top 20%
            for ticker in daily_data.iloc[:n_long]['ticker']:
                positions[ticker] = 1.0 / n_long

            # Short bottom 20%
            for ticker in daily_data.iloc[-n_short:]['ticker']:
                positions[ticker] = -1.0 / n_short

            # Calculate turnover (assume 100% first day, then actual changes)
            if not daily_turnover_list:
                turnover = 1.0  # First day: full deployment
            else:
                # In true baseline, we rebalance 100% every day
                # because we don't carry forward positions
                turnover = 1.0

            daily_turnover_list.append(turnover)

            # Calculate costs (one-way)
            trading_cost = turnover * self.transaction_cost

            # Calculate PnL
            pnl_gross = 0
            returns_map = daily_data.set_index('ticker')['y_true_reg'].to_dict()

            for ticker, position in positions.items():
                if ticker in returns_map:
                    pnl_gross += position * returns_map[ticker]

            pnl_net = pnl_gross - trading_cost

            daily_pnl.append({
                'date': date,
                'pnl_gross': pnl_gross,
                'pnl_net': pnl_net,
                'turnover': turnover,
                'cost': trading_cost
            })

            # CRITICAL: Do NOT carry forward positions
            # (positions reset to empty for next day)

        # Calculate metrics
        pnl_df = pd.DataFrame(daily_pnl)

        gross_returns = pnl_df['pnl_gross'].values
        net_returns = pnl_df['pnl_net'].values

        metrics = {
            'sharpe_gross': np.mean(gross_returns) / (np.std(gross_returns) + 1e-8) * np.sqrt(252),
            'sharpe_net': np.mean(net_returns) / (np.std(net_returns) + 1e-8) * np.sqrt(252),
            'annual_return_gross': np.mean(gross_returns) * 252,
            'annual_return_net': np.mean(net_returns) * 252,
            'avg_turnover': np.mean(daily_turnover_list),
            'total_costs': pnl_df['cost'].sum(),
            'trading_days': len(pnl_df)
        }

        # Print results
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("TRUE BASELINE RESULTS")
        self.logger.info("="*70)
        self.logger.info(f"Trading days: {metrics['trading_days']}")
        self.logger.info("")
        self.logger.info("SHARPE RATIO:")
        self.logger.info(f"  Gross: {metrics['sharpe_gross']:>8.2f}")
        self.logger.info(f"  Net:   {metrics['sharpe_net']:>8.2f}")
        self.logger.info("")
        self.logger.info("RETURNS (annualized):")
        self.logger.info(f"  Gross: {metrics['annual_return_gross']*100:>7.2f}%")
        self.logger.info(f"  Net:   {metrics['annual_return_net']*100:>7.2f}%")
        self.logger.info("")
        self.logger.info("COSTS:")
        self.logger.info(f"  Avg Turnover: {metrics['avg_turnover']*100:>6.1f}%")
        self.logger.info(f"  Total Costs:  {metrics['total_costs']*100:>6.2f}%")
        self.logger.info("="*70)

        return metrics


def main():
    """Compare TRUE baseline vs Phase 2B baseline."""

    config = ConfigLoader('config/config.yaml')
    logger = setup_logger('validate', 'logs/phase2b_validate.log', 'INFO')

    logger.info("="*70)
    logger.info("PHASE 2B: BASELINE VALIDATION")
    logger.info("="*70)
    logger.info("")
    logger.info("ChatGPT's Question:")
    logger.info("  Why is baseline Net Sharpe 2.02 when Phase 2A showed 0.10?")
    logger.info("")
    logger.info("This script tests TRUE baseline (full daily rebalance)")
    logger.info("Expected result: Net Sharpe ~ 0.1-0.3, Turnover ~ 100%")
    logger.info("="*70)
    logger.info("")

    # Load test data
    test_file = "data/processed/test_v2.parquet"
    test_df = pd.read_parquet(test_file)

    preprocessor = SimplifiedStockPreprocessor(config)
    preprocessor.feature_columns = preprocessor.get_feature_columns(test_df)

    sequence_length = 90
    X_test, y_test_reg, y_test_clf, test_metadata = preprocessor.create_sequences(
        test_df, sequence_length
    )

    logger.info(f"Test sequences: {X_test.shape}")

    # Load model
    input_size = X_test.shape[2]
    model_config = config.get('model.architecture')

    model = create_model(
        model_type="regression",
        input_size=input_size,
        config=model_config,
        num_classes=3
    )

    checkpoint_path = Path("models/checkpoints/lstm_phase2a_temp0.05_best.pth")

    if not checkpoint_path.exists():
        logger.error(f"Model not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Generate predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().squeeze()

    results_df = test_metadata.copy()
    results_df['y_true_reg'] = y_test_reg
    results_df['y_pred_reg'] = predictions

    # Run TRUE baseline
    backtester = TrueBaselineBacktest(
        long_pct=0.2,
        short_pct=0.2,
        transaction_cost_bps=5.0
    )

    metrics = backtester.run_backtest(results_df)

    # Comparison
    logger.info("")
    logger.info("="*70)
    logger.info("COMPARISON:")
    logger.info("="*70)
    logger.info("")
    logger.info(f"{'Metric':<20} | {'Phase 2A':<12} | {'Phase 2B Baseline':<18} | {'TRUE Baseline':<15}")
    logger.info("-"*70)
    logger.info(f"{'Net Sharpe':<20} | {'-1.04 to 0.10':<12} | {'2.02':<18} | {metrics['sharpe_net']:<15.2f}")
    logger.info(f"{'Gross Sharpe':<20} | {'0.55 to 2.47':<12} | {'2.53':<18} | {metrics['sharpe_gross']:<15.2f}")
    logger.info(f"{'Turnover':<20} | {'~100%':<12} | {'41%':<18} | {metrics['avg_turnover']*100:<14.1f}%")
    logger.info("="*70)
    logger.info("")

    # Verdict
    logger.info("VERDICT:")
    if abs(metrics['sharpe_net'] - 2.02) < 0.5:
        logger.info("[CONCERN] TRUE baseline also shows high Net Sharpe")
        logger.info("  This suggests Phase 2B baseline is correct")
        logger.info("  Phase 2A may have had bugs or different data")
    elif metrics['sharpe_net'] < 0.5:
        logger.info("[VALIDATED] TRUE baseline shows poor Net Sharpe")
        logger.info("  Phase 2B baseline has implicit smoothing")
        logger.info("  Position smoothing is the real improvement")
    else:
        logger.info("[UNCLEAR] Results in between - investigate further")

    logger.info("")
    logger.info("="*70)


if __name__ == "__main__":
    main()
