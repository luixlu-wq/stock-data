"""
PHASE 1: Evaluation Script

Evaluates the Phase 1 model (rank loss + simplified features)
and runs Phase 0 backtest to compare against baseline.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor_v2 import SimplifiedStockPreprocessor
from src.models.lstm_model import create_model
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from phase0_backtest import CrossSectionalBacktest


def main():
    """Run Phase 1 evaluation."""
    logger = setup_logger('phase1_eval', log_file='logs/phase1_evaluation.log')

    logger.info("="*70)
    logger.info("PHASE 1: MODEL EVALUATION")
    logger.info("="*70)

    # Load config
    config = ConfigLoader('config/config.yaml')

    # Load test data
    logger.info("Loading test data...")
    test_file = "data/processed/test_v2.parquet"

    if not Path(test_file).exists():
        logger.error(f"Test file not found: {test_file}")
        logger.error("Run phase1_train.py first!")
        return

    test_df = pd.read_parquet(test_file)
    logger.info(f"Test data shape: {test_df.shape}")

    # Create sequences
    logger.info("Creating test sequences...")

    preprocessor = SimplifiedStockPreprocessor(config)
    preprocessor.feature_columns = preprocessor.get_feature_columns(test_df)

    sequence_length = config.get('data.sequence_length', 60)
    X_test, y_test_reg, y_test_clf, metadata = preprocessor.create_sequences(
        test_df, sequence_length
    )

    logger.info(f"Test sequences: {X_test.shape}")

    # Load model
    logger.info("Loading Phase 1 model...")

    input_size = X_test.shape[2]
    model_config = config.get('model.architecture')

    model = create_model(
        model_type="regression",
        input_size=input_size,
        config=model_config
    )

    checkpoint_path = Path("models/checkpoints/lstm_phase1_rank_best.pth")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Train the model first with phase1_train.py")
        return

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from {checkpoint_path}")

    # Make predictions
    logger.info("Generating predictions...")

    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().squeeze()

    # Save predictions
    logger.info("Saving predictions...")

    results_df = metadata.copy()
    results_df['y_true_reg'] = y_test_reg
    results_df['y_pred_reg'] = predictions
    results_df['y_true_clf'] = y_test_clf

    results_file = "data/processed/phase1_predictions.parquet"
    results_df.to_parquet(results_file)

    logger.info(f"Predictions saved to {results_file}")

    # Calculate basic metrics
    from src.utils.metrics import calculate_all_metrics, print_metrics

    metrics = calculate_all_metrics(
        y_true=y_test_reg,
        y_pred=predictions,
        model_type="regression"
    )

    logger.info("")
    logger.info("="*70)
    logger.info("REGRESSION METRICS")
    logger.info("="*70)
    print_metrics(metrics, "regression")

    # Run Phase 0 backtest
    logger.info("")
    logger.info("="*70)
    logger.info("RUNNING CROSS-SECTIONAL BACKTEST (Phase 0)")
    logger.info("="*70)
    logger.info("Comparing Phase 1 model against baseline...")
    logger.info("")

    backtester = CrossSectionalBacktest(
        long_pct=0.2,
        short_pct=0.2,
        transaction_cost_bps=5.0,
        logger=logger
    )

    metrics = backtester.run_backtest(results_df)

    # Save backtest results
    output_file = 'data/processed/phase1_backtest_results.parquet'
    backtester.results_df.to_parquet(output_file)
    logger.info(f"\nBacktest results saved to: {output_file}")

    # Save metrics
    import json
    metrics_file = 'data/processed/phase1_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_file}")

    # Compare with baseline
    logger.info("")
    logger.info("="*70)
    logger.info("COMPARISON: BASELINE vs PHASE 1")
    logger.info("="*70)

    baseline_metrics_file = 'data/processed/phase0_metrics.json'
    if Path(baseline_metrics_file).exists():
        with open(baseline_metrics_file, 'r') as f:
            baseline_metrics = json.load(f)

        logger.info("Metric                  | Baseline  | Phase 1   | Change")
        logger.info("-"*70)
        logger.info(f"Sharpe (Net)            | {baseline_metrics['sharpe_net']:>8.2f}  | {metrics['sharpe_net']:>8.2f}  | {metrics['sharpe_net'] - baseline_metrics['sharpe_net']:>+7.2f}")
        logger.info(f"Sharpe (Gross)          | {baseline_metrics['sharpe_gross']:>8.2f}  | {metrics['sharpe_gross']:>8.2f}  | {metrics['sharpe_gross'] - baseline_metrics['sharpe_gross']:>+7.2f}")
        logger.info(f"Annual Return (Net)     | {baseline_metrics['annual_return_net']*100:>7.2f}% | {metrics['annual_return_net']*100:>7.2f}% | {(metrics['annual_return_net'] - baseline_metrics['annual_return_net'])*100:>+6.2f}%")
        logger.info(f"Turnover                | {baseline_metrics['avg_turnover']*100:>7.2f}% | {metrics['avg_turnover']*100:>7.2f}% | {(metrics['avg_turnover'] - baseline_metrics['avg_turnover'])*100:>+6.2f}%")
        logger.info(f"Win Rate                | {baseline_metrics['win_rate']*100:>7.2f}% | {metrics['win_rate']*100:>7.2f}% | {(metrics['win_rate'] - baseline_metrics['win_rate'])*100:>+6.2f}%")
        logger.info("="*70)

        # Verdict
        logger.info("")
        logger.info("VERDICT:")
        if metrics['sharpe_net'] > 0.3:
            logger.info("SUCCESS: Phase 1 achieves tradeable alpha (Sharpe > 0.3)")
            logger.info("  Next step: Consider Phase 2 improvements (turnover penalty, clustering)")
        elif metrics['sharpe_net'] > baseline_metrics['sharpe_net']:
            logger.info("IMPROVEMENT: Phase 1 is better than baseline but still weak")
            logger.info("  Sharpe not yet profitable. Consider:")
            logger.info("    - Longer sequence length (120 days)")
            logger.info("    - Different rank loss temperature")
            logger.info("    - Feature engineering refinement")
        else:
            logger.info("NO IMPROVEMENT: Phase 1 did not help")
            logger.info("  Possible issues:")
            logger.info("    - Rank loss not suitable for this data")
            logger.info("    - Features too simplified")
            logger.info("    - Data quality problems")

    logger.info("")
    logger.info("="*70)


if __name__ == "__main__":
    main()
