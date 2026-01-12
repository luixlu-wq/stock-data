"""
PHASE 2: Evaluation Script

Evaluates Phase 2 model and compares against:
- Baseline (original LSTM with MSE loss)
- Phase 1 (rank loss with 14 features, 60-day sequence)
- Phase 2 (enhanced rank loss with 21 features, 120-day sequence)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor_v3 import EnhancedStockPreprocessor
from src.models.lstm_model import create_model
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_all_metrics, print_metrics
from phase0_backtest import CrossSectionalBacktest


def main():
    """Main evaluation function."""

    # Setup
    config = ConfigLoader('config/config.yaml')
    logger = setup_logger(
        name='phase2_eval',
        log_file='logs/phase2_eval.log',
        level='INFO'
    )

    logger.info("="*70)
    logger.info("PHASE 2: MODEL EVALUATION")
    logger.info("="*70)

    # ===== LOAD TEST DATA =====
    logger.info("Loading test data...")

    test_file = "data/processed/test_v3.parquet"
    test_df = pd.read_parquet(test_file)

    logger.info(f"Test data shape: {test_df.shape}")

    # ===== CREATE TEST SEQUENCES =====
    logger.info("Creating test sequences...")

    preprocessor = EnhancedStockPreprocessor(config)
    preprocessor.feature_columns = preprocessor.get_feature_columns(test_df)

    sequence_length = 120  # Phase 2 uses 120-day sequences

    X_test, y_test_reg, y_test_clf, metadata = preprocessor.create_sequences(
        test_df, sequence_length
    )

    logger.info(f"Test sequences: {X_test.shape}")

    # ===== LOAD PHASE 2 MODEL =====
    logger.info("Loading Phase 2 model...")

    input_size = X_test.shape[2]
    model_config = config.get('model.architecture')

    model = create_model(
        model_type="regression",
        input_size=input_size,
        config=model_config,
        num_classes=3
    )

    checkpoint_path = Path("models/checkpoints/lstm_phase2_rank_best.pth")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please run phase2_train.py first")
        return

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Model loaded from {checkpoint_path}")

    # ===== GENERATE PREDICTIONS =====
    logger.info("Generating predictions...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().squeeze()

    # ===== SAVE PREDICTIONS =====
    logger.info("Saving predictions...")

    results_df = metadata.copy()
    results_df['y_true'] = y_test_reg
    results_df['y_pred'] = predictions

    results_file = "data/processed/phase2_predictions.parquet"
    results_df.to_parquet(results_file)

    logger.info(f"Predictions saved to {results_file}")

    # ===== REGRESSION METRICS =====
    predictions_array = predictions.squeeze()

    metrics = calculate_all_metrics(
        y_true=y_test_reg,
        y_pred=predictions_array,
        model_type="regression"
    )

    logger.info("")
    logger.info("="*70)
    logger.info("REGRESSION METRICS")
    logger.info("="*70)
    print_metrics(metrics, "regression")

    # ===== RUN CROSS-SECTIONAL BACKTEST =====
    logger.info("")
    logger.info("="*70)
    logger.info("RUNNING CROSS-SECTIONAL BACKTEST (Phase 0)")
    logger.info("="*70)
    logger.info("Comparing Phase 2 model...")
    logger.info("")

    backtester = CrossSectionalBacktest(
        long_pct=0.2,
        short_pct=0.2,
        transaction_cost_bps=5.0
    )

    metrics = backtester.run_backtest(results_df)

    # Save metrics
    metrics_file = "data/processed/phase2_metrics.json"
    import json
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to: {metrics_file}")

    # ===== COMPARISON TABLE =====
    logger.info("")
    logger.info("="*70)
    logger.info("COMPARISON: BASELINE vs PHASE 1 vs PHASE 2")
    logger.info("="*70)

    # Baseline results (from previous runs)
    baseline = {
        'sharpe_net': -1.74,
        'sharpe_gross': 0.71,
        'annual_return_net': -0.1082,
        'avg_turnover': 1.2071,
        'win_rate': 0.4402
    }

    # Phase 1 results (from previous runs)
    phase1 = {
        'sharpe_net': -1.04,
        'sharpe_gross': 0.55,
        'annual_return_net': -0.0787,
        'avg_turnover': 0.9518,
        'win_rate': 0.4309
    }

    # Phase 2 results (current)
    phase2 = metrics

    # Print comparison
    logger.info(f"{'Metric':<25} | {'Baseline':<10} | {'Phase 1':<10} | {'Phase 2':<10} | {'Change':<10}")
    logger.info("-"*70)

    def format_val(val, is_pct=True):
        if is_pct:
            return f"{val*100:>7.2f}%"
        else:
            return f"{val:>8.2f}"

    logger.info(f"{'Sharpe (Net)':<25} | {format_val(baseline['sharpe_net'], False):<10} | {format_val(phase1['sharpe_net'], False):<10} | {format_val(phase2['sharpe_net'], False):<10} | {format_val(phase2['sharpe_net'] - baseline['sharpe_net'], False):<10}")
    logger.info(f"{'Sharpe (Gross)':<25} | {format_val(baseline['sharpe_gross'], False):<10} | {format_val(phase1['sharpe_gross'], False):<10} | {format_val(phase2['sharpe_gross'], False):<10} | {format_val(phase2['sharpe_gross'] - baseline['sharpe_gross'], False):<10}")
    logger.info(f"{'Annual Return (Net)':<25} | {format_val(baseline['annual_return_net']):<10} | {format_val(phase1['annual_return_net']):<10} | {format_val(phase2['annual_return_net']):<10} | {format_val(phase2['annual_return_net'] - baseline['annual_return_net']):<10}")
    logger.info(f"{'Turnover':<25} | {format_val(baseline['avg_turnover']):<10} | {format_val(phase1['avg_turnover']):<10} | {format_val(phase2['avg_turnover']):<10} | {format_val(phase2['avg_turnover'] - baseline['avg_turnover']):<10}")
    logger.info(f"{'Win Rate':<25} | {format_val(baseline['win_rate']):<10} | {format_val(phase1['win_rate']):<10} | {format_val(phase2['win_rate']):<10} | {format_val(phase2['win_rate'] - baseline['win_rate']):<10}")
    logger.info("="*70)

    # ===== VERDICT =====
    logger.info("")
    logger.info("VERDICT:")

    sharpe_improvement = phase2['sharpe_net'] - baseline['sharpe_net']
    turnover_improvement = baseline['avg_turnover'] - phase2['avg_turnover']

    if phase2['sharpe_net'] > 0.5:
        logger.info("SUCCESS: Phase 2 is profitable and tradeable!")
        logger.info(f"  Net Sharpe: {phase2['sharpe_net']:.2f} (target: >0.5)")
        logger.info(f"  Turnover reduced by: {turnover_improvement*100:.1f}%")
        logger.info(f"  Total improvement: {sharpe_improvement:.2f} Sharpe points")
    elif phase2['sharpe_net'] > 0:
        logger.info("IMPROVEMENT: Phase 2 is positive but weak")
        logger.info(f"  Net Sharpe: {phase2['sharpe_net']:.2f} (needs >0.5 for profitability)")
        logger.info(f"  Turnover reduced by: {turnover_improvement*100:.1f}%")
        logger.info(f"  Consider Phase 3: ensemble models or different architectures")
    elif sharpe_improvement > 0.5:
        logger.info("PROGRESS: Phase 2 shows improvement but still unprofitable")
        logger.info(f"  Net Sharpe: {phase2['sharpe_net']:.2f} (improved by {sharpe_improvement:.2f})")
        logger.info(f"  Turnover reduced by: {turnover_improvement*100:.1f}%")
        logger.info(f"  Next: Try different model architectures (Transformer, GRU)")
    else:
        logger.info("LIMITED IMPROVEMENT: Phase 2 still struggling")
        logger.info(f"  Net Sharpe: {phase2['sharpe_net']:.2f}")
        logger.info(f"  Consider: Data quality issues, feature leakage, or fundamental problems")

    logger.info("")
    logger.info("="*70)


if __name__ == "__main__":
    main()
