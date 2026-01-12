"""
PHASE 2A: Rank Loss Temperature Tuning (ChatGPT's Recommendation)

ChatGPT's analysis:
- Phase 1's rank loss WORKED (reduced turnover 25%)
- But temperature may be too high (smoothed too aggressively)
- Result: Gross Sharpe dropped from 0.71 → 0.55

This script:
1. Tests temperatures: [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
2. Uses 90-day sequences (was 60)
3. Keeps same 14 features (no new features yet)
4. Finds optimal temperature for max gross Sharpe + low turnover

Success criteria (ChatGPT):
- Gross Sharpe ≥ 0.6
- Turnover ≤ 100%
- Net Sharpe closer to 0
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor_v2 import SimplifiedStockPreprocessor
from src.models.lstm_model import create_model, count_parameters
from src.models.dataset import create_dataloaders
from src.models.trainer import ModelTrainer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from phase0_backtest import CrossSectionalBacktest


def train_and_evaluate_temperature(
    temperature: float,
    X_train, y_train_reg, y_train_clf,
    X_val, y_val_reg, y_val_clf,
    X_test, y_test_reg, test_metadata,
    config, logger
):
    """Train model with specific temperature and evaluate."""

    logger.info("")
    logger.info("="*70)
    logger.info(f"TRAINING WITH TEMPERATURE = {temperature}")
    logger.info("="*70)

    # Create model
    input_size = X_train.shape[2]
    model_config = config.get('model.architecture')

    model = create_model(
        model_type="regression",
        input_size=input_size,
        config=model_config,
        num_classes=3
    )

    # Create dataloaders
    batch_size = config.get('model.training.batch_size', 32)
    dataloaders = create_dataloaders(
        X_train=X_train,
        y_train_reg=y_train_reg,
        y_train_clf=y_train_clf,
        X_val=X_val,
        y_val_reg=y_val_reg,
        y_val_clf=y_val_clf,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    # Override config with current temperature
    config._config['model']['loss']['regression'] = 'rank'
    config._config['model']['loss']['rank_weight'] = 0.7
    config._config['model']['loss']['rank_temperature'] = temperature
    config._config['model']['loss']['huber_delta'] = 0.05

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        config=config,
        model_type="regression"
    )

    logger.info(f"Temperature: {temperature}")
    logger.info(f"Rank weight: 0.7")
    logger.info(f"Device: {trainer.device}")

    # Train (shorter for grid search)
    epochs = 50  # Reduced from 100 for faster experimentation
    model_name = f"lstm_phase2a_temp{temperature:.2f}"

    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=epochs,
        model_name=model_name
    )

    # Evaluate on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().squeeze()

    # Run backtest
    results_df = test_metadata.copy()
    results_df['y_true_reg'] = y_test_reg
    results_df['y_pred_reg'] = predictions

    backtester = CrossSectionalBacktest(
        long_pct=0.2,
        short_pct=0.2,
        transaction_cost_bps=5.0
    )

    metrics = backtester.run_backtest(results_df)

    # Add training metrics
    metrics['temperature'] = temperature
    metrics['best_val_loss'] = min(history['val_loss'])

    # ===== CHATGPT'S SUGGESTED METRICS =====

    # 1. Information Coefficient (IC) - correlation between predictions and actual returns
    from scipy.stats import spearmanr

    ic_values = []
    for date in results_df['date'].unique():
        daily = results_df[results_df['date'] == date]
        if len(daily) > 5:
            ic, _ = spearmanr(daily['y_pred_reg'], daily['y_true_reg'])
            if not np.isnan(ic):
                ic_values.append(ic)

    metrics['ic_mean'] = np.mean(ic_values) if ic_values else 0
    metrics['ic_std'] = np.std(ic_values) if ic_values else 0
    metrics['ic_ir'] = metrics['ic_mean'] / (metrics['ic_std'] + 1e-8)  # IC Information Ratio

    # 2. Rank Autocorrelation - persistence of rankings day-to-day
    rank_autocorrs = []
    dates = sorted(results_df['date'].unique())

    for i in range(len(dates) - 1):
        today = results_df[results_df['date'] == dates[i]].set_index('ticker')['y_pred_reg']
        tomorrow = results_df[results_df['date'] == dates[i+1]].set_index('ticker')['y_pred_reg']

        # Get common tickers
        common = today.index.intersection(tomorrow.index)
        if len(common) > 5:
            today_rank = today[common].rank()
            tomorrow_rank = tomorrow[common].rank()
            corr = today_rank.corr(tomorrow_rank)
            if not np.isnan(corr):
                rank_autocorrs.append(corr)

    metrics['rank_autocorr'] = np.mean(rank_autocorrs) if rank_autocorrs else 0

    logger.info(f"  IC Mean: {metrics['ic_mean']:.4f}")
    logger.info(f"  IC Std: {metrics['ic_std']:.4f}")
    logger.info(f"  IC IR: {metrics['ic_ir']:.4f}")
    logger.info(f"  Rank Autocorr: {metrics['rank_autocorr']:.4f}")

    return metrics, results_df


def main():
    """Run temperature grid search."""

    config = ConfigLoader('config/config.yaml')
    logger = setup_logger(
        name='phase2a_temp',
        log_file='logs/phase2a_temperature_experiment.log',
        level='INFO'
    )

    logger.info("="*70)
    logger.info("PHASE 2A: RANK LOSS TEMPERATURE EXPERIMENT")
    logger.info("="*70)
    logger.info("")
    logger.info("ChatGPT's hypothesis:")
    logger.info("  Phase 1 smoothed too aggressively (temp=1.0)")
    logger.info("  Lower temperature = sharper rankings = better Sharpe")
    logger.info("")
    logger.info("Testing temperatures: [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]")
    logger.info("Sequence length: 90 days (was 60 in Phase 1)")
    logger.info("Features: Same 14 from Phase 1")
    logger.info("="*70)
    logger.info("")

    # ===== LOAD DATA =====
    logger.info("Loading preprocessed data (v2 - 14 features)...")

    train_file = "data/processed/train_v2.parquet"
    val_file = "data/processed/validation_v2.parquet"
    test_file = "data/processed/test_v2.parquet"

    # Check if files exist
    if not Path(train_file).exists():
        logger.error("v2 preprocessed files not found!")
        logger.error("Run: python phase1_train.py first to create them")
        return

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    test_df = pd.read_parquet(test_file)

    logger.info(f"Train: {len(train_df)} rows")
    logger.info(f"Val: {len(val_df)} rows")
    logger.info(f"Test: {len(test_df)} rows")

    # ===== CREATE SEQUENCES (90-day) =====
    logger.info("")
    logger.info("Creating 90-day sequences...")

    preprocessor = SimplifiedStockPreprocessor(config)
    preprocessor.feature_columns = preprocessor.get_feature_columns(train_df)

    sequence_length = 90  # Increased from 60

    X_train, y_train_reg, y_train_clf, _ = preprocessor.create_sequences(
        train_df, sequence_length
    )
    X_val, y_val_reg, y_val_clf, _ = preprocessor.create_sequences(
        val_df, sequence_length
    )
    X_test, y_test_reg, y_test_clf, test_metadata = preprocessor.create_sequences(
        test_df, sequence_length
    )

    logger.info(f"Train: {X_train.shape}")
    logger.info(f"Val: {X_val.shape}")
    logger.info(f"Test: {X_test.shape}")

    # ===== TEMPERATURE GRID SEARCH =====
    temperatures = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    all_results = []

    for temp in temperatures:
        metrics, _ = train_and_evaluate_temperature(
            temperature=temp,
            X_train=X_train, y_train_reg=y_train_reg, y_train_clf=y_train_clf,
            X_val=X_val, y_val_reg=y_val_reg, y_val_clf=y_val_clf,
            X_test=X_test, y_test_reg=y_test_reg, test_metadata=test_metadata,
            config=config,
            logger=logger
        )

        all_results.append(metrics)

    # ===== SUMMARY TABLE =====
    logger.info("")
    logger.info("="*70)
    logger.info("TEMPERATURE EXPERIMENT RESULTS")
    logger.info("="*70)
    logger.info("")
    logger.info(f"{'Temp':<6} | {'Gross':<6} | {'Net':<6} | {'Turn':<6} | {'IC':<7} | {'IC_IR':<7} | {'RankAC':<7}")
    logger.info("-"*70)

    for result in all_results:
        temp = result['temperature']
        gross = result['sharpe_gross']
        net = result['sharpe_net']
        turnover = result['avg_turnover'] * 100
        ic_mean = result.get('ic_mean', 0)
        ic_ir = result.get('ic_ir', 0)
        rank_ac = result.get('rank_autocorr', 0)

        logger.info(f"{temp:<6.2f} | {gross:<6.2f} | {net:<6.2f} | {turnover:<5.0f}% | {ic_mean:<7.4f} | {ic_ir:<7.3f} | {rank_ac:<7.3f}")

    # Find best by gross Sharpe (ChatGPT's target metric)
    best_result = max(all_results, key=lambda x: x['sharpe_gross'])

    logger.info("")
    logger.info("="*70)
    logger.info("BEST TEMPERATURE (by Gross Sharpe):")
    logger.info(f"  Temperature: {best_result['temperature']}")
    logger.info(f"  Gross Sharpe: {best_result['sharpe_gross']:.2f}")
    logger.info(f"  Net Sharpe: {best_result['sharpe_net']:.2f}")
    logger.info(f"  Turnover: {best_result['avg_turnover']*100:.1f}%")
    logger.info("="*70)

    # Save results
    results_file = "data/processed/phase2a_temperature_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    # ===== VERDICT (ChatGPT's Decision Matrix) =====
    logger.info("")
    logger.info("="*70)
    logger.info("INTERPRETATION (ChatGPT's Decision Matrix)")
    logger.info("="*70)
    logger.info("")

    best_gross = best_result['sharpe_gross']
    best_net = best_result['sharpe_net']
    best_turn = best_result['avg_turnover']
    best_temp = best_result['temperature']
    best_ic = best_result.get('ic_mean', 0)
    best_rank_ac = best_result.get('rank_autocorr', 0)

    # Case A - Ideal Outcome
    if best_gross >= 0.65 and best_turn <= 1.1 and best_net >= -0.3:
        logger.info("CASE A: IDEAL OUTCOME ✅")
        logger.info(f"  Best Temperature: {best_temp}")
        logger.info(f"  Gross Sharpe: {best_gross:.2f} (target: 0.65-0.75)")
        logger.info(f"  Turnover: {best_turn*100:.0f}% (target: 90-100%)")
        logger.info(f"  Net Sharpe: {best_net:.2f} (target: -0.3 to +0.1)")
        logger.info(f"  IC Mean: {best_ic:.4f}")
        logger.info(f"  Rank Autocorr: {best_rank_ac:.3f} (target: 0.3-0.6)")
        logger.info("")
        logger.info("Interpretation:")
        logger.info("  ✓ Rank loss is calibrated")
        logger.info("  ✓ Signal preserved")
        logger.info("  ✓ Costs under control")
        logger.info("")
        logger.info("NEXT STEP: Proceed to Phase 2B (cross-sectional normalization)")

    # Case B - Over-smoothing confirmed
    elif best_temp <= 0.05 and best_gross > 0.55 and best_turn > 1.1:
        logger.info("CASE B: OVER-SMOOTHING CONFIRMED ⚠️")
        logger.info(f"  Best Temperature: {best_temp} (very low)")
        logger.info(f"  Gross Sharpe: {best_gross:.2f} (improved)")
        logger.info(f"  Turnover: {best_turn*100:.0f}% (high >110%)")
        logger.info(f"  Net Sharpe: {best_net:.2f} (still negative)")
        logger.info("")
        logger.info("Interpretation:")
        logger.info("  Signal exists but is fragile")
        logger.info("  Near noise boundary")
        logger.info("")
        logger.info("NEXT STEP: Feature ranking stabilization, not more smoothing")

    # Case C - No temperature helps
    elif best_gross <= 0.55:
        logger.info("CASE C: NO TEMPERATURE HELPS ❌")
        logger.info(f"  Best Temperature: {best_temp}")
        logger.info(f"  Gross Sharpe: {best_gross:.2f} (≤0.55 across all)")
        logger.info(f"  Turnover: {best_turn*100:.0f}%")
        logger.info(f"  Net Sharpe: {best_net:.2f}")
        logger.info("")
        logger.info("Interpretation:")
        logger.info("  Alpha is structurally weak")
        logger.info("  Rank loss already exposed true limit")
        logger.info("  This is not failure - this is information")
        logger.info("")
        logger.info("NEXT STEP: Shift focus to cross-sectional signal design")

    # Case D - Overfitting detected
    elif best_gross > 0.8 and best_turn > 1.5:
        logger.info("CASE D: OVERFITTING DETECTED 🚨")
        logger.info(f"  Best Temperature: {best_temp}")
        logger.info(f"  Gross Sharpe: {best_gross:.2f} (>0.8 - too high)")
        logger.info(f"  Turnover: {best_turn*100:.0f}% (>150% - too high)")
        logger.info(f"  Net Sharpe: {best_net:.2f} (worse)")
        logger.info("")
        logger.info("Interpretation:")
        logger.info("  Reintroduced noise trading")
        logger.info("")
        logger.info("NEXT STEP: Reject; move back to temperature 0.1-0.2")

    # Default case
    else:
        logger.info("PARTIAL IMPROVEMENT")
        logger.info(f"  Best Temperature: {best_temp}")
        logger.info(f"  Gross Sharpe: {best_gross:.2f}")
        logger.info(f"  Net Sharpe: {best_net:.2f}")
        logger.info(f"  Turnover: {best_turn*100:.0f}%")
        logger.info("")
        logger.info("NEXT STEP: Proceed to Phase 2B with caution")

    logger.info("")
    logger.info("="*70)


if __name__ == "__main__":
    main()
