"""
PHASE 2: Enhanced Training Script

Improvements over Phase 1:
1. Sequence length: 60 → 120 days (capture longer-term patterns)
2. Features: 14 → 21 (added momentum, advanced vol, microstructure)
3. Rank loss temperature: 1.0 → 0.5 (sharper rankings, more stable)
4. Rank weight: 0.7 → 0.8 (stronger emphasis on rankings)

Expected improvements:
- Lower turnover (targeting 60-80% vs Phase 1's 95%)
- Higher gross Sharpe (better feature set)
- Net Sharpe > 0 (finally profitable!)
"""
import sys
from pathlib import Path

import os
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor_v3 import EnhancedStockPreprocessor
from src.models.lstm_model import create_model, count_parameters
from src.models.dataset import create_dataloaders
from src.models.trainer import ModelTrainer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


def main():
    """Main training function for Phase 2."""

    # Setup
    config = ConfigLoader('config/config.yaml')
    logger = setup_logger(
        name='phase2_train',
        log_file='logs/phase2_train.log',
        level='INFO'
    )

    logger.info("="*70)
    logger.info("PHASE 2: ENHANCED TRAINING")
    logger.info("="*70)
    logger.info("Improvements:")
    logger.info("  1. Sequence length: 60 → 120 days")
    logger.info("  2. Features: 14 → 21 (momentum, vol skew, microstructure)")
    logger.info("  3. Rank loss temperature: 1.0 → 0.5 (sharper rankings)")
    logger.info("  4. Rank loss weight: 0.7 → 0.8 (stronger emphasis)")
    logger.info("="*70)
    logger.info("")

    # ===== STEP 1: PREPROCESS DATA WITH V3 FEATURES =====
    logger.info("STEP 1: Preprocessing with 21 enhanced features...")

    preprocessor = EnhancedStockPreprocessor(config)

    # Check if preprocessed data already exists
    data_dir = Path("data/processed")
    train_file = data_dir / "train_v3.parquet"
    val_file = data_dir / "validation_v3.parquet"
    test_file = data_dir / "test_v3.parquet"

    if train_file.exists() and val_file.exists() and test_file.exists():
        logger.info("Found existing v3 preprocessed data")
        logger.info(f"  Train: {train_file}")
        logger.info(f"  Val: {val_file}")
        logger.info(f"  Test: {test_file}")
    else:
        logger.info("Preprocessed data not found, creating...")
        raw_data = config.get('data.paths.raw_data', 'data/raw/stock_data.parquet')
        output_files = preprocessor.preprocess_pipeline(
            input_file=raw_data,
            output_dir=str(data_dir)
        )
        logger.info("Preprocessing complete")

    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)

    logger.info(f"Train: {len(train_df)} rows")
    logger.info(f"Val: {len(val_df)} rows")

    # ===== STEP 2: CREATE SEQUENCES WITH LONGER LENGTH =====
    logger.info("")
    logger.info("STEP 2: Creating sequences with 120-day lookback...")

    # Get feature columns
    preprocessor.feature_columns = preprocessor.get_feature_columns(train_df)
    logger.info(f"Using {len(preprocessor.feature_columns)} features")

    # PHASE 2: Use 120-day sequence length (was 60 in Phase 1)
    sequence_length = 120

    X_train, y_train_reg, y_train_clf, _ = preprocessor.create_sequences(
        train_df, sequence_length
    )
    X_val, y_val_reg, y_val_clf, _ = preprocessor.create_sequences(
        val_df, sequence_length
    )

    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Validation sequences: {X_val.shape}")

    # ===== STEP 3: CREATE DATALOADERS =====
    logger.info("")
    logger.info("STEP 3: Creating dataloaders...")

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

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Train batches: {len(dataloaders['train'])}")
    logger.info(f"Val batches: {len(dataloaders['val'])}")

    # ===== STEP 4: CREATE MODEL =====
    logger.info("")
    logger.info("STEP 4: Creating LSTM model...")

    input_size = X_train.shape[2]  # 21 features
    model_config = config.get('model.architecture')

    model = create_model(
        model_type="regression",
        input_size=input_size,
        config=model_config,
        num_classes=3
    )

    num_params = count_parameters(model)
    logger.info(f"Model architecture: LSTM")
    logger.info(f"Input features: {input_size}")
    logger.info(f"Parameters: {num_params:,}")

    # ===== STEP 5: OVERRIDE CONFIG WITH PHASE 2 RANK LOSS =====
    logger.info("")
    logger.info("STEP 5: Configuring Phase 2 rank loss...")

    # PHASE 2: More aggressive rank loss settings
    config._config['model']['loss']['regression'] = 'rank'
    config._config['model']['loss']['rank_weight'] = 0.8  # Increased from 0.7
    config._config['model']['loss']['rank_temperature'] = 0.5  # Decreased from 1.0 (sharper)
    config._config['model']['loss']['huber_delta'] = 0.05

    logger.info("Rank loss configuration:")
    logger.info(f"  Rank weight: 0.8 (was 0.7 in Phase 1)")
    logger.info(f"  Temperature: 0.5 (was 1.0 in Phase 1 - SHARPER RANKINGS)")
    logger.info(f"  Huber weight: 0.2")
    logger.info(f"  Huber delta: 0.05")

    # ===== STEP 6: CREATE TRAINER =====
    logger.info("")
    logger.info("STEP 6: Creating trainer...")

    trainer = ModelTrainer(
        model=model,
        config=config,
        model_type="regression"
    )

    logger.info(f"Device: {trainer.device}")
    logger.info(f"Loss function: CombinedRankRegressionLoss (80% rank + 20% Huber)")

    # ===== STEP 7: TRAIN =====
    logger.info("")
    logger.info("="*70)
    logger.info("STEP 7: TRAINING PHASE 2 MODEL")
    logger.info("="*70)
    logger.info("")

    epochs = config.get('model.training.epochs', 100)
    model_name = "lstm_phase2_rank_best"

    logger.info(f"Training for up to {epochs} epochs with early stopping...")
    logger.info(f"Model will be saved to: models/checkpoints/{model_name}.pth")
    logger.info("")

    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=epochs,
        model_name=model_name
    )

    # ===== STEP 8: SUMMARY =====
    logger.info("")
    logger.info("="*70)
    logger.info("PHASE 2 TRAINING COMPLETED")
    logger.info("="*70)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run: python phase2_evaluate.py")
    logger.info("  2. Compare Phase 2 vs Phase 1 vs Baseline")
    logger.info("  3. Expected improvements:")
    logger.info("     - Turnover: 95% → 60-80%")
    logger.info("     - Net Sharpe: -1.04 → >0")
    logger.info("     - Finally profitable!")
    logger.info("")
    logger.info("="*70)


if __name__ == "__main__":
    main()
