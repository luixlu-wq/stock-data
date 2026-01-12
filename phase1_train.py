"""
PHASE 1: Training Script with Rank Loss and Simplified Features

This script implements the Phase 1 improvements:
1. Simplified features (14 core features, no overfitted TA indicators)
2. Cross-sectional rank loss (optimizes ranking, not just magnitude)
3. Focus on tradeable alpha (stable predictions, low turnover)

Expected improvements over baseline:
- Turnover: 120% -> 20-40% (3-6x reduction)
- Net Sharpe: -1.74 -> 0.3-0.8 (becomes profitable)
- Ranking stability: Much higher
"""
import sys
from pathlib import Path

import os
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor_v2 import SimplifiedStockPreprocessor
from src.models.lstm_model import create_model, count_parameters
from src.models.dataset import create_dataloaders
from src.models.trainer import ModelTrainer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


def main():
    """Run Phase 1 training pipeline."""
    logger = setup_logger('phase1', log_file='logs/phase1_training.log')

    logger.info("="*70)
    logger.info("PHASE 1: RANK LOSS + SIMPLIFIED FEATURES")
    logger.info("="*70)
    logger.info("This training run implements:")
    logger.info("  1. Only 14 core features (vs 40+ before)")
    logger.info("  2. Cross-sectional rank loss (70% weight)")
    logger.info("  3. Huber regression loss (30% weight)")
    logger.info("")
    logger.info("Goal: Reduce turnover from 120% to <40%")
    logger.info("Goal: Increase net Sharpe from -1.74 to >0.3")
    logger.info("="*70)
    logger.info("")

    # Load config
    config = ConfigLoader('config/config.yaml')

    # STEP 1: Preprocess with simplified features
    logger.info("STEP 1: Preprocessing with 14 core features...")

    preprocessor = SimplifiedStockPreprocessor(config)

    input_file = config.get('data.paths.raw_data')
    output_dir = "data/processed"

    output_files = preprocessor.preprocess_pipeline(
        input_file=input_file,
        output_dir=output_dir
    )

    logger.info(f"Preprocessed data saved:")
    for key, path in output_files.items():
        logger.info(f"  {key}: {path}")

    # STEP 2: Load preprocessed data
    logger.info("")
    logger.info("STEP 2: Loading preprocessed data...")

    train_df = pd.read_parquet(output_files['train'])
    val_df = pd.read_parquet(output_files['validation'])

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Val shape: {val_df.shape}")

    # STEP 3: Create sequences
    logger.info("")
    logger.info("STEP 3: Creating sequences...")

    sequence_length = config.get('data.sequence_length', 60)

    # Ensure preprocessor has feature columns set
    preprocessor.feature_columns = preprocessor.get_feature_columns(train_df)

    X_train, y_train_reg, y_train_clf, _ = preprocessor.create_sequences(train_df, sequence_length)
    X_val, y_val_reg, y_val_clf, _ = preprocessor.create_sequences(val_df, sequence_length)

    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Validation sequences: {X_val.shape}")
    logger.info(f"Features per timestep: {X_train.shape[2]}")

    # STEP 4: Create dataloaders
    logger.info("")
    logger.info("STEP 4: Creating dataloaders...")

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

    # STEP 5: Create model
    logger.info("")
    logger.info("STEP 5: Creating model...")

    input_size = X_train.shape[2]
    model_config = config.get('model.architecture')

    # Use regression model (simpler than multitask for Phase 1)
    model = create_model(
        model_type="regression",
        input_size=input_size,
        config=model_config,
        num_classes=3
    )

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Model type: Regression LSTM")

    # STEP 6: Create trainer with rank loss
    logger.info("")
    logger.info("STEP 6: Setting up trainer...")
    logger.info("")
    logger.info("CRITICAL: Configuring cross-sectional rank loss")
    logger.info("  This is the KEY innovation for Phase 1")
    logger.info("  Loss = 0.7 * RankLoss + 0.3 * HuberLoss")
    logger.info("  Optimizes: Ranking accuracy (what drives trading P&L)")
    logger.info("  Not: Return magnitude accuracy (what MSE optimizes)")
    logger.info("")

    # Override config to use rank loss
    if 'model' not in config._config:
        config._config['model'] = {}
    if 'loss' not in config._config['model']:
        config._config['model']['loss'] = {}
    config._config['model']['loss']['regression'] = 'rank'
    config._config['model']['loss']['rank_weight'] = 0.7
    config._config['model']['loss']['rank_temperature'] = 1.0
    config._config['model']['loss']['huber_delta'] = 0.05

    trainer = ModelTrainer(
        model=model,
        config=config,
        model_type="regression",
        class_weights=None
    )

    # STEP 7: Train
    logger.info("")
    logger.info("STEP 7: Training model...")
    logger.info("="*70)

    epochs = config.get('model.training.epochs', 100)
    model_name = "lstm_phase1_rank"

    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=epochs,
        model_name=model_name
    )

    logger.info("")
    logger.info("="*70)
    logger.info("PHASE 1 TRAINING COMPLETED")
    logger.info("="*70)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Model saved: models/checkpoints/{model_name}_best.pth")
    logger.info("")
    logger.info("NEXT STEP: Run evaluation to generate predictions")
    logger.info(f"  python phase1_evaluate.py")
    logger.info("="*70)


if __name__ == "__main__":
    main()
