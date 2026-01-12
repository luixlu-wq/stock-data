"""
Resume interrupted training from latest checkpoint.
"""
import argparse
from pathlib import Path
import torch

from src.data.preprocessor import StockPreprocessor
from src.models.lstm_model import create_model
from src.models.dataset import create_dataloaders
from src.models.trainer import ModelTrainer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Resume interrupted training')
    parser.add_argument(
        '--model-type',
        choices=['regression', 'classification', 'multitask'],
        default='multitask',
        help='Model type to resume training'
    )
    parser.add_argument(
        '--checkpoint',
        help='Path to checkpoint file (default: latest checkpoint)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate (default: use config value)'
    )
    parser.add_argument(
        '--reset-scheduler',
        action='store_true',
        help='Reset learning rate scheduler (useful if stuck at plateau)'
    )
    parser.add_argument(
        '--reset-early-stopping',
        action='store_true',
        help='Reset early stopping counter'
    )
    parser.add_argument(
        '--full-epochs',
        action='store_true',
        help='Train for full epoch count from config (instead of remaining epochs)'
    )

    args = parser.parse_args()

    # Load config
    config = ConfigLoader('config/config.yaml')

    # Setup logger
    logger = setup_logger(
        name='resume_training',
        log_file=config.get('logging.file', 'logs/stock_data.log'),
        level=config.get('logging.level', 'INFO')
    )

    logger.info("=" * 70)
    logger.info(f"RESUMING TRAINING ({args.model_type.upper()})")
    logger.info("=" * 70)

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Path(f"models/checkpoints/lstm_{args.model_type}_best.pth")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("\nAvailable checkpoints:")
        checkpoint_dir = Path("models/checkpoints")
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.glob("*.pth"):
                logger.info(f"  - {ckpt}")
        return

    # Load checkpoint to get epoch info
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    logger.info(f"Last completed epoch: {checkpoint.get('epoch', 0)}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Will resume from epoch: {start_epoch}")
    logger.info("")

    # Load data
    logger.info("Loading training data...")
    import pandas as pd
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/validation.parquet")

    # Create preprocessor
    preprocessor = StockPreprocessor(config)
    preprocessor.feature_columns = preprocessor.get_feature_columns(train_df)

    sequence_length = config.get('data.sequence_length', 60)

    # Create sequences
    logger.info("Creating sequences...")
    X_train, y_train_reg, y_train_clf, _ = preprocessor.create_sequences(train_df, sequence_length)
    X_val, y_val_reg, y_val_clf, _ = preprocessor.create_sequences(val_df, sequence_length)

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

    # Create model
    input_size = X_train.shape[2]
    model_config = config.get('model.architecture')

    model = create_model(
        model_type=args.model_type,
        input_size=input_size,
        config=model_config
    )

    # Calculate class weights if needed
    class_weights = None
    if args.model_type in ["classification", "multitask"]:
        import numpy as np
        unique, counts = np.unique(y_train_clf, return_counts=True)
        total_samples = len(y_train_clf)
        class_weights = torch.FloatTensor([
            total_samples / (len(unique) * counts[i])
            for i in range(len(unique))
        ])
        logger.info(f"Class weights: {class_weights}")

    # Create trainer
    trainer = ModelTrainer(model, config, model_type=args.model_type, class_weights=class_weights)

    # Load checkpoint into trainer
    logger.info(f"\nLoading model weights from checkpoint...")
    trainer.load_checkpoint(checkpoint_path)
    trainer.best_val_loss = best_val_loss

    # Override learning rate if specified
    if args.lr is not None:
        logger.info(f"🔧 Overriding learning rate: {args.lr}")
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.lr
        logger.info(f"   New LR set to: {args.lr}")

    # Reset scheduler if requested
    if args.reset_scheduler:
        logger.info("🔄 Resetting learning rate scheduler")
        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer,
            mode='min',
            patience=config.get('model.training.scheduler.patience', 5),
            factor=config.get('model.training.scheduler.factor', 0.5),
            min_lr=config.get('model.training.scheduler.min_lr', 0.00001)
        )

    # Reset early stopping if requested
    if args.reset_early_stopping:
        logger.info("🔄 Resetting early stopping counter")
        trainer.epochs_without_improvement = 0

    # Calculate remaining epochs
    total_epochs = config.get('model.training.epochs', 100)

    if args.full_epochs:
        # Train for full epoch count (ignoring checkpoint epoch)
        epochs_to_train = total_epochs
        logger.info(f"\n📊 Training status:")
        logger.info(f"   Mode: Full training (--full-epochs)")
        logger.info(f"   Checkpoint was at epoch: {start_epoch}")
        logger.info(f"   Will train for: {epochs_to_train} epochs")
        logger.info(f"   Note: Using checkpoint weights, but training full epoch count")
        logger.info("")
    else:
        # Train remaining epochs only
        remaining_epochs = max(0, total_epochs - start_epoch)

        if remaining_epochs == 0:
            logger.info("\n⚠️  Training already completed all epochs!")
            logger.info(f"   Total epochs: {total_epochs}")
            logger.info(f"   Last epoch: {checkpoint.get('epoch', 0)}")
            logger.info(f"   Tip: Use --full-epochs to train for full {total_epochs} epochs")
            return

        epochs_to_train = remaining_epochs
        logger.info(f"\n📊 Training status:")
        logger.info(f"   Completed epochs: {start_epoch}")
        logger.info(f"   Remaining epochs: {remaining_epochs}")
        logger.info(f"   Total target: {total_epochs}")
        logger.info("")

    # Resume training
    logger.info("🚀 Resuming training...\n")

    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=epochs_to_train,
        model_name=f"lstm_{args.model_type}"
    )

    logger.info("\n✅ Training resumed and completed successfully!")

if __name__ == "__main__":
    main()
