"""
Main script for stock prediction pipeline.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.downloader import PolygonDownloader
from src.data.yahoo_downloader import YahooDownloader
from src.data.preprocessor import StockPreprocessor
from src.database.qdrant_client import StockQdrantClient
from src.database.embeddings import EmbeddingGenerator
from src.models.lstm_model import create_model, count_parameters
from src.models.dataset import create_dataloaders
from src.models.trainer import ModelTrainer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.utils.gpu_check import check_gpu
from src.utils.metrics import calculate_all_metrics, print_metrics


def download_data(config: ConfigLoader, logger):
    """Download stock data from configured source (Yahoo Finance or Polygon.io)."""
    logger.info("Starting data download...")

    # Get data source from config
    data_source = config.get('data.source', 'polygon')

    # Select appropriate downloader
    if data_source == 'yahoo':
        logger.info("Using Yahoo Finance downloader...")
        downloader = YahooDownloader(config)
    else:
        logger.info("Using Polygon.io downloader...")
        downloader = PolygonDownloader(config)

    tickers = config.get('data.tickers')
    start_date = config.get('data.date_ranges.download_start')
    end_date = config.get('data.date_ranges.download_end')
    output_file = config.get('data.paths.raw_data')

    df = downloader.download_multiple_stocks(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        output_file=output_file
    )

    logger.info(f"Downloaded {len(df)} total records")
    logger.info(f"Tickers: {df['ticker'].unique().tolist()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")


def preprocess_data(config: ConfigLoader, logger):
    """Preprocess data and engineer features."""
    logger.info("Starting data preprocessing...")

    preprocessor = StockPreprocessor(config)

    input_file = config.get('data.paths.raw_data')
    output_dir = "data/processed"

    output_files = preprocessor.preprocess_pipeline(
        input_file=input_file,
        output_dir=output_dir
    )

    logger.info("Preprocessing completed")
    logger.info(f"Output files: {output_files}")


def upload_to_qdrant(config: ConfigLoader, logger):
    """Generate embeddings and upload to Qdrant."""
    logger.info("Uploading data to Qdrant...")

    # Initialize clients
    qdrant_client = StockQdrantClient(config)
    embedding_gen = EmbeddingGenerator(config)

    # Create collection
    qdrant_client.create_collection(recreate=True)

    # Load processed data
    train_file = config.get('data.paths.processed_train', 'data/processed/train.parquet')
    val_file = config.get('data.paths.processed_val', 'data/processed/validation.parquet')
    test_file = config.get('data.paths.processed_test', 'data/processed/test.parquet')

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    test_df = pd.read_parquet(test_file)

    # Get feature columns from preprocessor
    preprocessor = StockPreprocessor(config)
    feature_columns = preprocessor.get_feature_columns(train_df)

    # Generate and upload embeddings for each dataset
    for df, dataset_name in [(train_df, 'train'), (val_df, 'validation'), (test_df, 'test')]:
        logger.info(f"Processing {dataset_name} set...")

        embeddings, metadata = embedding_gen.generate_embeddings(
            df=df,
            feature_columns=feature_columns,
            method="simple"
        )

        metadata['dataset'] = dataset_name

        qdrant_client.upsert_embeddings(embeddings, metadata)

    # Print collection info
    info = qdrant_client.get_collection_info()
    logger.info(f"Qdrant collection info: {info}")


def train_model(config: ConfigLoader, logger, model_type: str = "regression"):
    """Train LSTM model."""
    logger.info(f"Training {model_type} model...")

    # Load processed data
    train_file = "data/processed/train.parquet"
    val_file = "data/processed/validation.parquet"

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)

    # Create preprocessor to get sequences
    preprocessor = StockPreprocessor(config)
    preprocessor.feature_columns = preprocessor.get_feature_columns(train_df)

    sequence_length = config.get('data.sequence_length', 60)

    # Create sequences
    X_train, y_train_reg, y_train_clf, _ = preprocessor.create_sequences(train_df, sequence_length)
    X_val, y_val_reg, y_val_clf, _ = preprocessor.create_sequences(val_df, sequence_length)

    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Validation sequences: {X_val.shape}")

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
        num_workers=4,  # Use 4 parallel workers for data loading
        pin_memory=True
    )

    # Create model
    input_size = X_train.shape[2]  # number of features
    model_config = config.get('model.architecture')

    # Detect number of classes for classification models
    num_classes = 3  # Default
    if model_type in ["classification", "multitask"]:
        num_classes = len(np.unique(y_train_clf))
        logger.info(f"Detected {num_classes} classes in training data")

    model = create_model(
        model_type=model_type,
        input_size=input_size,
        config=model_config,
        num_classes=num_classes
    )

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")

    # Calculate class weights for classification/multitask models (handle class imbalance)
    class_weights = None
    if model_type in ["classification", "multitask"]:
        # Calculate class distribution in training data
        unique, counts = np.unique(y_train_clf, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        logger.info(f"Training class distribution: {class_distribution}")

        # Calculate inverse frequency weights
        total_samples = len(y_train_clf)
        class_weights = torch.FloatTensor([
            total_samples / (len(unique) * counts[i])
            for i in range(len(unique))
        ])
        logger.info(f"Calculated class weights: {class_weights}")

    # Create trainer
    trainer = ModelTrainer(model, config, model_type=model_type, class_weights=class_weights)

    # Train
    epochs = config.get('model.training.epochs', 100)
    model_name = f"lstm_{model_type}"

    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=epochs,
        model_name=model_name
    )

    logger.info("Training completed")


def evaluate_model(config: ConfigLoader, logger, model_type: str = "regression"):
    """Evaluate trained model on test set."""
    logger.info(f"Evaluating {model_type} model...")

    # Load test data
    test_file = "data/processed/test.parquet"
    test_df = pd.read_parquet(test_file)

    # Create sequences
    preprocessor = StockPreprocessor(config)
    preprocessor.feature_columns = preprocessor.get_feature_columns(test_df)

    sequence_length = config.get('data.sequence_length', 60)
    X_test, y_test_reg, y_test_clf, metadata = preprocessor.create_sequences(test_df, sequence_length)

    logger.info(f"Test sequences: {X_test.shape}")

    # Load model
    input_size = X_test.shape[2]
    model_config = config.get('model.architecture')

    # Detect number of classes for classification models
    num_classes = 3  # Default
    if model_type in ["classification", "multitask"]:
        num_classes = len(np.unique(y_test_clf))
        logger.info(f"Detected {num_classes} classes in test data")

    model = create_model(
        model_type=model_type,
        input_size=input_size,
        config=model_config,
        num_classes=num_classes
    )

    # Load checkpoint
    checkpoint_path = Path(f"models/checkpoints/lstm_{model_type}_best.pth")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    import torch
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Make predictions
    import torch
    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        if model_type == "multitask":
            reg_predictions, clf_logits = model(X_tensor)
            reg_predictions = reg_predictions.cpu().numpy().squeeze()
            clf_predictions = clf_logits.argmax(axis=1).cpu().numpy()
        else:
            predictions = model(X_tensor).cpu().numpy()

    # Calculate metrics
    if model_type == "regression":
        predictions = predictions.squeeze()
        metrics = calculate_all_metrics(
            y_true=y_test_reg,
            y_pred=predictions,
            model_type="regression"
        )
        # Print metrics
        print_metrics(metrics, model_type)

        # Save predictions
        results_df = metadata.copy()
        results_df['y_true'] = y_test_reg
        results_df['y_pred'] = predictions
        results_file = f"data/processed/{model_type}_predictions.parquet"

    elif model_type == "classification":
        predictions = predictions.argmax(axis=1)
        metrics = calculate_all_metrics(
            y_true=y_test_clf,
            y_pred=predictions,
            model_type="classification"
        )
        # Print metrics
        print_metrics(metrics, model_type)

        # Save predictions
        results_df = metadata.copy()
        results_df['y_true'] = y_test_clf
        results_df['y_pred'] = predictions
        results_file = f"data/processed/{model_type}_predictions.parquet"

    else:  # multitask
        # Calculate metrics for both tasks
        reg_metrics = calculate_all_metrics(
            y_true=y_test_reg,
            y_pred=reg_predictions,
            model_type="regression"
        )
        clf_metrics = calculate_all_metrics(
            y_true=y_test_clf,
            y_pred=clf_predictions,
            model_type="classification"
        )

        # Print both sets of metrics
        logger.info("\n=== REGRESSION METRICS ===")
        print_metrics(reg_metrics, "regression")
        logger.info("\n=== CLASSIFICATION METRICS ===")
        print_metrics(clf_metrics, "classification")

        # Save both predictions
        results_df = metadata.copy()
        results_df['y_true_reg'] = y_test_reg
        results_df['y_pred_reg'] = reg_predictions
        results_df['y_true_clf'] = y_test_clf
        results_df['y_pred_clf'] = clf_predictions
        results_file = f"data/processed/{model_type}_predictions.parquet"

    results_df.to_parquet(results_file)
    logger.info(f"Saved predictions to {results_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stock Prediction Pipeline")

    parser.add_argument(
        'command',
        choices=['download', 'preprocess', 'upload', 'train-reg', 'train-clf', 'train-multitask', 'eval-reg', 'eval-clf', 'eval-multitask', 'check-gpu', 'all'],
        help='Command to run'
    )

    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    config = ConfigLoader(args.config)

    # Setup logger
    logger = setup_logger(
        name='main',
        log_file=config.get('logging.file', 'logs/stock_data.log'),
        level=config.get('logging.level', 'INFO')
    )

    logger.info(f"Running command: {args.command}")

    # Execute command
    if args.command == 'check-gpu':
        check_gpu()

    elif args.command == 'download':
        download_data(config, logger)

    elif args.command == 'preprocess':
        preprocess_data(config, logger)

    elif args.command == 'upload':
        upload_to_qdrant(config, logger)

    elif args.command == 'train-reg':
        logger.info("\n" + "="*70)
        logger.info("NOTE: Training regression model only.")
        logger.info("RECOMMENDED: Use 'train-multitask' for better performance")
        logger.info("Multi-task model learns both regression + classification together")
        logger.info("="*70 + "\n")
        train_model(config, logger, model_type="regression")

    elif args.command == 'train-clf':
        logger.info("\n" + "="*70)
        logger.info("NOTE: Training classification model only.")
        logger.info("RECOMMENDED: Use 'train-multitask' for better performance")
        logger.info("Multi-task model learns both regression + classification together")
        logger.info("Benefits: Shared learning, automatic class weighting, single model")
        logger.info("="*70 + "\n")
        train_model(config, logger, model_type="classification")

    elif args.command == 'train-multitask':
        logger.info("\n" + "="*70)
        logger.info("Training multi-task model (RECOMMENDED)")
        logger.info("Learns both regression + classification with shared LSTM encoder")
        logger.info("Includes automatic class weighting for imbalanced data")
        logger.info("="*70 + "\n")
        train_model(config, logger, model_type="multitask")

    elif args.command == 'eval-reg':
        evaluate_model(config, logger, model_type="regression")

    elif args.command == 'eval-clf':
        evaluate_model(config, logger, model_type="classification")

    elif args.command == 'eval-multitask':
        evaluate_model(config, logger, model_type="multitask")

    elif args.command == 'all':
        logger.info("Running full pipeline with multi-task model...")
        download_data(config, logger)
        preprocess_data(config, logger)
        upload_to_qdrant(config, logger)
        train_model(config, logger, model_type="multitask")
        evaluate_model(config, logger, model_type="multitask")
        logger.info("Full pipeline completed!")

    logger.info("Done!")


if __name__ == "__main__":
    main()
