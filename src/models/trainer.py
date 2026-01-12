"""
Model training pipeline with GPU support.
"""
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.models.losses import DirectionalHuberLoss, WeightedDirectionalLoss, CombinedRankRegressionLoss


class ModelTrainer:
    """Trainer for LSTM models with GPU support."""

    def __init__(
        self,
        model: nn.Module,
        config: ConfigLoader,
        model_type: str = "regression",
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            config: Configuration loader
            model_type: "regression" or "classification"
            class_weights: Optional class weights for classification (for imbalanced classes)
        """
        self.model = model
        self.config = config
        self.model_type = model_type
        self.class_weights = class_weights
        self.logger = get_logger(__name__)

        # Device configuration
        device_config = config.get('model.device', 'cuda')

        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)

        self.logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)

        # Loss function
        if model_type == "regression":
            loss_type = config.get('model.loss.regression', 'directional_huber')

            if loss_type == 'mse':
                self.criterion = nn.MSELoss()
            elif loss_type == 'mae':
                self.criterion = nn.L1Loss()
            elif loss_type == 'huber':
                # CRITICAL FIX: Delta must match target scale!
                # For 1-day returns (std~0.02), use delta=0.05 instead of 1.0
                delta = config.get('model.loss.huber_delta', 0.05)
                self.criterion = nn.HuberLoss(delta=delta)
                self.logger.info(f"Using HuberLoss with delta={delta}")
            elif loss_type == 'directional_huber':
                # NEW: Directional Huber - encourages correct direction prediction
                directional_weight = config.get('model.loss.directional_weight', 0.3)
                self.criterion = DirectionalHuberLoss(delta=1.0, directional_weight=directional_weight)
                self.logger.info(f"Using DirectionalHuberLoss with directional_weight={directional_weight}")
            elif loss_type == 'weighted_directional':
                # NEW: Weighted directional - penalizes wrong direction heavily
                directional_weight = config.get('model.loss.directional_weight', 0.3)
                magnitude_scale = config.get('model.loss.magnitude_scale', 1.0)
                self.criterion = WeightedDirectionalLoss(
                    base_loss='huber',
                    directional_weight=directional_weight,
                    magnitude_scale=magnitude_scale
                )
                self.logger.info(f"Using WeightedDirectionalLoss with dir_weight={directional_weight}, mag_scale={magnitude_scale}")
            elif loss_type == 'rank':
                # PHASE 1: Cross-sectional rank loss (THE KEY INNOVATION)
                rank_weight = config.get('model.loss.rank_weight', 0.7)
                temperature = config.get('model.loss.rank_temperature', 1.0)
                huber_delta = config.get('model.loss.huber_delta', 0.05)
                self.criterion = CombinedRankRegressionLoss(
                    rank_weight=rank_weight,
                    huber_delta=huber_delta,
                    temperature=temperature
                )
                self.logger.info(f"PHASE 1: Using CombinedRankRegressionLoss")
                self.logger.info(f"  Rank weight: {rank_weight} (remaining: {1-rank_weight} for Huber)")
                self.logger.info(f"  Temperature: {temperature}")
                self.logger.info(f"  This optimizes RANKING QUALITY, not just return accuracy")
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

        elif model_type == "classification":
            # Use class weights if provided (for imbalanced classes)
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                self.logger.info(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
            else:
                self.criterion = nn.CrossEntropyLoss()

        elif model_type == "multitask":
            # Multi-task learning uses both losses
            reg_loss_type = config.get('model.loss.regression', 'huber')
            if reg_loss_type == 'huber':
                self.criterion_reg = nn.HuberLoss(delta=1.0)
            elif reg_loss_type == 'mse':
                self.criterion_reg = nn.MSELoss()
            else:
                self.criterion_reg = nn.L1Loss()

            # Use class weights for classification task if provided
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                self.criterion_clf = nn.CrossEntropyLoss(weight=class_weights)
                self.logger.info(f"Using weighted CrossEntropyLoss with weights: {class_weights}")
            else:
                self.criterion_clf = nn.CrossEntropyLoss()

            # Task weights for multi-task learning
            self.reg_weight = config.get('model.training.reg_weight', 1.0)
            self.clf_weight = config.get('model.training.clf_weight', 0.5)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Gradient clipping
        self.max_grad_norm = config.get('model.training.max_grad_norm', 1.0)

        # Optimizer
        self.base_lr = config.get('model.training.learning_rate', 0.001)
        weight_decay = config.get('model.training.weight_decay', 0.0001)

        # Learning rate warm-up configuration
        warmup_config = config.get('model.training.warmup', {})
        self.warmup_enabled = warmup_config.get('enabled', False)
        self.warmup_epochs = warmup_config.get('epochs', 5)
        self.warmup_start_lr = warmup_config.get('start_lr', self.base_lr * 0.1)
        self.current_epoch = 0

        # Start with warmup LR if enabled, otherwise base LR
        initial_lr = self.warmup_start_lr if self.warmup_enabled else self.base_lr

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        scheduler_config = config.get('model.training.scheduler', {})
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')

        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config.get('patience', 5),
                factor=scheduler_config.get('factor', 0.5),
                min_lr=scheduler_config.get('min_lr', 0.00001)
            )
        else:
            self.scheduler = None

        # Early stopping
        early_stop_config = config.get('model.training.early_stopping', {})
        self.early_stop_patience = early_stop_config.get('patience', 15)
        self.early_stop_min_delta = early_stop_config.get('min_delta', 0.0001)

        # Checkpointing
        checkpoint_config = config.get('model.checkpoint', {})
        self.checkpoint_dir = Path(checkpoint_config.get('save_dir', 'models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = checkpoint_config.get('save_best_only', True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader

        Returns:
            Tuple of (average loss, average metric)
        """
        self.model.train()
        total_loss = 0
        total_metric = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc="Training")

        for batch in progress_bar:
            # Get batch data
            if self.model_type == "regression":
                X, y, _ = batch  # Get X and regression target, skip classification
                y = y.unsqueeze(1)  # (batch_size, 1)
            else:  # classification
                X, _, y = batch  # Skip regression target, get classification

            # Move to device
            X = X.to(self.device)
            y = y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.model_type == "multitask":
                # Multi-task model returns both outputs
                reg_out, clf_out = self.model(X)

                # Get both targets
                X_batch, y_reg, y_clf = batch
                y_reg = y_reg.unsqueeze(1).to(self.device)
                y_clf = y_clf.to(self.device)

                # Calculate combined loss
                reg_loss = self.criterion_reg(reg_out, y_reg)
                clf_loss = self.criterion_clf(clf_out, y_clf)
                loss = self.reg_weight * reg_loss + self.clf_weight * clf_loss
            else:
                outputs = self.model(X)
                loss = self.criterion(outputs, y)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                if self.model_type == "multitask":
                    # Multi-task metrics
                    reg_metric = torch.sqrt(reg_loss)
                    _, predicted = torch.max(clf_out, 1)
                    clf_metric = (predicted == y_clf).float().mean()
                    metric = (reg_metric + clf_metric) / 2
                elif self.model_type == "regression":
                    # RMSE
                    metric = torch.sqrt(loss)
                else:
                    # Accuracy
                    _, predicted = torch.max(outputs, 1)
                    metric = (predicted == y).float().mean()

            total_loss += loss.item()
            total_metric += metric.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'metric': f'{metric.item():.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches

        return avg_loss, avg_metric

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.

        Args:
            dataloader: Validation dataloader

        Returns:
            Tuple of (average loss, average metric)
        """
        self.model.eval()
        total_loss = 0
        total_metric = 0
        num_batches = 0

        # Collect predictions for model collapse detection
        all_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                # Get batch data
                if self.model_type == "regression":
                    X, y, _ = batch  # Get X and regression target, skip classification
                    y = y.unsqueeze(1)
                elif self.model_type == "classification":
                    X, _, y = batch  # Skip regression target, get classification
                else:  # multitask
                    X, y_reg, y_clf = batch
                    y_reg = y_reg.unsqueeze(1)

                # Move to device
                X = X.to(self.device)

                if self.model_type == "multitask":
                    y_reg = y_reg.to(self.device)
                    y_clf = y_clf.to(self.device)

                    # Forward pass
                    reg_out, clf_out = self.model(X)

                    # Calculate combined loss
                    reg_loss = self.criterion_reg(reg_out, y_reg)
                    clf_loss = self.criterion_clf(clf_out, y_clf)
                    loss = self.reg_weight * reg_loss + self.clf_weight * clf_loss

                    # Metrics: use regression RMSE and classification accuracy
                    reg_metric = torch.sqrt(reg_loss)
                    _, predicted = torch.max(clf_out, 1)
                    clf_metric = (predicted == y_clf).float().mean()
                    metric = (reg_metric + clf_metric) / 2  # Average both metrics
                else:
                    y = y.to(self.device)

                    # Forward pass
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)

                    # Calculate metrics
                    if self.model_type == "regression":
                        metric = torch.sqrt(loss)
                    else:
                        _, predicted = torch.max(outputs, 1)
                        metric = (predicted == y).float().mean()

                total_loss += loss.item()
                total_metric += metric.item()
                num_batches += 1

                # Collect predictions for variance check
                if self.model_type == "multitask":
                    all_predictions.extend(reg_out.cpu().numpy().flatten()[:100])  # Sample first 100
                elif self.model_type == "regression":
                    all_predictions.extend(outputs.cpu().numpy().flatten()[:100])

        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches

        # Check for model collapse (constant predictions)
        if len(all_predictions) > 10:
            pred_std = np.std(all_predictions)
            if pred_std < 1e-6:
                self.logger.warning(f"⚠️  MODEL COLLAPSE DETECTED! Prediction std={pred_std:.8f}")
                self.logger.warning("   Model is outputting constant values - training may have failed")

        return avg_loss, avg_metric

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_name: str = "model"
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs to train
            model_name: Name for saving checkpoints

        Returns:
            Training history dictionary
        """
        # Print to both console and log
        header = "=" * 70
        msg = f"STARTING TRAINING ({self.model_type.upper()})"

        print("\n" + header)
        print(msg)
        print(header)
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Early stopping patience: {self.early_stop_patience}")
        print("")
        print("WARNING: Do NOT interrupt training!")
        print("   Let it run until completion or early stopping.")
        print("   Expected time: 10-30 minutes")
        print(header + "\n")

        self.logger.info(header)
        self.logger.info(msg)
        self.logger.info(header)
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Total epochs: {epochs}")
        self.logger.info(f"Batch size: {train_loader.batch_size}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"Early stopping patience: {self.early_stop_patience}")
        self.logger.info("WARNING: Do NOT interrupt training!")
        self.logger.info("   Let it run until completion or early stopping.")
        self.logger.info("   Expected time: 45-90 minutes")
        self.logger.info(header)

        for epoch in range(epochs):
            epoch_start = time.time()

            # Learning rate warm-up
            if self.warmup_enabled and self.current_epoch < self.warmup_epochs:
                # Linear warm-up from warmup_start_lr to base_lr
                progress = self.current_epoch / self.warmup_epochs
                current_lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                self.logger.info(f"Warm-up: epoch {self.current_epoch+1}/{self.warmup_epochs}, LR={current_lr:.6f}")

            # Train
            train_loss, train_metric = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metric = self.validate(val_loader)

            epoch_time = time.time() - epoch_start

            # Save to history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_metric)
            self.history['val_metric'].append(val_metric)

            # Log progress
            if self.model_type == "multitask":
                metric_name = "Combined"
            elif self.model_type == "regression":
                metric_name = "RMSE"
            else:
                metric_name = "Acc"

            # Calculate progress percentage
            progress = (epoch + 1) / epochs * 100

            # Format progress message
            progress_msg = (
                f"Epoch [{epoch+1:3d}/{epochs}] ({epoch_time:.1f}s) [{progress:5.1f}%] - "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"{metric_name}: {train_metric:.4f}/{val_metric:.4f} | "
                f"Best: {self.best_val_loss:.4f}"
            )

            # Print to console AND log
            print(progress_msg)
            self.logger.info(progress_msg)

            # Increment epoch counter
            self.current_epoch += 1

            # Learning rate scheduler (skip during warm-up)
            if self.scheduler and (not self.warmup_enabled or self.current_epoch >= self.warmup_epochs):
                self.scheduler.step(val_loss)

            # Save checkpoint
            if val_loss < self.best_val_loss - self.early_stop_min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                checkpoint_path = self.checkpoint_dir / f"{model_name}_best.pth"
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                self.logger.info(f"Saved best model: {checkpoint_path}")

            else:
                self.epochs_without_improvement += 1

            # Save latest checkpoint
            if not self.save_best_only:
                checkpoint_path = self.checkpoint_dir / f"{model_name}_latest.pth"
                self.save_checkpoint(checkpoint_path, epoch, val_loss)

            # Progress milestones (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                elapsed_epochs = epoch + 1
                remaining_epochs = epochs - elapsed_epochs
                milestone_msg = [
                    "",
                    f"MILESTONE: {elapsed_epochs} epochs completed",
                    f"   Best val loss: {self.best_val_loss:.4f}",
                    f"   Epochs without improvement: {self.epochs_without_improvement}/{self.early_stop_patience}"
                ]
                if remaining_epochs > 0:
                    milestone_msg.append(f"   Remaining: {remaining_epochs} epochs")
                milestone_msg.append("")

                for line in milestone_msg:
                    print(line)
                    self.logger.info(line)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stop_patience:
                stop_msg = [
                    "",
                    "=" * 70,
                    f"EARLY STOPPING triggered after {epoch+1} epochs",
                    f"   No improvement for {self.early_stop_patience} consecutive epochs",
                    f"   Best validation loss: {self.best_val_loss:.4f}",
                    "=" * 70
                ]
                for line in stop_msg:
                    print(line)
                    self.logger.info(line)
                break

        # Print completion summary
        completion_msg = [
            "",
            "=" * 70,
            "TRAINING COMPLETED SUCCESSFULLY",
            "=" * 70,
            f"Total epochs trained: {len(self.history['train_loss'])}",
            f"Best validation loss: {self.best_val_loss:.4f}",
            f"Final train loss: {self.history['train_loss'][-1]:.4f}",
            f"Final val loss: {self.history['val_loss'][-1]:.4f}",
            f"Model saved to: {self.checkpoint_dir / f'{model_name}_best.pth'}",
            "=" * 70,
            ""
        ]

        for line in completion_msg:
            print(line)
            if line:  # Don't log empty lines twice
                self.logger.info(line)

        return self.history

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_loss: Validation loss
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

        self.logger.info(f"Loaded checkpoint from {path}")
