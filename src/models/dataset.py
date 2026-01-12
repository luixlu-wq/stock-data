"""
PyTorch Dataset and DataLoader utilities for stock data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""

    def __init__(
        self,
        X: np.ndarray,
        y_reg: np.ndarray = None,
        y_clf: np.ndarray = None,
        mode: str = "train"
    ):
        """
        Initialize dataset.

        Args:
            X: Feature sequences (N, sequence_length, num_features)
            y_reg: Regression targets (N,) - optional
            y_clf: Classification targets (N,) - optional
            mode: "train", "val", or "test"
        """
        self.X = torch.FloatTensor(X)
        self.y_reg = torch.FloatTensor(y_reg) if y_reg is not None else None
        self.y_clf = torch.LongTensor(y_clf) if y_clf is not None else None
        self.mode = mode

    def __len__(self):
        """Get dataset size."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (features, targets) or just features for test mode
        """
        x = self.X[idx]

        if self.mode == "test":
            return x

        # Return features and targets
        return_items = [x]

        if self.y_reg is not None:
            return_items.append(self.y_reg[idx])

        if self.y_clf is not None:
            return_items.append(self.y_clf[idx])

        return tuple(return_items) if len(return_items) > 1 else return_items[0]


def create_dataloaders(
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    y_train_clf: np.ndarray,
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    y_val_clf: np.ndarray,
    X_test: np.ndarray = None,
    y_test_reg: np.ndarray = None,
    y_test_clf: np.ndarray = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> dict:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        X_train: Training features
        y_train_reg: Training regression targets
        y_train_clf: Training classification targets
        X_val: Validation features
        y_val_reg: Validation regression targets
        y_val_clf: Validation classification targets
        X_test: Test features (optional)
        y_test_reg: Test regression targets (optional)
        y_test_clf: Test classification targets (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Dictionary with dataloaders
    """
    # Create datasets
    train_dataset = StockDataset(X_train, y_train_reg, y_train_clf, mode="train")
    val_dataset = StockDataset(X_val, y_val_reg, y_val_clf, mode="val")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Create test dataloader if test data provided
    if X_test is not None:
        test_dataset = StockDataset(X_test, y_test_reg, y_test_clf, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        dataloaders['test'] = test_loader

    return dataloaders
