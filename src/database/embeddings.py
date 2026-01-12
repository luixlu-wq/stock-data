"""
Generate embeddings from stock features for vector storage.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


class AutoEncoder(nn.Module):
    """Simple autoencoder for dimensionality reduction."""

    def __init__(self, input_dim: int, embedding_dim: int):
        """
        Initialize autoencoder.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Embedding dimension (compressed)
        """
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        """Encode input to embedding."""
        return self.encoder(x)


class EmbeddingGenerator:
    """Generate embeddings from stock features."""

    def __init__(self, config: ConfigLoader):
        """
        Initialize embedding generator.

        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.logger = get_logger(__name__)

        self.embedding_dim = config.get('qdrant.vector_size', 64)
        self.autoencoder: Optional[AutoEncoder] = None

    def simple_embedding(self, features: np.ndarray) -> np.ndarray:
        """
        Create simple embeddings by taking last time step features.

        This is the simplest approach - just use the latest features
        as the embedding (optionally with PCA reduction).

        Args:
            features: Feature array (N, sequence_length, num_features)
                     or (N, num_features)

        Returns:
            Embeddings array (N, embedding_dim)
        """
        # If 3D (sequences), take last time step
        if len(features.shape) == 3:
            features = features[:, -1, :]  # (N, num_features)

        # If features already match embedding dim, return as-is
        if features.shape[1] == self.embedding_dim:
            return features

        # Otherwise, use PCA-like reduction (simple averaging with window)
        num_features = features.shape[1]

        if num_features > self.embedding_dim:
            # Downsample by averaging
            window_size = num_features // self.embedding_dim
            embeddings = []

            for i in range(self.embedding_dim):
                start_idx = i * window_size
                end_idx = min(start_idx + window_size, num_features)
                embeddings.append(features[:, start_idx:end_idx].mean(axis=1, keepdims=True))

            return np.concatenate(embeddings, axis=1)

        else:
            # Pad if too small
            padding = np.zeros((features.shape[0], self.embedding_dim - num_features))
            return np.concatenate([features, padding], axis=1)

    def train_autoencoder(
        self,
        features: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001
    ) -> None:
        """
        Train autoencoder for feature compression.

        Args:
            features: Training features (N, num_features)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        self.logger.info("Training autoencoder for embedding generation...")

        # Take last time step if sequences
        if len(features.shape) == 3:
            features = features[:, -1, :]

        input_dim = features.shape[1]

        # Initialize autoencoder
        self.autoencoder = AutoEncoder(input_dim, self.embedding_dim)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Convert to torch tensors
        features_tensor = torch.FloatTensor(features)

        # Training loop
        self.autoencoder.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(features_tensor), batch_size):
                batch = features_tensor[i:i + batch_size]

                # Forward pass
                encoded, decoded = self.autoencoder(batch)
                loss = criterion(decoded, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        self.logger.info("Autoencoder training completed")

    def autoencoder_embedding(self, features: np.ndarray) -> np.ndarray:
        """
        Generate embeddings using trained autoencoder.

        Args:
            features: Feature array (N, sequence_length, num_features)
                     or (N, num_features)

        Returns:
            Embeddings array (N, embedding_dim)
        """
        if self.autoencoder is None:
            raise ValueError("Autoencoder not trained. Call train_autoencoder first.")

        # Take last time step if sequences
        if len(features.shape) == 3:
            features = features[:, -1, :]

        # Convert to tensor
        features_tensor = torch.FloatTensor(features)

        # Generate embeddings
        self.autoencoder.eval()
        with torch.no_grad():
            embeddings = self.autoencoder.encode(features_tensor)

        return embeddings.numpy()

    def generate_embeddings(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        method: str = "simple"
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate embeddings from processed data.

        Args:
            df: Processed DataFrame with features
            feature_columns: List of feature column names
            method: Embedding method ("simple" or "autoencoder")

        Returns:
            Tuple of (embeddings array, metadata DataFrame)
        """
        self.logger.info(f"Generating embeddings using {method} method...")

        # Extract features
        features = df[feature_columns].values  # (N, num_features)

        # Generate embeddings
        if method == "simple":
            embeddings = self.simple_embedding(features)

        elif method == "autoencoder":
            if self.autoencoder is None:
                self.logger.info("Training autoencoder on data...")
                self.train_autoencoder(features)

            embeddings = self.autoencoder_embedding(features)

        else:
            raise ValueError(f"Unknown embedding method: {method}")

        # Extract metadata
        metadata = df[['ticker', 'date', 'close']].copy()

        if 'target_class' in df.columns:
            metadata['target_class'] = df['target_class']

        self.logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

        return embeddings, metadata
