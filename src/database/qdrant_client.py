"""
Qdrant vector database client for storing stock embeddings.
"""
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


class StockQdrantClient:
    """Client for interacting with Qdrant vector database."""

    def __init__(self, config: ConfigLoader):
        """
        Initialize Qdrant client.

        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Connection settings
        host = config.get('qdrant.host', 'localhost')
        port = config.get('qdrant.port', 6333)

        # Collection settings
        self.collection_name = config.get('qdrant.collection_name', 'stock_patterns')
        self.vector_size = config.get('qdrant.vector_size', 64)
        self.distance = self._get_distance_metric(
            config.get('qdrant.distance', 'Cosine')
        )

        # Initialize client
        try:
            self.client = QdrantClient(host=host, port=port)
            self.logger.info(f"Connected to Qdrant at {host}:{port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _get_distance_metric(self, distance_str: str) -> Distance:
        """
        Convert distance string to Qdrant Distance enum.

        Args:
            distance_str: Distance metric name

        Returns:
            Qdrant Distance enum
        """
        distance_map = {
            'Cosine': Distance.COSINE,
            'Euclidean': Distance.EUCLID,
            'Dot': Distance.DOT
        }

        return distance_map.get(distance_str, Distance.COSINE)

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create collection in Qdrant.

        Args:
            recreate: Whether to recreate collection if it exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if collection_exists:
                if recreate:
                    self.logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    self.logger.info(f"Collection {self.collection_name} already exists")
                    return

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )

            self.logger.info(
                f"Created collection: {self.collection_name} "
                f"(vector_size={self.vector_size}, distance={self.distance})"
            )

        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            raise

    def upsert_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        batch_size: int = 100
    ) -> None:
        """
        Upload embeddings with metadata to Qdrant.

        Args:
            embeddings: Array of embeddings (N, vector_size)
            metadata: DataFrame with metadata (ticker, date, close, etc.)
            batch_size: Number of points to upload per batch
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")

        self.logger.info(f"Uploading {len(embeddings)} embeddings to Qdrant...")

        points = []
        for i, (embedding, row) in enumerate(zip(embeddings, metadata.itertuples())):
            # Create unique ID
            point_id = f"{row.ticker}_{row.date.strftime('%Y-%m-%d')}"
            point_id_hash = hash(point_id) % (10 ** 12)  # Convert to numeric ID

            # Create payload
            payload = {
                'ticker': row.ticker,
                'date': row.date.isoformat(),
                'close': float(row.close)
            }

            # Add optional fields if present
            if hasattr(row, 'target_class'):
                payload['target_class'] = int(row.target_class)

            if hasattr(row, 'dataset'):
                payload['dataset'] = row.dataset

            # Create point
            point = PointStruct(
                id=point_id_hash,
                vector=embedding.tolist(),
                payload=payload
            )

            points.append(point)

            # Upload in batches
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []

        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        self.logger.info(f"Successfully uploaded {len(embeddings)} embeddings")

    def search_similar(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar patterns in Qdrant.

        Args:
            query_vector: Query embedding vector
            limit: Number of results to return
            filter_dict: Optional filters (e.g., {'ticker': 'AAPL'})

        Returns:
            List of search results with scores and metadata
        """
        try:
            # Build filter if provided
            query_filter = None
            if filter_dict:
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )

                query_filter = Filter(must=conditions)

            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                query_filter=query_filter
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'score': result.score,
                    'payload': result.payload
                })

            return formatted_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection information
        """
        try:
            info = self.client.get_collection(self.collection_name)

            # Handle both old and new Qdrant API versions
            # Newer versions use points_count, older versions used vectors_count
            points_count = getattr(info, 'points_count', None)
            vectors_count = getattr(info, 'vectors_count', points_count)

            return {
                'name': self.collection_name,
                'vectors_count': vectors_count,
                'points_count': points_count,
                'status': str(info.status) if hasattr(info, 'status') else 'unknown'
            }

        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {}

    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            raise
