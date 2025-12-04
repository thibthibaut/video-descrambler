from abc import ABC, abstractmethod
from typing import List
import logging

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

logger = logging.getLogger(__name__)


class DistanceCalculator(ABC):
    """Compute pairwise distances matrix between frame features."""

    @abstractmethod
    def compute_distance_matrix(self, features_list: List) -> np.ndarray:
        """Returns symetric NxN distance matrix."""
        pass


class PixelDistanceCalculator(DistanceCalculator):
    """Euclidean distance between pixel vectors."""

    def compute_distance_matrix(self, features_list: List[np.ndarray]) -> np.ndarray:
        logger.info(
            f"Computing pixel-based distance matrix for {len(features_list)} frames"
        )
        data = np.array(features_list)
        dist_matrix = pairwise_distances(data, metric="euclidean")
        logger.info(f"Distance matrix computed: shape {dist_matrix.shape}")
        return dist_matrix
