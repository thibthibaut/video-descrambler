from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """Extract features from a single frame."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Returns features as a numpy array."""
        pass


class RawPixelExtractor(FeatureExtractor):
    """Uses raw pixel values as features."""

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Simply flatten the image
        return image.ravel()
