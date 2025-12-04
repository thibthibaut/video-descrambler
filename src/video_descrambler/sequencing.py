from abc import ABC, abstractmethod
from typing import List
import logging

import numpy as np
import fast_tsp

logger = logging.getLogger(__name__)


class Sequencer(ABC):
    """Find optimal frame ordering from a distance matrix."""

    @abstractmethod
    def order_frames(self, distance_matrix: np.ndarray) -> List[int]:
        """Returns ordered list of frame indices."""
        pass


class TSPSequencer(Sequencer):
    """Use Traveling Salesman Problem solver to find optimal frame ordering."""

    def order_frames(self, distance_matrix: np.ndarray) -> List[int]:
        N = distance_matrix.shape[0]
        logger.info(f"Solving TSP for {N} frames")

        # Convert to integer matrix for fast_tsp
        dist_int = distance_matrix.copy()
        dist_int[dist_int == np.inf] = -1
        max_distance = dist_int.max()
        max_u16 = 2**16 - 1
        scaling_factor = max_u16 / max_distance
        dist_int[dist_int == -1] = max_u16
        dist_int = (dist_int * scaling_factor).astype(np.int64)

        # Solve TSP
        tour = fast_tsp.find_tour(dist_int)

        # Convert loop to sequence
        tour = self._find_begin_and_end(tour, distance_matrix)

        return tour

    def _find_begin_and_end(
        self, tour: List[int], distance_matrix: np.ndarray
    ) -> List[int]:
        """
        Convert circular tour to linear sequence by breaking at largest distance gap.

        Args:
            tour: Circular tour of frame indices
            distance_matrix: Distance matrix between frames

        Returns:
            Linear sequence with proper begin and end points
        """
        logger.info("Converting circular tour to linear sequence")

        if len(tour) == 0:
            return tour

        # Find the largest gap between consecutive frames in the tour
        max_gap = -np.inf
        max_gap_index = 0

        for i in range(len(tour)):
            # Wrap around for circular sequences
            next_idx = (i + 1) % len(tour)
            dist = distance_matrix[tour[i], tour[next_idx]]

            if dist > max_gap:
                max_gap = dist
                max_gap_index = i

        logger.debug(
            f"Largest gap found at index {max_gap_index} with distance {max_gap:.2f}"
        )

        # Split the tour at the largest gap to create a linear sequence
        # Start after the gap and wrap around
        if max_gap_index != 0 or max_gap != 0:
            linear_sequence = tour[max_gap_index + 1 :] + tour[: max_gap_index + 1]
        else:
            linear_sequence = tour

        logger.info(f"Linear sequence created with {len(linear_sequence)} frames")

        return linear_sequence
