from pathlib import Path
from typing import List, Tuple
import logging
import cv2
import numpy as np
import tempfile
import shutil

from .feature_extraction import FeatureExtractor, RawPixelExtractor
from .distance_calculation import DistanceCalculator, PixelDistanceCalculator
from .sequencing import Sequencer, TSPSequencer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

logger = logging.getLogger(__name__)


class VideoDescrambler:
    """Reorder scrambled video frames into correct sequence while removing frames that are not part of the original video."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        distance_calculator: DistanceCalculator,
        sequencer: Sequencer,
    ):
        self.feature_extractor = feature_extractor
        self.distance_calculator = distance_calculator
        self.sequencer = sequencer
        # Use a temp_dir to store frames
        self.temp_dir: Path = Path(tempfile.mkdtemp(prefix="video_descrambler_"))
        # Store frames path for later reconstruction
        self.frame_paths: List[Path] = []
        self.fps: float = 30.0

    @classmethod
    def pixel_based(cls):
        """Create VideoDescrambler using simple pixel-based feature matching."""
        return cls(
            feature_extractor=RawPixelExtractor(),
            distance_calculator=PixelDistanceCalculator(),
            sequencer=TSPSequencer(),
        )

    def descramble(
        self,
        video_path: Path,
        output_path: Path = Path("output.mp4"),
    ) -> List[int]:
        """
        Complete pipeline: load video, compute ordering, write output.

        Args:
            video_path: Path to scrambled input video
            output_path: Path for descrambled output video

        Returns:
            List of frame indices in correct order
        """
        logger.info("Starting video descrambling pipeline")
        # Step 1: Extract features and save frames to temp directory
        logger.info("Extracting features")
        features = self._extract_features_from_video(video_path)

        # Step 2: Compute pairwise distance matrix
        logger.info("Computing distance matrix")
        distance_matrix = self.distance_calculator.compute_distance_matrix(features)

        # Step 3: Filter outliers
        logger.info("Filtering outliers")
        filtered_distance_matrix, valid_indices = self._filter_outliers(distance_matrix)

        # Step 4: Find optimal frame ordering
        logger.info("Finding optimal frame ordering")
        sequence_filtered = self.sequencer.order_frames(filtered_distance_matrix)

        # Map back to original frame indices
        sequence = [valid_indices[i] for i in sequence_filtered]

        # Step 5: Write output video using saved frames
        self.write_video(sequence, output_path)

        logger.info("Descrambling complete!")
        return sequence

    def _extract_features_from_video(self, video_path: Path) -> List[np.ndarray]:
        """
        Extract features from video and save frames to temporary directory.

        Args:
            video_path: Path to video file

        Returns:
            List of extracted features
        """
        logger.info(f"Extracting features from {video_path}")
        cap = cv2.VideoCapture(str(video_path))  # type: ignore

        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)

        features = []
        self.frame_paths = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame to temporary file
            frame_path = self.temp_dir / f"frame_{frame_idx:06d}.npy"
            np.save(frame_path, frame)
            self.frame_paths.append(frame_path)

            # Extract features from downsampled grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            downsample = 0.5
            gray = cv2.resize(
                gray,
                None,  # type: ignore
                fx=downsample,
                fy=downsample,  # type: ignore
                interpolation=cv2.INTER_AREA,
            )

            feat = self.feature_extractor.extract(gray)
            features.append(feat)

            frame_idx += 1

        cap.release()
        logger.info(f"Extracted features from {frame_idx} frames")

        return features

    def __del__(self):
        """Clean up temporary directory and all saved frames."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")

    def write_video(
        self,
        sequence: List[int],
        output_path: Path,
    ):
        """
        Write reordered frames to video file using saved frames.

        Args:
            sequence: Ordered list of frame indices
            output_path: Path for output video
        """
        logger.info(f"Writing output video to {output_path}")

        # Load first frame to get dimensions
        first_frame = np.load(self.frame_paths[sequence[0]])
        height, width = first_frame.shape[:2]

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))

        # Write frames in correct order
        for idx in sequence:
            frame = np.load(self.frame_paths[idx])
            out.write(frame)

        out.release()
        logger.info(f"Video successfully saved to {output_path}")

    def _filter_outliers(
        self, distance_matrix: np.ndarray, k: float = 1.5
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Remove outlier frames using IQR method.

        Args:
            distance_matrix: NxN distance matrix
            k: IQR multiplier for outlier threshold (higher = more lenient)

        Returns:
            Tuple of (filtered_distance_matrix, valid_indices)
        """
        # Compute mean distance from each frame to all others
        mean_distances = np.mean(distance_matrix, axis=1)

        # IQR-based outlier detection
        Q1 = np.percentile(mean_distances, 25)
        Q3 = np.percentile(mean_distances, 75)
        IQR = Q3 - Q1
        threshold = Q3 + k * IQR

        logger.info(
            f"IQR outlier detection: Q1={Q1:.2f}, Q3={Q3:.2f}, threshold={threshold:.2f}"
        )

        # Identify valid (non-outlier) frames
        valid_mask = mean_distances <= threshold
        valid_indices = np.where(valid_mask)[0].tolist()

        # Extract submatrix containing only valid frames
        filtered_matrix = distance_matrix[valid_mask][:, valid_mask]

        return filtered_matrix, valid_indices
