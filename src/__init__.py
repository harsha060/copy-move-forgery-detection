"""Copy-Move Image Forgery Detection Package.

This package provides tools for digital image processing-based forgery detection,
including preprocessing, keypoint-based matching (SIFT/ORB), geometric verification (RANSAC),
and forensic visualization.
"""

from src.preprocessing import PreprocessedImage, preprocess_image, to_grayscale
from src.detector import CopyMoveDetector, DetectionResult
from src.visualizer import (
    draw_keypoints,
    draw_matches,
    create_binary_mask,
    create_overlay,
)

__all__ = [
    "PreprocessedImage",
    "preprocess_image",
    "to_grayscale",
    "CopyMoveDetector",
    "DetectionResult",
    "draw_keypoints",
    "draw_matches",
    "create_binary_mask",
    "create_overlay",
]
