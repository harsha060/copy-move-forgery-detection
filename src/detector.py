"""Core Copy-Move Forgery Detection Engine using Digital Image Processing.

Implements feature extraction (SIFT with ORB fallback), self-matching with spatial
distance filtering, ratio testing, RANSAC geometric verification, and forensic mask generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.preprocessing import PreprocessedImage, preprocess_image


def is_sift_available() -> bool:
    """Check if OpenCV has SIFT support available in the current environment."""
    return hasattr(cv2, "SIFT_create")


@dataclass
class MatchPair:
    """Represents a matched pair of coordinates between cloned and original regions."""

    pt1: Tuple[float, float]  # (x, y) coordinates of source keypoint
    pt2: Tuple[float, float]  # (x, y) coordinates of target keypoint
    distance: float           # Descriptor distance between the keypoints
    spatial_distance: float   # Euclidean distance in pixel space


@dataclass
class DetectionResult:
    """Structured result returned by the Copy-Move Forgery Detector."""

    is_forgery: bool
    confidence_score: float  # Value between 0.0 (authentic) and 1.0 (forged)
    num_keypoints: int
    num_candidate_matches: int
    num_verified_matches: int
    selected_method: str     # "SIFT" or "ORB"
    keypoints_coords: List[Tuple[float, float]] = field(default_factory=list)
    candidate_matches: List[MatchPair] = field(default_factory=list)
    verified_matches: List[MatchPair] = field(default_factory=list)
    transformation_matrix: Optional[np.ndarray] = None
    binary_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.uint8))
    parameters: dict[str, Any] = field(default_factory=dict)
    diagnostic_warnings: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0


class CopyMoveDetector:
    """Detector for copy-move image tampering using keypoint analysis and geometric consistency."""

    def __init__(
        self,
        method: str = "auto",
        n_features: int = 2000,
        ratio_threshold: float = 0.75,
        min_spatial_distance: float = 30.0,
        min_matches_for_ransac: int = 4,
        min_inliers_for_forgery: int = 6,
        ransac_reproj_threshold: float = 5.0,
        mask_dilation_radius: int = 25,
    ):
        """Initialize the detector with configurable forensic parameters.

        Parameters
        ----------
        method : str
            Feature extractor method: 'sift', 'orb', or 'auto' (default: 'auto').
        n_features : int
            Maximum number of keypoints to extract.
        ratio_threshold : float
            Lowe's ratio test threshold (0.0 to 1.0).
        min_spatial_distance : float
            Minimum pixel distance required between matched keypoints to prevent
            false matches caused by adjacent natural textures.
        min_matches_for_ransac : int
            Minimum candidate matches needed before running geometric verification.
        min_inliers_for_forgery : int
            Minimum geometrically consistent inliers needed to confirm forgery.
        ransac_reproj_threshold : float
            Maximum allowable reprojection error (in pixels) for RANSAC inliers.
        mask_dilation_radius : int
            Radius (in pixels) used when generating the suspicious region mask.
        """
        valid_methods = ("auto", "sift", "orb")
        method_lower = method.strip().lower()
        if method_lower not in valid_methods:
            raise ValueError(
                f"Unsupported method '{method}'. Allowed options are: {valid_methods}"
            )

        self.requested_method = method_lower
        self.n_features = max(100, int(n_features))
        self.ratio_threshold = float(ratio_threshold)
        self.min_spatial_distance = max(1.0, float(min_spatial_distance))
        self.min_matches_for_ransac = max(4, int(min_matches_for_ransac))
        self.min_inliers_for_forgery = max(4, int(min_inliers_for_forgery))
        self.ransac_reproj_threshold = max(0.5, float(ransac_reproj_threshold))
        self.mask_dilation_radius = max(5, int(mask_dilation_radius))

    def _select_extractor(self) -> Tuple[Any, str, List[str]]:
        """Instantiate the feature extractor (SIFT or ORB fallback)."""
        warnings: List[str] = []
        sift_ok = is_sift_available()

        if self.requested_method == "sift":
            if sift_ok:
                return cv2.SIFT_create(nfeatures=self.n_features), "SIFT", warnings
            warnings.append("SIFT requested but not available in cv2; falling back to ORB.")
            return cv2.ORB_create(nfeatures=self.n_features), "ORB", warnings

        if self.requested_method == "orb":
            return cv2.ORB_create(nfeatures=self.n_features), "ORB", warnings

        # 'auto' mode
        if sift_ok:
            return cv2.SIFT_create(nfeatures=self.n_features), "SIFT", warnings

        warnings.append("SIFT unavailable in cv2; automatically selected ORB fallback.")
        return cv2.ORB_create(nfeatures=self.n_features), "ORB", warnings

    def _match_self_descriptors(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray,
        method_name: str,
    ) -> List[MatchPair]:
        """Perform self-matching on descriptors with spatial and ratio filtering."""
        num_kps = len(keypoints)
        if num_kps < 4 or descriptors is None or len(descriptors) < 4:
            return []

        # SIFT uses L2 distance; ORB uses Hamming distance
        norm = cv2.NORM_HAMMING if method_name == "ORB" else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)

        # Retrieve top k neighbors for each descriptor
        k_neighbors = min(12, num_kps)
        try:
            knn_matches = matcher.knnMatch(descriptors, descriptors, k=k_neighbors)
        except cv2.error:
            return []

        kp_pts = [kp.pt for kp in keypoints]
        candidate_pairs: List[MatchPair] = []
        seen_pairs: set[Tuple[int, int]] = set()

        for query_idx, matches in enumerate(knn_matches):
            p_query = kp_pts[query_idx]

            # Filter out self-match and nearby spatial neighbors
            valid_distant_matches: List[cv2.DMatch] = []
            for m in matches:
                if m.trainIdx == query_idx:
                    continue  # Self-match

                p_train = kp_pts[m.trainIdx]
                spatial_dist = float(
                    ((p_query[0] - p_train[0]) ** 2 + (p_query[1] - p_train[1]) ** 2) ** 0.5
                )

                if spatial_dist >= self.min_spatial_distance:
                    valid_distant_matches.append(m)

            # Ratio test on distant neighbors
            if len(valid_distant_matches) >= 2:
                best_match = valid_distant_matches[0]
                second_best = valid_distant_matches[1]

                # Lowe's ratio test
                if best_match.distance < self.ratio_threshold * second_best.distance:
                    idx1 = min(query_idx, best_match.trainIdx)
                    idx2 = max(query_idx, best_match.trainIdx)

                    if (idx1, idx2) not in seen_pairs:
                        seen_pairs.add((idx1, idx2))
                        pt_a = kp_pts[idx1]
                        pt_b = kp_pts[idx2]

                        # Ensure canonical coordinate ordering (pt1 -> pt2 with positive displacement)
                        # This aligns the displacement vector so RANSAC fits consistent forward transformations
                        if (pt_a[0] > pt_b[0]) or (pt_a[0] == pt_b[0] and pt_a[1] > pt_b[1]):
                            pt1, pt2 = pt_b, pt_a
                        else:
                            pt1, pt2 = pt_a, pt_b

                        s_dist = float(
                            ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
                        )
                        candidate_pairs.append(
                            MatchPair(
                                pt1=pt1,
                                pt2=pt2,
                                distance=float(best_match.distance),
                                spatial_distance=s_dist,
                            )
                        )

        return candidate_pairs

    def _verify_geometry_ransac(
        self, candidate_matches: List[MatchPair]
    ) -> Tuple[List[MatchPair], Optional[np.ndarray]]:
        """Verify candidate matches using RANSAC partial affine geometric model."""
        if len(candidate_matches) < self.min_matches_for_ransac:
            return [], None

        pts1 = np.float32([m.pt1 for m in candidate_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([m.pt2 for m in candidate_matches]).reshape(-1, 1, 2)

        # Estimate partial affine (translation + rotation + uniform scaling)
        try:
            matrix, inliers = cv2.estimateAffinePartial2D(
                pts1,
                pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
                maxIters=2000,
                confidence=0.99,
            )
        except Exception:
            return [], None

        if matrix is None or inliers is None:
            return [], None

        inliers_flat = inliers.ravel()
        verified: List[MatchPair] = [
            candidate_matches[i] for i, inl in enumerate(inliers_flat) if inl == 1
        ]

        if len(verified) < self.min_inliers_for_forgery:
            return [], None

        # Transformation matrix sanity checks
        scale_est = float(np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2))
        trans_magnitude = float(np.sqrt(matrix[0, 2] ** 2 + matrix[1, 2] ** 2))

        # Check plausible scaling (0.3x to 3.5x)
        if scale_est < 0.3 or scale_est > 3.5:
            return [], None

        # Defensible cluster separation check:
        # For genuine copy-move forgery, source region and target region are two DISTINCT spatial clusters.
        # If the centroid distance between source points and target points is below min_spatial_distance,
        # it indicates an internal symmetry within a single object rather than a copy-move forgery.
        v_pts1 = np.array([m.pt1 for m in verified])
        v_pts2 = np.array([m.pt2 for m in verified])
        c1 = v_pts1.mean(axis=0)
        c2 = v_pts2.mean(axis=0)
        centroid_separation = float(np.linalg.norm(c1 - c2))

        if centroid_separation < self.min_spatial_distance:
            # Internal self-symmetry within a single object, not a cloned region elsewhere
            return [], None

        return verified, matrix

    def _generate_binary_mask(
        self,
        image_shape: Tuple[int, int],
        verified_matches: List[MatchPair],
    ) -> np.ndarray:
        """Create a binary localization mask highlighting suspicious cloned regions."""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not verified_matches:
            return mask

        # Mark circular regions around both source and target points
        radius = self.mask_dilation_radius
        for match in verified_matches:
            x1, y1 = int(round(match.pt1[0])), int(round(match.pt1[1]))
            x2, y2 = int(round(match.pt2[0])), int(round(match.pt2[1]))

            cv2.circle(mask, (x1, y1), radius, 255, -1)
            cv2.circle(mask, (x2, y2), radius, 255, -1)

        # Morphological closing to fill small gaps between clustered points
        kernel_size = max(5, int(radius * 0.8))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def detect(
        self,
        image_input: Union[str, np.ndarray, Image.Image, PreprocessedImage],
    ) -> DetectionResult:
        """Detect copy-move forgery on the provided image.

        Parameters
        ----------
        image_input : str, np.ndarray, Image.Image, or PreprocessedImage
            The input image to examine.

        Returns
        -------
        DetectionResult
            Structured result containing matches, mask, confidence, and diagnostics.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []

        # 1. Preprocess and validate
        if isinstance(image_input, PreprocessedImage):
            prep = image_input
        else:
            prep = preprocess_image(image_input)

        if prep.is_resized:
            warnings.append(
                f"Image was downscaled by factor {prep.scale_factor:.2f} "
                f"to maintain safe memory limits."
            )

        # 2. Setup extractor
        extractor, method_name, extractor_warnings = self._select_extractor()
        warnings.extend(extractor_warnings)

        # 3. Detect keypoints and descriptors
        keypoints, descriptors = extractor.detectAndCompute(prep.gray, None)

        if keypoints is None:
            keypoints = []

        num_kps = len(keypoints)
        kp_coords = [kp.pt for kp in keypoints]

        parameters = {
            "method": method_name,
            "n_features": self.n_features,
            "ratio_threshold": self.ratio_threshold,
            "min_spatial_distance": self.min_spatial_distance,
            "min_matches_for_ransac": self.min_matches_for_ransac,
            "min_inliers_for_forgery": self.min_inliers_for_forgery,
            "ransac_reproj_threshold": self.ransac_reproj_threshold,
            "mask_dilation_radius": self.mask_dilation_radius,
            "original_shape": prep.original_shape,
            "working_shape": prep.gray.shape,
        }

        # Handle low-feature images (e.g. solid white/black or flat images)
        if num_kps < 15 or descriptors is None or len(descriptors) < 15:
            warnings.append(
                f"Low keypoint count ({num_kps} points detected). Image has insufficient "
                "texture to reliably evaluate copy-move tampering."
            )
            elapsed = time.perf_counter() - start_time
            return DetectionResult(
                is_forgery=False,
                confidence_score=0.0,
                num_keypoints=num_kps,
                num_candidate_matches=0,
                num_verified_matches=0,
                selected_method=method_name,
                keypoints_coords=kp_coords,
                candidate_matches=[],
                verified_matches=[],
                transformation_matrix=None,
                binary_mask=np.zeros(prep.gray.shape[:2], dtype=np.uint8),
                parameters=parameters,
                diagnostic_warnings=warnings,
                execution_time_seconds=elapsed,
            )

        # 4. Self-matching and spatial/ratio filtering
        candidate_matches = self._match_self_descriptors(keypoints, descriptors, method_name)
        num_candidates = len(candidate_matches)

        if num_candidates == 0:
            warnings.append("No candidate feature pairs passed the distance and ratio filters.")
            elapsed = time.perf_counter() - start_time
            return DetectionResult(
                is_forgery=False,
                confidence_score=0.0,
                num_keypoints=num_kps,
                num_candidate_matches=0,
                num_verified_matches=0,
                selected_method=method_name,
                keypoints_coords=kp_coords,
                candidate_matches=[],
                verified_matches=[],
                transformation_matrix=None,
                binary_mask=np.zeros(prep.gray.shape[:2], dtype=np.uint8),
                parameters=parameters,
                diagnostic_warnings=warnings,
                execution_time_seconds=elapsed,
            )

        # 5. Geometric verification using RANSAC
        verified_matches, transform_matrix = self._verify_geometry_ransac(candidate_matches)
        num_verified = len(verified_matches)

        is_forgery = num_verified >= self.min_inliers_for_forgery

        if not is_forgery and num_candidates > 0:
            if num_candidates < self.min_matches_for_ransac:
                warnings.append(
                    f"Candidate matches ({num_candidates}) below minimum required "
                    f"for RANSAC geometric check ({self.min_matches_for_ransac})."
                )
            else:
                warnings.append(
                    f"Candidate matches ({num_candidates}) failed RANSAC geometric verification "
                    f"({num_verified} inliers found; minimum required is {self.min_inliers_for_forgery})."
                )

        # 6. Confidence score calculation
        if is_forgery:
            inlier_ratio = num_verified / max(num_candidates, 1)
            inlier_scale = min(1.0, num_verified / (self.min_inliers_for_forgery * 2.0))
            confidence = float(np.clip(0.5 + 0.3 * inlier_scale + 0.2 * inlier_ratio, 0.0, 1.0))
        else:
            confidence = 0.0

        # 7. Generate predicted binary mask only when supported by evidence
        if is_forgery:
            binary_mask = self._generate_binary_mask(prep.gray.shape, verified_matches)
        else:
            binary_mask = np.zeros(prep.gray.shape[:2], dtype=np.uint8)

        elapsed = time.perf_counter() - start_time

        return DetectionResult(
            is_forgery=is_forgery,
            confidence_score=confidence,
            num_keypoints=num_kps,
            num_candidate_matches=num_candidates,
            num_verified_matches=num_verified,
            selected_method=method_name,
            keypoints_coords=kp_coords,
            candidate_matches=candidate_matches,
            verified_matches=verified_matches,
            transformation_matrix=transform_matrix,
            binary_mask=binary_mask,
            parameters=parameters,
            diagnostic_warnings=warnings,
            execution_time_seconds=elapsed,
        )
