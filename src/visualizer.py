"""Visualization module for Copy-Move Image Forgery Detection results."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from src.detector import DetectionResult, MatchPair


def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure an image is 3-channel RGB with uint8 data type."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Cannot visualize empty image array.")

    # Convert float if necessary
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3:
        if image.shape[2] == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 3:
            return image.copy()
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    raise ValueError(f"Unsupported array shape for visualization: {image.shape}")


def draw_keypoints(
    image: np.ndarray,
    keypoints_coords: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 128),  # Vibrant green-cyan (RGB)
    radius: int = 3,
) -> np.ndarray:
    """Render extracted keypoints on the image.

    Parameters
    ----------
    image : np.ndarray
        Base image (RGB uint8 or convertible).
    keypoints_coords : list of (x, y)
        Keypoint coordinate tuples.
    color : tuple of 3 ints
        Color of the keypoint circles in RGB format.
    radius : int
        Radius of keypoint circles in pixels.

    Returns
    -------
    np.ndarray
        RGB uint8 image with keypoints drawn.
    """
    vis = _ensure_rgb_uint8(image)
    if not keypoints_coords:
        return vis

    for pt in keypoints_coords:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

    return vis


def draw_matches(
    image: np.ndarray,
    verified_matches: List[MatchPair],
    candidate_matches: Optional[List[MatchPair]] = None,
    show_candidates: bool = False,
    verified_color: Tuple[int, int, int] = (0, 255, 0),       # Green for verified
    candidate_color: Tuple[int, int, int] = (255, 200, 0),    # Amber/yellow for candidates
    line_thickness: int = 2,
    point_radius: int = 4,
) -> np.ndarray:
    """Render connection lines between matched suspicious patches.

    Parameters
    ----------
    image : np.ndarray
        Base RGB image.
    verified_matches : list of MatchPair
        Geometrically verified inlier match pairs.
    candidate_matches : list of MatchPair, optional
        Pre-RANSAC candidate matches.
    show_candidates : bool
        If True, draws unverified candidate matches as thin reference lines.
    verified_color : tuple of 3 ints
        RGB color for verified match lines and endpoints.
    candidate_color : tuple of 3 ints
        RGB color for candidate match lines.
    line_thickness : int
        Pixel thickness of verified connection vectors.
    point_radius : int
        Pixel radius of match endpoints.

    Returns
    -------
    np.ndarray
        RGB uint8 image with match vectors drawn.
    """
    vis = _ensure_rgb_uint8(image)
    h, w = vis.shape[:2]

    # Optionally draw candidate matches first (background)
    if show_candidates and candidate_matches:
        for match in candidate_matches:
            x1, y1 = int(round(match.pt1[0])), int(round(match.pt1[1]))
            x2, y2 = int(round(match.pt2[0])), int(round(match.pt2[1]))
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(vis, (x1, y1), (x2, y2), candidate_color, 1, lineType=cv2.LINE_AA)

    # Draw verified matches
    for match in verified_matches:
        x1, y1 = int(round(match.pt1[0])), int(round(match.pt1[1]))
        x2, y2 = int(round(match.pt2[0])), int(round(match.pt2[1]))

        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            # Draw line vector
            cv2.line(vis, (x1, y1), (x2, y2), verified_color, line_thickness, lineType=cv2.LINE_AA)
            # Draw endpoint circles
            cv2.circle(vis, (x1, y1), point_radius, verified_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, (x2, y2), point_radius, (255, 50, 50), -1, lineType=cv2.LINE_AA)  # Red for copy

    return vis


def create_binary_mask(
    height: int,
    width: int,
    verified_matches: List[MatchPair],
    dilation_radius: int = 25,
) -> np.ndarray:
    """Generate a binary mask (0 or 255) marking detected suspicious regions.

    Parameters
    ----------
    height : int
        Height of output mask.
    width : int
        Width of output mask.
    verified_matches : list of MatchPair
        Verified match pairs.
    dilation_radius : int
        Radius around each verified point to include in the mask.

    Returns
    -------
    np.ndarray
        Single-channel binary mask of shape (height, width) with values 0 or 255.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not verified_matches:
        return mask

    for match in verified_matches:
        x1, y1 = int(round(match.pt1[0])), int(round(match.pt1[1]))
        x2, y2 = int(round(match.pt2[0])), int(round(match.pt2[1]))
        cv2.circle(mask, (x1, y1), dilation_radius, 255, -1)
        cv2.circle(mask, (x2, y2), dilation_radius, 255, -1)

    kernel_size = max(5, int(dilation_radius * 0.8))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def create_overlay(
    image: np.ndarray,
    binary_mask: np.ndarray,
    overlay_color: Tuple[int, int, int] = (255, 40, 40),  # Red tint (RGB)
    contour_color: Tuple[int, int, int] = (255, 255, 0),  # Yellow border (RGB)
    alpha: float = 0.45,
) -> np.ndarray:
    """Create a translucent color overlay highlighting detected regions on the original image.

    Parameters
    ----------
    image : np.ndarray
        Original image (RGB format).
    binary_mask : np.ndarray
        Binary mask (0 or 255) indicating suspicious regions.
    overlay_color : tuple of 3 ints
        Highlight color in RGB format.
    contour_color : tuple of 3 ints
        Border color for suspicious region boundaries in RGB format.
    alpha : float
        Opacity of the highlight color (0.0 to 1.0).

    Returns
    -------
    np.ndarray
        RGB uint8 image with overlay and boundary contours applied.
    """
    vis = _ensure_rgb_uint8(image)
    h, w = vis.shape[:2]

    if not isinstance(binary_mask, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray for binary_mask, got {type(binary_mask).__name__}")

    if binary_mask.shape[:2] != (h, w):
        # Resize mask to match image dimensions safely
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Ensure mask is single channel uint8 boolean
    mask_bool = (binary_mask > 127)

    if not np.any(mask_bool):
        # No suspicious regions found: return clean image
        return vis

    # Apply alpha blending over highlighted regions
    overlay = vis.copy()
    color_layer = np.zeros_like(vis, dtype=np.uint8)
    color_layer[mask_bool] = overlay_color

    blended = cv2.addWeighted(overlay, 1.0 - alpha, color_layer, alpha, 0)
    vis[mask_bool] = blended[mask_bool]

    # Draw crisp contour borders around suspicious regions
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, contour_color, 2, lineType=cv2.LINE_AA)

    return vis
