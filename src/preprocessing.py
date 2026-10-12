"""Image preprocessing and validation module for Copy-Move Forgery Detection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

# Safety limits for memory and performance
MAX_ALLOWED_DIMENSION = 2048
MAX_ALLOWED_PIXELS = 10_000_000  # ~10 Megapixels
MIN_ALLOWED_DIMENSION = 32


@dataclass
class PreprocessedImage:
    """Container holding both display-ready RGB and algorithm-ready grayscale images."""

    original_rgb: np.ndarray  # HxWx3 uint8 in RGB format for visualization
    gray: np.ndarray          # HxW uint8 grayscale for feature extraction
    original_shape: tuple[int, int]  # (height, width) before any resizing
    is_resized: bool
    scale_factor: float


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to a single-channel 8-bit grayscale array.

    Parameters
    ----------
    image : np.ndarray
        Input image array. Can be 2D (grayscale), 3-channel (RGB or BGR),
        or 4-channel (RGBA).

    Returns
    -------
    np.ndarray
        Grayscale uint8 image of shape (H, W).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Cannot convert empty image array to grayscale.")

    # Convert to uint8 if float
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        return image.copy()

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].copy()
        if channels == 3:
            # Assumes input is RGB
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if channels == 4:
            # RGBA to RGB then Gray
            rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    raise ValueError(f"Unsupported image array shape for grayscale conversion: {image.shape}")


def _load_image_from_path(file_path: Union[str, Path]) -> np.ndarray:
    """Safely load an image from a file path as an RGB uint8 array."""
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file does not exist: {file_path}")

    if not path_obj.is_file():
        raise ValueError(f"Provided path is not a regular file: {file_path}")

    if path_obj.stat().st_size == 0:
        raise ValueError(f"File is empty (0 bytes): {file_path}")

    # Read binary bytes first to support non-ASCII and Windows paths safely
    try:
        with open(path_obj, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise ValueError(f"Failed to decode image from {file_path}: {exc}") from exc

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError(f"File could not be decoded as a valid image: {file_path}")

    # Standardize to RGB uint8
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    if img_bgr.ndim == 3:
        if img_bgr.shape[2] == 4:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
        if img_bgr.shape[2] == 3:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_bgr.shape[2] == 1:
            return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)

    raise ValueError(f"Decoded image has unexpected shape: {img_bgr.shape}")


def preprocess_image(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    max_dimension: int = MAX_ALLOWED_DIMENSION,
    max_pixels: int = MAX_ALLOWED_PIXELS,
) -> PreprocessedImage:
    """Validate, standardize, and safely preprocess an input image.

    Parameters
    ----------
    image_input : str, Path, np.ndarray, or PIL.Image.Image
        The input image to preprocess.
    max_dimension : int
        Maximum allowable width or height before proportional downscaling.
    max_pixels : int
        Maximum allowable total pixels before proportional downscaling.

    Returns
    -------
    PreprocessedImage
        Container with original_rgb, gray, original_shape, and scaling info.
    """
    # 1. Resolve image input into standard RGB numpy array
    if isinstance(image_input, (str, Path)):
        rgb_image = _load_image_from_path(image_input)
    elif isinstance(image_input, Image.Image):
        rgb_image = np.array(image_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Input numpy array is empty (size 0).")
        if image_input.ndim not in (2, 3):
            raise ValueError(f"Array must be 2D or 3D, got {image_input.ndim}D array.")

        # Handle float normalization
        arr = image_input
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            rgb_image = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 1:
            rgb_image = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 3:
            rgb_image = arr.copy()
        elif arr.shape[2] == 4:
            rgb_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported number of channels: {arr.shape[2]}")
    else:
        raise TypeError(
            f"Unsupported image input type: {type(image_input).__name__}. "
            "Expected file path, PIL Image, or numpy array."
        )

    orig_h, orig_w = rgb_image.shape[:2]

    # Validate minimal dimensions
    if orig_h < MIN_ALLOWED_DIMENSION or orig_w < MIN_ALLOWED_DIMENSION:
        raise ValueError(
            f"Image dimensions ({orig_w}x{orig_h}) are too small. "
            f"Minimum allowable dimension is {MIN_ALLOWED_DIMENSION}px."
        )

    # 2. Check memory limits & downscale if needed (preserving aspect ratio)
    scale_factor = 1.0
    is_resized = False

    current_pixels = orig_h * orig_w
    if orig_h > max_dimension or orig_w > max_dimension or current_pixels > max_pixels:
        dim_scale = max_dimension / max(orig_h, orig_w)
        pixel_scale = (max_pixels / current_pixels) ** 0.5
        scale_factor = min(dim_scale, pixel_scale)

        new_w = max(MIN_ALLOWED_DIMENSION, int(round(orig_w * scale_factor)))
        new_h = max(MIN_ALLOWED_DIMENSION, int(round(orig_h * scale_factor)))

        rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        is_resized = True

    # 3. Create grayscale counterpart
    gray_image = to_grayscale(rgb_image)

    return PreprocessedImage(
        original_rgb=rgb_image,
        gray=gray_image,
        original_shape=(orig_h, orig_w),
        is_resized=is_resized,
        scale_factor=scale_factor,
    )
