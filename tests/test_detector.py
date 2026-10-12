"""Unit and integration tests for Copy-Move Forgery Detection engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.preprocessing import PreprocessedImage, preprocess_image, to_grayscale
from src.detector import CopyMoveDetector, DetectionResult, is_sift_available
from src.visualizer import (
    create_binary_mask,
    create_overlay,
    draw_keypoints,
    draw_matches,
)


@pytest.fixture
def authentic_synthetic_image() -> np.ndarray:
    """Generate a textured authentic image without duplicate cloned regions."""
    np.random.seed(42)
    h, w = 300, 300
    # Create smooth gradient background with distinctive non-repeating shapes
    x = np.linspace(0, 1, w, endpoint=False)
    y = np.linspace(0, 1, h, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    base = (np.sin(xx * 5) * np.cos(yy * 7) * 127 + 128).astype(np.uint8)
    image = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    # Add distinct unique geometric features across the canvas
    cv2.circle(image, (60, 60), 25, (255, 0, 0), -1)
    cv2.rectangle(image, (150, 40), (220, 110), (0, 255, 0), -1)
    cv2.line(image, (30, 250), (280, 270), (0, 0, 255), 4)
    cv2.ellipse(image, (80, 190), (30, 15), 45, 0, 360, (255, 255, 0), -1)
    return image


@pytest.fixture
def forged_synthetic_image() -> np.ndarray:
    """Generate an image with an intentionally cloned and pasted textured patch."""
    np.random.seed(101)
    h, w = 400, 400
    image = np.full((h, w, 3), 180, dtype=np.uint8)

    # Add random background texture
    noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
    image = cv2.add(image, noise)

    # Create a feature-rich source patch (80x80) with textured concentric shapes
    patch = np.zeros((80, 80, 3), dtype=np.uint8)
    for r in range(35, 5, -6):
        color = ((r * 37) % 255, (r * 67) % 255, (r * 97) % 255)
        cv2.circle(patch, (40, 40), r, color, 3)
    cv2.rectangle(patch, (15, 15), (65, 65), (255, 255, 255), 2)
    cv2.line(patch, (10, 10), (70, 70), (0, 255, 255), 2)
    cv2.line(patch, (10, 70), (70, 10), (255, 0, 255), 2)

    # Place source patch at (50, 50)
    image[50:130, 50:130] = patch

    # Clone identical patch to (230, 230) (Euclidean distance ~254px)
    image[230:310, 230:310] = patch.copy()

    return image


# --------------------------------------------------------------------------
# Preprocessing tests
# --------------------------------------------------------------------------

def test_preprocess_valid_numpy(authentic_synthetic_image: np.ndarray):
    """Test preprocessing with a valid RGB numpy array."""
    prep = preprocess_image(authentic_synthetic_image)
    assert isinstance(prep, PreprocessedImage)
    assert prep.original_rgb.shape == authentic_synthetic_image.shape
    assert prep.gray.ndim == 2
    assert prep.gray.shape == authentic_synthetic_image.shape[:2]
    assert prep.is_resized is False
    assert prep.scale_factor == 1.0


def test_preprocess_from_file(tmp_path: Path, authentic_synthetic_image: np.ndarray):
    """Test preprocessing with an image file loaded from disk."""
    file_path = tmp_path / "test_sample.png"
    cv2.imwrite(str(file_path), cv2.cvtColor(authentic_synthetic_image, cv2.COLOR_RGB2BGR))

    prep = preprocess_image(file_path)
    assert prep.original_rgb.shape == authentic_synthetic_image.shape
    assert prep.gray.shape == authentic_synthetic_image.shape[:2]


def test_preprocess_pil_image(authentic_synthetic_image: np.ndarray):
    """Test preprocessing with a PIL Image object."""
    pil_img = Image.fromarray(authentic_synthetic_image)
    prep = preprocess_image(pil_img)
    assert prep.original_rgb.shape == authentic_synthetic_image.shape


def test_preprocess_memory_limit_downscaling():
    """Test that oversized images are automatically and safely downscaled."""
    large_image = np.zeros((3000, 3000, 3), dtype=np.uint8)
    prep = preprocess_image(large_image, max_dimension=1000)
    assert prep.is_resized is True
    assert max(prep.original_rgb.shape[:2]) <= 1000
    assert prep.scale_factor < 1.0


def test_preprocess_empty_image_raises_error():
    """Empty numpy array must raise ValueError."""
    empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        preprocess_image(empty_arr)


def test_preprocess_nonexistent_file_raises_error(tmp_path: Path):
    """Non-existent file path must raise FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError):
        preprocess_image(missing_path)


def test_preprocess_corrupted_file_raises_error(tmp_path: Path):
    """Corrupted non-image text file must raise ValueError."""
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_text("This is not a real image file.")
    with pytest.raises(ValueError, match="Failed to decode|could not be decoded"):
        preprocess_image(corrupt_file)


def test_preprocess_subminimum_dimension_raises_error():
    """Tiny image below MIN_ALLOWED_DIMENSION (32px) must raise ValueError."""
    tiny_image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        preprocess_image(tiny_image)


def test_to_grayscale_conversions():
    """Test grayscale conversion on various channel configurations."""
    # 2D array
    g2 = np.full((50, 50), 100, dtype=np.uint8)
    assert to_grayscale(g2).shape == (50, 50)

    # 3D with 1 channel
    g3_1 = np.full((50, 50, 1), 100, dtype=np.uint8)
    assert to_grayscale(g3_1).shape == (50, 50)

    # 3D RGB
    rgb = np.full((50, 50, 3), 100, dtype=np.uint8)
    assert to_grayscale(rgb).shape == (50, 50)


# --------------------------------------------------------------------------
# Detector tests
# --------------------------------------------------------------------------

def test_detector_invalid_method_raises_error():
    """Unsupported method name should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported method"):
        CopyMoveDetector(method="invalid_method_xyz")


def test_detector_low_feature_image_handled_gracefully():
    """Solid blank white canvas should not crash and report low features."""
    solid_image = np.full((200, 200, 3), 255, dtype=np.uint8)
    detector = CopyMoveDetector(method="auto")
    result = detector.detect(solid_image)

    assert isinstance(result, DetectionResult)
    assert result.is_forgery is False
    assert result.confidence_score == 0.0
    assert result.num_verified_matches == 0
    assert result.binary_mask.shape == (200, 200)
    assert np.all(result.binary_mask == 0)
    assert any("Low keypoint count" in w for w in result.diagnostic_warnings)


def test_detector_authentic_image_no_forgery(authentic_synthetic_image: np.ndarray):
    """Authentic image without cloned patches should report no forgery."""
    detector = CopyMoveDetector(method="auto", min_inliers_for_forgery=6)
    result = detector.detect(authentic_synthetic_image)

    assert isinstance(result, DetectionResult)
    assert result.is_forgery is False
    assert result.confidence_score == 0.0
    assert result.num_verified_matches < 6
    assert result.binary_mask.shape == authentic_synthetic_image.shape[:2]
    assert np.all(result.binary_mask == 0)


def test_detector_forged_image_detected_sift(forged_synthetic_image: np.ndarray):
    """Forged image with cloned patch must be correctly identified using SIFT."""
    if not is_sift_available():
        pytest.skip("SIFT is not available in current OpenCV build.")

    detector = CopyMoveDetector(
        method="sift",
        n_features=2000,
        ratio_threshold=0.75,
        min_spatial_distance=30.0,
        min_matches_for_ransac=4,
        min_inliers_for_forgery=5,
    )
    result = detector.detect(forged_synthetic_image)

    assert isinstance(result, DetectionResult)
    assert result.selected_method == "SIFT"
    assert result.is_forgery is True
    assert result.confidence_score > 0.5
    assert result.num_verified_matches >= 5
    assert result.transformation_matrix is not None
    # Binary mask must have suspicious regions marked
    assert result.binary_mask.shape == forged_synthetic_image.shape[:2]
    assert np.any(result.binary_mask == 255)


def test_detector_forged_image_detected_orb(forged_synthetic_image: np.ndarray):
    """Forged image with cloned patch must also be detectable using ORB."""
    detector = CopyMoveDetector(
        method="orb",
        n_features=2500,
        ratio_threshold=0.80,
        min_spatial_distance=30.0,
        min_matches_for_ransac=4,
        min_inliers_for_forgery=5,
    )
    result = detector.detect(forged_synthetic_image)

    assert isinstance(result, DetectionResult)
    assert result.selected_method == "ORB"
    assert result.num_keypoints > 0


def test_detector_auto_selection():
    """Auto mode should pick SIFT when available or ORB otherwise."""
    detector = CopyMoveDetector(method="auto")
    extractor, name, _ = detector._select_extractor()
    if is_sift_available():
        assert name == "SIFT"
    else:
        assert name == "ORB"


# --------------------------------------------------------------------------
# Visualizer tests
# --------------------------------------------------------------------------

def test_visualizer_output_dimensions_and_channels(forged_synthetic_image: np.ndarray):
    """Visualizer functions must strictly preserve image dimensions and RGB format."""
    detector = CopyMoveDetector(method="auto", min_inliers_for_forgery=5)
    result = detector.detect(forged_synthetic_image)

    h, w = forged_synthetic_image.shape[:2]

    # 1. Keypoint visualization
    kps_vis = draw_keypoints(forged_synthetic_image, result.keypoints_coords)
    assert kps_vis.shape == (h, w, 3)
    assert kps_vis.dtype == np.uint8

    # 2. Match lines visualization
    matches_vis = draw_matches(forged_synthetic_image, result.verified_matches)
    assert matches_vis.shape == (h, w, 3)
    assert matches_vis.dtype == np.uint8

    # 3. Binary mask
    mask = create_binary_mask(h, w, result.verified_matches)
    assert mask.shape == (h, w)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})

    # 4. Highlight Overlay
    overlay = create_overlay(forged_synthetic_image, result.binary_mask)
    assert overlay.shape == (h, w, 3)
    assert overlay.dtype == np.uint8


def test_visualizer_empty_mask_overlay(authentic_synthetic_image: np.ndarray):
    """Overlay with an empty mask should return original image without modifications."""
    h, w = authentic_synthetic_image.shape[:2]
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    overlay = create_overlay(authentic_synthetic_image, empty_mask)

    assert overlay.shape == authentic_synthetic_image.shape
    assert np.array_equal(overlay, authentic_synthetic_image)


# --------------------------------------------------------------------------
# Evaluation & Metric Tests
# --------------------------------------------------------------------------

from src.evaluation import evaluate_masks, evaluate_forgery_detection, PixelMetrics, FullEvaluationReport
from scripts.generate_samples import generate_synthetic_samples
from app import image_to_png_bytes, load_image_bytes


def test_evaluation_identical_positive_masks():
    """Identical non-empty masks should return 1.0 for Precision, Recall, F1, and IoU."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:60, 20:60] = 255
    pred = gt.copy()

    metrics = evaluate_masks(pred, gt)
    assert isinstance(metrics, PixelMetrics)
    assert metrics.tp_pixels == 40 * 40
    assert metrics.fp_pixels == 0
    assert metrics.fn_pixels == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.iou == 1.0


def test_evaluation_both_empty_masks():
    """Both masks empty (authentic correctly classified) must return 1.0 across all metrics."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    pred = np.zeros((100, 100), dtype=np.uint8)

    metrics = evaluate_masks(pred, gt)
    assert metrics.tp_pixels == 0
    assert metrics.fp_pixels == 0
    assert metrics.fn_pixels == 0
    assert metrics.tn_pixels == 10000
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.iou == 1.0


def test_evaluation_disjoint_masks():
    """Completely non-overlapping masks must yield 0.0 for Precision, Recall, F1, and IoU."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[60:80, 60:80] = 255

    metrics = evaluate_masks(pred, gt)
    assert metrics.tp_pixels == 0
    assert metrics.fp_pixels == 400
    assert metrics.fn_pixels == 400
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0
    assert metrics.iou == 0.0


def test_evaluation_partial_overlap():
    """Controlled 50% overlap should compute correct numerical IoU."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:60, 20:60] = 255  # 40x40 = 1600 px

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[20:60, 40:80] = 255  # 40x40 = 1600 px, overlap is [20:60, 40:60] = 40x20 = 800 px

    metrics = evaluate_masks(pred, gt)
    assert metrics.tp_pixels == 800
    assert metrics.fp_pixels == 800
    assert metrics.fn_pixels == 800
    # union = 800 + 800 + 800 = 2400
    expected_iou = round(800 / 2400, 4)
    assert metrics.iou == expected_iou
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5


def test_evaluation_full_report_structure():
    """Full evaluation report must correctly serialize detection and pixel metrics."""
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((50, 50), dtype=np.uint8)

    report = evaluate_forgery_detection(
        image_name="test_sample.png",
        predicted_is_forgery=False,
        predicted_mask=pred,
        ground_truth_is_forgery=False,
        ground_truth_mask=gt,
    )
    assert isinstance(report, FullEvaluationReport)
    assert report.detection_level.classification_category == "TN"
    assert report.detection_level.verdict_correct is True

    rep_dict = report.to_dict()
    assert "detection_level" in rep_dict
    assert "pixel_level" in rep_dict
    assert rep_dict["pixel_level"]["iou"] == 1.0


def test_synthetic_sample_generator_reproducibility(tmp_path: Path):
    """Synthetic dataset generator must create all expected images and metadata."""
    meta = generate_synthetic_samples(tmp_path, image_size=(200, 200), random_seed=99)
    assert (tmp_path / "authentic.png").exists()
    assert (tmp_path / "forged.png").exists()
    assert (tmp_path / "ground_truth_mask.png").exists()
    assert (tmp_path / "ground_truth_both_mask.png").exists()
    assert (tmp_path / "metadata.json").exists()
    assert meta["image_dimensions"]["width"] == 200
    assert meta["image_dimensions"]["height"] == 200


def test_app_helpers_image_to_png_and_decode(tmp_path: Path, authentic_synthetic_image: np.ndarray):
    """Streamlit app helpers must convert numpy arrays to PNG bytes and safely decode."""
    png_bytes = image_to_png_bytes(authentic_synthetic_image)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    # PNG signature check: \x89PNG
    assert png_bytes[:4] == b"\x89PNG"

    # Test load_image_bytes
    img_arr, name, size = load_image_bytes(png_bytes)
    assert img_arr.shape == authentic_synthetic_image.shape
    assert size == len(png_bytes)
