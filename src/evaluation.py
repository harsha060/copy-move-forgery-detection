"""Forensic evaluation metrics for Copy-Move Image Forgery Detection.

Implements pixel-level and image-level metrics including Precision, Recall,
F1-Score, and Intersection over Union (IoU) with rigorous edge-case handling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class PixelMetrics:
    """Pixel-level localization metrics comparing predicted mask to ground truth."""

    tp_pixels: int  # True Positives: Tampered pixels correctly detected
    fp_pixels: int  # False Positives: Authentic pixels falsely flagged
    fn_pixels: int  # False Negatives: Tampered pixels missed by detector
    tn_pixels: int  # True Negatives: Authentic pixels correctly identified
    precision: float
    recall: float
    f1_score: float
    iou: float


@dataclass
class DetectionMetrics:
    """Image-level classification performance metrics."""

    is_forged_ground_truth: bool
    is_forged_predicted: bool
    verdict_correct: bool
    classification_category: str  # "TP", "TN", "FP", or "FN"


@dataclass
class FullEvaluationReport:
    """Comprehensive evaluation containing both detection-level and pixel-level metrics."""

    image_name: str
    detection_level: DetectionMetrics
    pixel_level: PixelMetrics
    total_pixels: int
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a serializable dictionary."""
        return asdict(self)


def evaluate_masks(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: int = 127,
) -> PixelMetrics:
    """Calculate pixel-level confusion matrix, Precision, Recall, F1, and IoU.

    Parameters
    ----------
    predicted_mask : np.ndarray
        Predicted binary mask (2D array, uint8 or boolean).
    ground_truth_mask : np.ndarray
        Ground-truth binary mask (2D array, uint8 or boolean).
    threshold : int
        Binarization threshold (default 127).

    Returns
    -------
    PixelMetrics
        Calculated pixel metrics with documented division-by-zero handling.
    """
    if not isinstance(predicted_mask, np.ndarray) or not isinstance(ground_truth_mask, np.ndarray):
        raise TypeError("Both predicted_mask and ground_truth_mask must be numpy arrays.")

    if predicted_mask.shape != ground_truth_mask.shape:
        # Resize predicted mask to match ground truth dimensions safely
        h_gt, w_gt = ground_truth_mask.shape[:2]
        predicted_mask = cv2.resize(predicted_mask, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)

    p_bool = (predicted_mask > threshold).astype(bool)
    g_bool = (ground_truth_mask > threshold).astype(bool)

    tp = int(np.logical_and(p_bool, g_bool).sum())
    fp = int(np.logical_and(p_bool, np.logical_not(g_bool)).sum())
    fn = int(np.logical_and(np.logical_not(p_bool), g_bool).sum())
    tn = int(np.logical_and(np.logical_not(p_bool), np.logical_not(g_bool)).sum())

    total_predicted_positive = tp + fp
    total_actual_positive = tp + fn

    # Edge cases handling:
    # 1. Both GT and Prediction are completely empty (authentic image correctly predicted with no mask)
    if total_actual_positive == 0 and total_predicted_positive == 0:
        precision = 1.0
        recall = 1.0
        f1_score = 1.0
        iou = 1.0
    # 2. Prediction is empty, but GT has positive pixels (missed forgery)
    elif total_predicted_positive == 0 and total_actual_positive > 0:
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        iou = 0.0
    # 3. GT is empty, but prediction hallucinated positive pixels (false alarm on authentic)
    elif total_actual_positive == 0 and total_predicted_positive > 0:
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        iou = 0.0
    # 4. Standard case
    else:
        precision = float(tp / total_predicted_positive) if total_predicted_positive > 0 else 0.0
        recall = float(tp / total_actual_positive) if total_actual_positive > 0 else 0.0
        denom_f1 = precision + recall
        f1_score = float((2 * precision * recall) / denom_f1) if denom_f1 > 0 else 0.0
        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0

    return PixelMetrics(
        tp_pixels=tp,
        fp_pixels=fp,
        fn_pixels=fn,
        tn_pixels=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1_score, 4),
        iou=round(iou, 4),
    )


def evaluate_forgery_detection(
    image_name: str,
    predicted_is_forgery: bool,
    predicted_mask: np.ndarray,
    ground_truth_is_forgery: bool,
    ground_truth_mask: Optional[np.ndarray] = None,
) -> FullEvaluationReport:
    """Perform complete evaluation combining detection-level and pixel-level metrics.

    Parameters
    ----------
    image_name : str
        Name or identifier of the evaluated image.
    predicted_is_forgery : bool
        Whether the algorithm declared the image tampered.
    predicted_mask : np.ndarray
        Predicted binary mask.
    ground_truth_is_forgery : bool
        True ground truth label for the image.
    ground_truth_mask : np.ndarray, optional
        Ground truth localization mask. If None, an all-zero mask is assumed.

    Returns
    -------
    FullEvaluationReport
        Complete forensic evaluation report.
    """
    # 1. Detection-Level Classification
    verdict_correct = (predicted_is_forgery == ground_truth_is_forgery)

    if ground_truth_is_forgery and predicted_is_forgery:
        cat = "TP"  # True Positive
    elif not ground_truth_is_forgery and not predicted_is_forgery:
        cat = "TN"  # True Negative
    elif not ground_truth_is_forgery and predicted_is_forgery:
        cat = "FP"  # False Positive
    else:
        cat = "FN"  # False Negative

    det_metrics = DetectionMetrics(
        is_forged_ground_truth=ground_truth_is_forgery,
        is_forged_predicted=predicted_is_forgery,
        verdict_correct=verdict_correct,
        classification_category=cat,
    )

    # 2. Pixel-Level Metrics
    if ground_truth_mask is None:
        ground_truth_mask = np.zeros(predicted_mask.shape[:2], dtype=np.uint8)

    pix_metrics = evaluate_masks(predicted_mask, ground_truth_mask)

    summary = {
        "detection_verdict": "CORRECT" if verdict_correct else "INCORRECT",
        "classification": cat,
        "pixel_precision": pix_metrics.precision,
        "pixel_recall": pix_metrics.recall,
        "pixel_f1": pix_metrics.f1_score,
        "pixel_iou": pix_metrics.iou,
    }

    return FullEvaluationReport(
        image_name=image_name,
        detection_level=det_metrics,
        pixel_level=pix_metrics,
        total_pixels=predicted_mask.size,
        summary=summary,
    )
