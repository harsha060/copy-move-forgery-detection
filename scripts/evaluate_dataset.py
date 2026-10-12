"""Run comprehensive detection and quantitative evaluation across benchmark samples.

Saves visualizations, binary masks, overlays, evaluation metrics, and JSON reports to outputs/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.detector import CopyMoveDetector
from src.evaluation import evaluate_forgery_detection, evaluate_masks
from src.preprocessing import preprocess_image
from src.visualizer import create_overlay, draw_keypoints, draw_matches


def run_full_evaluation():
    samples_dir = Path("samples")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    auth_path = samples_dir / "authentic.png"
    forged_path = samples_dir / "forged.png"
    gt_path = samples_dir / "ground_truth_mask.png"
    gt_both_path = samples_dir / "ground_truth_both_mask.png"

    detector = CopyMoveDetector(
        method="sift",
        n_features=2500,
        ratio_threshold=0.75,
        min_spatial_distance=50.0,
        min_matches_for_ransac=4,
        min_inliers_for_forgery=6,
    )

    evaluation_records = []

    # -------------------------------------------------------------------------
    # 1. Evaluate Authentic Sample (Negative Case)
    # -------------------------------------------------------------------------
    print("Evaluating Authentic Sample...")
    prep_auth = preprocess_image(auth_path)
    res_auth = detector.detect(prep_auth)

    gt_auth_mask = np.zeros(prep_auth.gray.shape[:2], dtype=np.uint8)
    report_auth = evaluate_forgery_detection(
        image_name="authentic.png",
        predicted_is_forgery=res_auth.is_forgery,
        predicted_mask=res_auth.binary_mask,
        ground_truth_is_forgery=False,
        ground_truth_mask=gt_auth_mask,
    )
    evaluation_records.append(report_auth.to_dict())

    # Save authentic visual outputs
    cv2.imwrite(
        str(outputs_dir / "authentic_matches.png"),
        cv2.cvtColor(draw_matches(prep_auth.original_rgb, res_auth.verified_matches), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(str(outputs_dir / "authentic_mask.png"), res_auth.binary_mask)
    cv2.imwrite(
        str(outputs_dir / "authentic_overlay.png"),
        cv2.cvtColor(create_overlay(prep_auth.original_rgb, res_auth.binary_mask), cv2.COLOR_RGB2BGR),
    )

    # -------------------------------------------------------------------------
    # 2. Evaluate Forged Sample (Positive Case)
    # -------------------------------------------------------------------------
    print("Evaluating Forged Sample...")
    prep_forged = preprocess_image(forged_path)
    res_forged = detector.detect(prep_forged)

    gt_forged_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    gt_both_mask = cv2.imread(str(gt_both_path), cv2.IMREAD_GRAYSCALE)

    # Evaluate against pasted region
    report_forged_pasted = evaluate_forgery_detection(
        image_name="forged.png (pasted region)",
        predicted_is_forgery=res_forged.is_forgery,
        predicted_mask=res_forged.binary_mask,
        ground_truth_is_forgery=True,
        ground_truth_mask=gt_forged_mask,
    )
    # Also evaluate against both regions (source + target)
    report_forged_both = evaluate_forgery_detection(
        image_name="forged.png (source & target regions)",
        predicted_is_forgery=res_forged.is_forgery,
        predicted_mask=res_forged.binary_mask,
        ground_truth_is_forgery=True,
        ground_truth_mask=gt_both_mask,
    )

    evaluation_records.append(report_forged_pasted.to_dict())
    evaluation_records.append(report_forged_both.to_dict())

    # Save forged visual outputs
    cv2.imwrite(
        str(outputs_dir / "forged_matches.png"),
        cv2.cvtColor(draw_matches(prep_forged.original_rgb, res_forged.verified_matches), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(str(outputs_dir / "forged_mask.png"), res_forged.binary_mask)
    cv2.imwrite(
        str(outputs_dir / "forged_overlay.png"),
        cv2.cvtColor(create_overlay(prep_forged.original_rgb, res_forged.binary_mask), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(outputs_dir / "forged_keypoints.png"),
        cv2.cvtColor(draw_keypoints(prep_forged.original_rgb, res_forged.keypoints_coords), cv2.COLOR_RGB2BGR),
    )

    # -------------------------------------------------------------------------
    # 3. Save Ground Truth Comparison Visualization
    # -------------------------------------------------------------------------
    # Visual comparison grid: Original, Ground Truth Mask, Predicted Mask, Overlay
    h, w = prep_forged.original_rgb.shape[:2]
    gt_color = cv2.cvtColor(gt_both_mask, cv2.COLOR_GRAY2RGB)
    pred_color = cv2.cvtColor(res_forged.binary_mask, cv2.COLOR_GRAY2RGB)
    vis_overlay = create_overlay(prep_forged.original_rgb, res_forged.binary_mask)

    comparison_row1 = np.hstack([prep_forged.original_rgb, gt_color])
    comparison_row2 = np.hstack([pred_color, vis_overlay])
    comparison_grid = np.vstack([comparison_row1, comparison_row2])

    cv2.imwrite(
        str(outputs_dir / "ground_truth_comparison.png"),
        cv2.cvtColor(comparison_grid, cv2.COLOR_RGB2BGR),
    )

    # -------------------------------------------------------------------------
    # 4. Save Master JSON Report
    # -------------------------------------------------------------------------
    master_report = {
        "benchmark_evaluation": {
            "dataset": "CMFD_Synthetic_Benchmark_v1",
            "detector_configuration": res_forged.parameters,
            "results": evaluation_records,
            "test_summary": {
                "authentic_sample_verdict": "UNTAMPERED" if not res_auth.is_forgery else "TAMPERED",
                "authentic_sample_correct": not res_auth.is_forgery,
                "forged_sample_verdict": "TAMPERED" if res_forged.is_forgery else "UNTAMPERED",
                "forged_sample_correct": res_forged.is_forgery,
                "forged_verified_inliers": res_forged.num_verified_matches,
                "forged_confidence_score": round(res_forged.confidence_score, 4),
                "pixel_iou_against_both_cloned_regions": report_forged_both.pixel_level.iou,
                "pixel_f1_against_both_cloned_regions": report_forged_both.pixel_level.f1_score,
                "pixel_precision_against_both": report_forged_both.pixel_level.precision,
                "pixel_recall_against_both": report_forged_both.pixel_level.recall,
            },
        }
    }

    with open(outputs_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    # -------------------------------------------------------------------------
    # 5. Save Human-Readable Test Execution Summary
    # -------------------------------------------------------------------------
    summary_md = f"""# Copy-Move Forgery Detection - Evaluation Summary

## 1. Test Dataset Evaluation
- **Authentic Sample (`authentic.png`)**:
  - Ground Truth: Untampered (Negative)
  - Detector Verdict: {'TAMPERED (False Alarm)' if res_auth.is_forgery else 'UNTAMPERED (Correct Negative)'}
  - Classification: {'FP' if res_auth.is_forgery else 'TN'}
  - Verified Matches: {res_auth.num_verified_matches}
  - Pixel IoU: {report_auth.pixel_level.iou:.4f}
  - Pixel Precision: {report_auth.pixel_level.precision:.4f}
  - Pixel Recall: {report_auth.pixel_level.recall:.4f}

- **Forged Sample (`forged.png`)**:
  - Ground Truth: Copy-Move Forged (Positive)
  - Detector Verdict: {'TAMPERED (Detected)' if res_forged.is_forgery else 'UNTAMPERED (Missed)'}
  - Classification: {'TP' if res_forged.is_forgery else 'FN'}
  - Confidence Score: {res_forged.confidence_score * 100:.2f}%
  - Keypoints Extracted: {res_forged.num_keypoints}
  - Candidate Pairs: {res_forged.num_candidate_matches}
  - Verified Inliers (RANSAC): {res_forged.num_verified_matches}
  - Latency: {res_forged.execution_time_seconds * 1000:.1f} ms
  - Pixel IoU (both regions): {report_forged_both.pixel_level.iou:.4f}
  - Pixel F1-Score (both regions): {report_forged_both.pixel_level.f1_score:.4f}
  - Pixel Precision: {report_forged_both.pixel_level.precision:.4f}
  - Pixel Recall: {report_forged_both.pixel_level.recall:.4f}

## 2. Generated Output Files in `outputs/`
- `outputs/authentic_matches.png`
- `outputs/authentic_mask.png`
- `outputs/authentic_overlay.png`
- `outputs/forged_matches.png`
- `outputs/forged_mask.png`
- `outputs/forged_overlay.png`
- `outputs/forged_keypoints.png`
- `outputs/ground_truth_comparison.png`
- `outputs/evaluation_report.json`
- `outputs/test_summary.md`
"""
    with open(outputs_dir / "test_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("Successfully completed full evaluation and saved all outputs to outputs/!")
    return master_report


if __name__ == "__main__":
    run_full_evaluation()
