# Copy-Move Forgery Detection - Evaluation Summary

## 1. Test Dataset Evaluation
- **Authentic Sample (`authentic.png`)**:
  - Ground Truth: Untampered (Negative)
  - Detector Verdict: UNTAMPERED (Correct Negative)
  - Classification: TN
  - Verified Matches: 0
  - Pixel IoU: 1.0000
  - Pixel Precision: 1.0000
  - Pixel Recall: 1.0000

- **Forged Sample (`forged.png`)**:
  - Ground Truth: Copy-Move Forged (Positive)
  - Detector Verdict: TAMPERED (Detected)
  - Classification: TP
  - Confidence Score: 99.79%
  - Keypoints Extracted: 380
  - Candidate Pairs: 187
  - Verified Inliers (RANSAC): 185
  - Latency: 67.7 ms
  - Pixel IoU (both regions): 0.7234
  - Pixel F1-Score (both regions): 0.8395
  - Pixel Precision: 0.7268
  - Pixel Recall: 0.9935

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
