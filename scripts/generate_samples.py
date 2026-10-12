"""Reproducible synthetic sample generator for Copy-Move Image Forgery Detection."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def generate_synthetic_samples(
    output_dir: Path,
    image_size: Tuple[int, int] = (500, 500),
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Generate reproducible authentic, copy-move forged, ground-truth masks, and metadata.

    Parameters
    ----------
    output_dir : Path
        Directory where generated images and metadata will be saved.
    image_size : tuple of (int, int)
        (width, height) of generated canvas.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Metadata dictionary describing the generated dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(random_seed)

    w, h = image_size

    # 1. Generate smooth, non-periodic naturalistic gradient background
    x = np.linspace(0, 3 * np.pi, w)
    y = np.linspace(0, 3 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    base_gray = (np.sin(xx * 0.7) * np.cos(yy * 0.6) * 55 + 165).astype(np.uint8)
    authentic = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

    # 2. Add distinct non-repeating shapes to authentic background
    rect_x1, rect_y1 = int(w * 0.6), int(h * 0.15)
    rect_x2, rect_y2 = int(w * 0.9), int(h * 0.35)
    cv2.rectangle(authentic, (rect_x1, rect_y1), (rect_x2, rect_y2), (180, 60, 180), -1)
    cv2.line(authentic, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), 3)

    ell_cx, ell_cy = int(w * 0.5), int(h * 0.75)
    ell_ax1, ell_ax2 = max(10, int(w * 0.15)), max(5, int(h * 0.08))
    cv2.ellipse(authentic, (ell_cx, ell_cy), (ell_ax1, ell_ax2), 25, 0, 360, (50, 150, 220), -1)

    # 3. Create a feature-rich source object
    patch_w = max(32, int(w * 0.2))
    patch_h = max(32, int(h * 0.2))
    src_x = int(w * 0.1)
    src_y = int(h * 0.1)

    cx = src_x + patch_w // 2
    cy = src_y + patch_h // 2
    max_r = min(patch_w, patch_h) // 2 - 4

    for r in range(max_r, 4, -max(2, max_r // 6)):
        color = ((r * 37) % 255, (r * 67) % 255, (r * 97) % 255)
        cv2.circle(authentic, (cx, cy), r, color, 2)

    cv2.rectangle(
        authentic,
        (cx - patch_w // 4, cy - patch_h // 4),
        (cx + patch_w // 4, cy + patch_h // 4),
        (40, 220, 220),
        2,
    )
    cv2.line(authentic, (src_x + 10, src_y + 10), (src_x + patch_w - 10, src_y + patch_h - 10), (255, 0, 255), 2)
    cv2.line(authentic, (src_x + 10, src_y + patch_h - 10), (src_x + patch_w - 10, src_y + 10), (0, 255, 255), 2)
    cv2.putText(authentic, "CMFD", (cx - 20, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Save authentic baseline image (Negative case)
    auth_path = output_dir / "authentic.png"
    cv2.imwrite(str(auth_path), authentic)

    # 4. Create copy-move forged image
    # Extract source patch and paste at distant target coordinates
    dst_x = min(w - patch_w, int(w * 0.6))
    dst_y = min(h - patch_h, int(h * 0.6))

    forged = authentic.copy()
    cloned_patch = authentic[src_y : src_y + patch_h, src_x : src_x + patch_w].copy()
    forged[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = cloned_patch

    forged_path = output_dir / "forged.png"
    cv2.imwrite(str(forged_path), forged)

    # 5. Create ground-truth binary mask (Pasted region only: 255, background: 0)
    gt_mask_pasted = np.zeros((h, w), dtype=np.uint8)
    gt_mask_pasted[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = 255
    gt_pasted_path = output_dir / "ground_truth_mask.png"
    cv2.imwrite(str(gt_pasted_path), gt_mask_pasted)

    # 6. Create ground-truth binary mask for both regions (Source + Target)
    gt_mask_both = np.zeros((h, w), dtype=np.uint8)
    gt_mask_both[src_y : src_y + patch_h, src_x : src_x + patch_w] = 255
    gt_mask_both[dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = 255
    gt_both_path = output_dir / "ground_truth_both_mask.png"
    cv2.imwrite(str(gt_both_path), gt_mask_both)

    # 7. Calculate displacement and metadata
    dx = dst_x - src_x
    dy = dst_y - src_y
    euclidean_disp = float(np.sqrt(dx**2 + dy**2))

    metadata: Dict[str, Any] = {
        "dataset_name": "CMFD_Synthetic_Benchmark_v1",
        "generated_at": datetime.datetime.now().isoformat(),
        "random_seed": random_seed,
        "image_dimensions": {
            "width": w,
            "height": h,
            "channels": 3,
        },
        "manipulation_type": "Copy-Move (Cloning)",
        "source_region": {
            "x": src_x,
            "y": src_y,
            "width": patch_w,
            "height": patch_h,
            "center": [src_x + patch_w / 2, src_y + patch_h / 2],
        },
        "target_region": {
            "x": dst_x,
            "y": dst_y,
            "width": patch_w,
            "height": patch_h,
            "center": [dst_x + patch_w / 2, dst_y + patch_h / 2],
        },
        "displacement": {
            "dx": dx,
            "dy": dy,
            "euclidean_distance_px": round(euclidean_disp, 2),
        },
        "transformation": "Translation (Identity scaling and 0 rotation)",
        "files": {
            "authentic_image": "authentic.png",
            "forged_image": "forged.png",
            "ground_truth_pasted_mask": "ground_truth_mask.png",
            "ground_truth_both_mask": "ground_truth_both_mask.png",
        },
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    target_dir = Path(__file__).parent.parent / "samples"
    meta = generate_synthetic_samples(target_dir)
    print(f"Successfully generated samples in: {target_dir}")
    print(f"Metadata summary: {meta['dataset_name']}, displacement={meta['displacement']}")
