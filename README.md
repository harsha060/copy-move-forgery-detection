# Copy-Move Image Forgery Detection Using Digital Image Processing

An AI-powered digital image forensics application designed to detect, localize, and evaluate **Copy-Move Image Tampering** (cloned patches) using classical Digital Image Processing (DIP) algorithms with an interactive Streamlit web dashboard.

Developed as a **B.Tech Major Project** in Computer Science & Engineering.

---

## 🌟 Overview

Copy-Move forgery occurs when an attacker copies a part of an image and pastes it into another location within the **same image**—often to conceal an unwanted object or duplicate elements (such as people, vehicles, or signatures). Because the cloned region shares identical noise characteristics, lighting, and texture with the rest of the canvas, standard noise analysis cannot reliably flag the manipulation.

This project implements a robust forensic pipeline combining:
1. **Multi-Scale Feature Extraction**: SIFT (Scale-Invariant Feature Transform) with automatic ORB (Oriented FAST and Rotated BRIEF) fallback.
2. **Self-Referential $k$-NN Matching**: Compares the image descriptors against themselves while filtering out trivial self-matches.
3. **Spatial Distance Thresholding**: Discards adjacent natural texture matches, retaining only distant cloned candidates.
4. **Geometric Consensus Verification (RANSAC)**: Fits an Affine Partial model ($s \cdot R \cdot x + t$) to identify geometrically consistent displacement vectors.
5. **Cluster Disjointness & Rotational Symmetry Filtering**: Eliminates false positives arising from naturally symmetric single objects (e.g. concentric logos, wheels).
6. **Binary Mask & Translucent Overlay Localization**: Generates pixel-accurate localization masks and visual overlay highlights.
7. **Quantitative Evaluation Module**: Calculates Precision, Recall, F1-Score, and Intersection over Union (IoU) against ground-truth masks.

---

## 🚀 Key Features

* **Interactive Streamlit Web Dashboard**: High-contrast, cybersecurity-themed dark UI with real-time feedback.
* **Side-by-Side Forensic Inspection**: Compare the original image directly with the predicted highlight overlay and match vectors.
* **Detailed Visual Workspaces**:
  * Original Image
  * Feature Match Displacement Vectors (green connecting lines)
  * Predicted Binary Localization Mask
  * Extracted Keypoints Distribution
  * Diagnostic Audit Trail (Affine Transformation Matrix, parameters, and warnings)
* **One-Click Sample Benchmarks**: Instantly test on authentic and copy-move forged benchmark images.
* **Artifact Exports**: Download predicted masks, overlay images, match vector plots, and complete JSON forensic reports.
* **Fully Tested**: 24 automated pytest unit tests covering image loading, edge cases, metrics, and application logic.

---

## 🏗️ Project Architecture

```
Copy_Move_Forgery_Detection/
│
├── app.py                      # Streamlit forensic web application
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git exclusions (virtual env, caches, secrets)
├── README.md                   # Project documentation
│
├── src/                        # Core algorithmic package
│   ├── __init__.py             # Public package exports
│   ├── preprocessing.py         # Image validation, memory safeguards, grayscale conversion
│   ├── detector.py             # SIFT/ORB extraction, self-matching, RANSAC verification
│   ├── visualizer.py           # Keypoint, match vector, mask, and overlay renderers
│   └── evaluation.py           # Precision, Recall, F1, IoU metrics and reports
│
├── scripts/                    # Utilities and evaluation pipelines
│   ├── generate_samples.py     # Reproducible synthetic test dataset generator
│   └── evaluate_dataset.py     # Benchmark evaluation and report generator
│
├── samples/                    # Test images and ground truth
│   ├── authentic.png           # Authentic negative sample
│   ├── forged.png              # Copy-move forged positive sample
│   ├── ground_truth_mask.png   # Ground truth mask (pasted region)
│   ├── ground_truth_both_mask.png # Ground truth mask (both cloned regions)
│   └── metadata.json           # Sample metadata and coordinates
│
├── outputs/                    # Exported evaluation artifacts & reports
│   ├── authentic_matches.png
│   ├── authentic_mask.png
│   ├── authentic_overlay.png
│   ├── forged_matches.png
│   ├── forged_mask.png
│   ├── forged_overlay.png
│   ├── ground_truth_comparison.png
│   └── evaluation_report.json
│
└── tests/
    └── test_detector.py        # Automated pytest unit test suite (24 tests)
```

---

## ⚙️ Installation & Setup

### Prerequisites
* **Python 3.10 - 3.14**
* **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/copy-move-forgery-detection.git
cd copy-move-forgery-detection
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Running the Application

Launch the Streamlit web dashboard locally:
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🧪 Running Automated Tests

Run the complete test suite with pytest:
```bash
pytest -v
```

All 24 test cases verify:
* Input validation (empty arrays, corrupted files, tiny images, oversized downscaling)
* Grayscale conversion and format safety
* SIFT vs. ORB extractor initialization and fallback
* Authentic image rejection (negative test case)
* Copy-move forged image detection (positive test case)
* Visualizer shapes, color spaces, and empty mask overlays
* Evaluation metrics (Precision, Recall, F1, IoU, and empty mask edge cases)
* Sample dataset generation reproducibility

---

## 📊 Benchmark Evaluation Results

Evaluated on the synthetic dataset generated via `scripts/generate_samples.py`:

| Test Sample | Image-Level Verdict | Classification | Verified Inliers | Pixel Precision | Pixel Recall | Pixel F1 | Pixel IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Authentic Sample** (`authentic.png`) | Untampered | **True Negative (TN)** | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Forged Sample** (`forged.png`) | Tampered | **True Positive (TP)** | 185 | 0.7268 | 0.9935 | 0.8395 | 0.7234 |

---

## 📚 Technologies Used

* **Python 3.14**
* **OpenCV 5.0 (cv2)**: Feature extraction (SIFT, ORB), morphological operations, partial affine RANSAC
* **NumPy**: Matrix computations, vectorized boolean mask indexing
* **Pillow (PIL)**: Safe multi-format image decoding
* **Streamlit**: Web dashboard interface
* **Pytest**: Test automation framework

---

## 📄 License & Attribution

This project is open-source and intended for academic and research purposes as part of a **B.Tech Major Project** in Digital Image Processing and Image Forensics.
