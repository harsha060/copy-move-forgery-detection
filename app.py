"""Copy-Move Image Forgery Detection - AI Forensic Dashboard.

A modern, high-contrast digital image forensics interface for analyzing
and localizing copy-move image manipulation using DIP (SIFT/ORB + RANSAC).
"""

from __future__ import annotations

import datetime
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.detector import CopyMoveDetector, DetectionResult, is_sift_available
from src.preprocessing import PreprocessedImage, preprocess_image
from src.visualizer import (
    create_binary_mask,
    create_overlay,
    draw_keypoints,
    draw_matches,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Forensic Dark Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ForensIQ | Copy-Move Forgery Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Base Dark Theme Overrides */
:root {
    --bg-main: #0B0F19;
    --bg-card: #111827;
    --bg-surface: #1E293B;
    --border-card: #334155;
    --accent-cyan: #00E5FF;
    --accent-blue: #3B82F6;
    --text-primary: #F8FAFC;
    --text-muted: #CBD5E1;
    --color-danger: #EF4444;
    --color-success: #10B981;
    --color-warning: #F59E0B;
}

/* 1. Header & Top Bar - Remove white strip */
[data-testid="stDecoration"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background-color: #0B0F19 !important;
    background: #0B0F19 !important;
    border-bottom: 1px solid #1E293B !important;
    color: #F8FAFC !important;
}

header[data-testid="stHeader"] svg {
    fill: #F8FAFC !important;
    color: #F8FAFC !important;
}

header[data-testid="stHeader"] button {
    color: #F8FAFC !important;
    background: transparent !important;
}

/* 2. Background & Main Container */
.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 3. Universal High-Contrast Text Labels */
label,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: #F8FAFC !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    opacity: 1 !important;
}

.stMarkdown p,
.stMarkdown span {
    color: #CBD5E1;
}

/* 4. Sidebar Customization & Text Visibility */
[data-testid="stSidebar"] {
    background-color: #0E1424 !important;
    border-right: 1px solid #1E293B !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div {
    color: #CBD5E1 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    opacity: 1 !important;
}

/* 5. Sliders (Labels, Values, Tracks, Handles) */
div[data-testid="stThumbValue"],
[data-testid="stSidebar"] div[data-testid="stThumbValue"] {
    color: #00E5FF !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
}

div[data-testid="stSliderTickBarMin"],
div[data-testid="stSliderTickBarMax"],
div[data-testid="stSliderTickBar"] span {
    color: #94A3B8 !important;
    font-weight: 500 !important;
}

div[data-baseweb="slider"] div[role="slider"] {
    background-color: #00E5FF !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.6) !important;
}

/* 6. Selectbox (SIFT / ORB Dropdown) */
div[data-baseweb="select"] > div {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: #00E5FF !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] p {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] svg {
    fill: #00E5FF !important;
}

div[data-baseweb="popover"],
ul[data-baseweb="menu"] {
    background-color: #111827 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

li[role="option"] {
    background-color: #111827 !important;
    color: #F8FAFC !important;
    font-weight: 500 !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #1E293B !important;
    color: #00E5FF !important;
}

/* 7. Number Input */
div[data-baseweb="input"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] input {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* 8. Expander (Geometric Verification) */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    margin-top: 12px !important;
}

[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] summary:hover {
    color: #00E5FF !important;
}

[data-testid="stExpander"] summary svg {
    fill: #00E5FF !important;
    color: #00E5FF !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    border-top: 1px solid #1E293B !important;
    padding-top: 14px !important;
}

/* 9. Image Upload Area & Dropzone */
[data-testid="stFileUploader"] {
    background-color: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #162032 !important;
    border: 2px dashed #334155 !important;
    border-radius: 10px !important;
    padding: 26px 20px !important;
    text-align: center !important;
    transition: border-color 0.2s ease, background-color 0.2s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #00E5FF !important;
    background-color: #1B2942 !important;
}

[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] p {
    color: #F8FAFC !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploaderDropzone"] small {
    color: #94A3B8 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #38BDF8 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4) !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, #0369A1 0%, #0284C7 100%) !important;
    border-color: #00E5FF !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(0, 229, 255, 0.4) !important;
}

[data-testid="stFileUploaderFile"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #F8FAFC !important;
}

[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] div {
    color: #F8FAFC !important;
}

/* 10. Buttons */
button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #0284C7 0%, #00B4D8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #00E5FF !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(0, 180, 216, 0.35) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #00B4D8 0%, #0284C7 100%) !important;
    box-shadow: 0 6px 20px rgba(0, 229, 255, 0.5) !important;
    transform: translateY(-1px) !important;
    color: #FFFFFF !important;
}

button[kind="primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled {
    background: #1E293B !important;
    color: #64748B !important;
    border: 1px solid #334155 !important;
    box-shadow: none !important;
}

button[kind="secondary"],
button[data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] button {
    background: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
}

button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stDownloadButton"] button:hover {
    background: #2D3E56 !important;
    border-color: #00E5FF !important;
    color: #00E5FF !important;
    box-shadow: 0 2px 10px rgba(0, 229, 255, 0.2) !important;
}

/* 11. Tabs */
[data-testid="stTabs"] {
    border-bottom: 1px solid #1E293B !important;
}

[data-testid="stTabs"] button[role="tab"] {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 18px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}

[data-testid="stTabs"] button[role="tab"]:hover {
    color: #F8FAFC !important;
    border-bottom: 2px solid #334155 !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #00E5FF !important;
    border-bottom: 2px solid #00E5FF !important;
}

[data-testid="stTabs"] [data-testid="stTabContent"] {
    padding-top: 16px !important;
}

/* 12. Tooltips */
[data-testid="stTooltipIcon"] svg {
    fill: #00E5FF !important;
}

div[data-baseweb="tooltip"] {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
}

/* Header Banner */
.forensic-header {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
    border: 1px solid #1E293B;
    border-left: 4px solid var(--accent-cyan);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.forensic-header h1 {
    color: #FFFFFF;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.forensic-header p.subtitle {
    color: var(--accent-cyan);
    font-size: 1.05rem;
    font-weight: 500;
    margin: 0 0 12px 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.forensic-header p.description {
    color: #CBD5E1;
    font-size: 0.95rem;
    line-height: 1.5;
    margin: 0;
    max-width: 900px;
}

/* Status Pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.status-online {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.status-tamper {
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.status-authentic {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

/* Metric Cards */
.metric-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: left;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.metric-card:hover {
    border-color: var(--accent-cyan);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.78rem;
    color: #CBD5E1;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #94A3B8;
    margin-top: 4px;
}

/* Footer */
.forensic-footer {
    text-align: center;
    padding: 28px 10px 16px 10px;
    color: #94A3B8;
    font-size: 0.85rem;
    border-top: 1px solid #1E293B;
    margin-top: 40px;
}

/* File Upload info */
.uploaded-info {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #F8FAFC;
    margin-top: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------
def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Convert an RGB or Grayscale numpy array to PNG bytes."""
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def get_sample_images_dir() -> Path:
    """Return the Path to the local samples directory."""
    return Path(__file__).parent / "samples"


def load_image_bytes(image_file: io.BytesIO | bytes | Path) -> Tuple[np.ndarray, str, int]:
    """Safely decode image file into standard RGB numpy array, filename, and byte size."""
    if isinstance(image_file, Path):
        name = image_file.name
        raw_bytes = image_file.read_bytes()
    else:
        name = getattr(image_file, "name", "uploaded_image.png")
        if isinstance(image_file, bytes):
            raw_bytes = image_file
        else:
            raw_bytes = image_file.getvalue()

    file_size = len(raw_bytes)
    # Decode with PIL to handle PNG, JPG, TIFF, WebP, etc.
    pil_img = Image.open(io.BytesIO(raw_bytes))
    rgb_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    return rgb_arr, name, file_size


# -----------------------------------------------------------------------------
# 3. Sidebar Configuration & Controls
# -----------------------------------------------------------------------------
def render_sidebar() -> Dict[str, Any]:
    """Render the sidebar controls and return analysis configuration."""
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0 16px 0;">
                <span style="font-size: 2rem;">🛡️</span>
                <div>
                    <h3 style="margin: 0; color: #FFFFFF; font-size: 1.25rem;">ForensIQ</h3>
                    <span style="font-size: 0.75rem; color: #00E5FF; letter-spacing: 0.05em; font-weight: 600;">CMFD ENGINE v1.0</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sift_status = "Available" if is_sift_available() else "Unavailable (ORB Only)"
        st.markdown(
            f"""
            <div class="status-pill status-online" style="margin-bottom: 20px;">
                <span>●</span> SYSTEM OPERATIONAL | SIFT {sift_status}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### ⚙️ Detection Parameters")

        # Feature Extraction Method
        method_options = ["auto", "sift", "orb"]
        method_index = 0
        selected_method = st.selectbox(
            "Feature Extractor",
            options=method_options,
            index=method_index,
            help=(
                "SIFT: Scale & rotation invariant floating-point descriptors. Best for general forensics.\n"
                "ORB: Fast binary descriptors (Hamming distance). Low memory footprint.\n"
                "Auto: Dynamically selects SIFT if available, else ORB."
            ),
        )

        # Feature count
        n_features = st.slider(
            "Max Keypoints",
            min_value=500,
            max_value=5000,
            value=2000,
            step=250,
            help="Higher count captures finer textures but increases analysis time.",
        )

        # Matching ratio threshold
        ratio_threshold = st.slider(
            "Lowe's Ratio Threshold",
            min_value=0.50,
            max_value=0.90,
            value=0.75,
            step=0.01,
            help=(
                "Ratio between 1st and 2nd distant neighbor descriptors. "
                "Lower values (e.g. 0.65) enforce stricter match uniqueness."
            ),
        )

        # Spatial distance threshold
        min_spatial_distance = st.slider(
            "Min Spatial Distance (px)",
            min_value=15,
            max_value=120,
            value=40,
            step=5,
            help=(
                "Minimum pixel distance required between matched points. "
                "Filters out adjacent pixels belonging to the same local texture."
            ),
        )

        # Geometric Verification (RANSAC)
        with st.expander("📐 Geometric Verification (RANSAC)", expanded=False):
            min_matches_for_ransac = st.number_input(
                "Min Candidate Matches for RANSAC",
                min_value=4,
                max_value=20,
                value=4,
                help="Minimum candidates required before attempting geometric consensus.",
            )
            min_inliers_for_forgery = st.slider(
                "Min Inliers for Forgery Verdict",
                min_value=4,
                max_value=25,
                value=6,
                step=1,
                help="Minimum geometrically consistent inliers needed to flag a forgery.",
            )
            ransac_reproj_threshold = st.slider(
                "RANSAC Error Tolerance (px)",
                min_value=1.0,
                max_value=12.0,
                value=5.0,
                step=0.5,
                help="Maximum allowed reprojection error for affine model consensus.",
            )
            mask_dilation_radius = st.slider(
                "Mask Dilation Radius (px)",
                min_value=10,
                max_value=50,
                value=25,
                step=5,
                help="Radius around verified inliers used when painting suspicious mask regions.",
            )

        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset Parameters to Defaults", use_container_width=True):
            st.rerun()

        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 15px; line-height: 1.45;">
                <strong style="color: #CBD5E1;">About CMFD Engine:</strong><br>
                Extracts invariant keypoints, executes self-referential k-NN matching, and eliminates
                natural spatial proximity before verifying affine displacement via RANSAC.
            </div>
            """,
            unsafe_allow_html=True,
        )

        return {
            "method": selected_method,
            "n_features": n_features,
            "ratio_threshold": ratio_threshold,
            "min_spatial_distance": float(min_spatial_distance),
            "min_matches_for_ransac": int(min_matches_for_ransac),
            "min_inliers_for_forgery": int(min_inliers_for_forgery),
            "ransac_reproj_threshold": float(ransac_reproj_threshold),
            "mask_dilation_radius": int(mask_dilation_radius),
        }


# -----------------------------------------------------------------------------
# 4. Header Section
# -----------------------------------------------------------------------------
def render_header():
    """Render the top forensic dashboard header."""
    st.markdown(
        """
        <div class="forensic-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1>Copy-Move Forgery Detection</h1>
                    <p class="subtitle">AI-Powered Digital Image Forensics</p>
                </div>
                <div class="status-pill status-online">
                    <span>●</span> DIP Engine Ready
                </div>
            </div>
            <p class="description">
                Specialized Digital Image Processing (DIP) forensic system designed to uncover copy-paste tampering.
                The system detects cloned textures using keypoint descriptors (SIFT/ORB), filters spatial proximity,
                and confirms authentic geometric shifts using partial-affine RANSAC verification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 5. Image Upload & Selection Section
# -----------------------------------------------------------------------------
def render_image_input_section() -> Optional[Tuple[np.ndarray, str, int]]:
    """Render image uploader and sample selectors, returning the active image if loaded."""
    samples_dir = get_sample_images_dir()
    auth_sample_path = samples_dir / "authentic.png"
    forged_sample_path = samples_dir / "forged.png"

    # Tabbed selection: Custom Upload vs Preset Samples
    input_tab1, input_tab2 = st.tabs(["📤 Upload Image File", "🧪 Load Benchmark Samples"])

    active_image = None
    active_name = ""
    active_size = 0

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Select an image file to analyze (drag & drop supported)",
            type=["png", "jpg", "jpeg", "tif", "tiff", "webp"],
            help="Supported formats: PNG, JPG/JPEG, TIFF, WebP.",
            key="user_file_uploader",
        )
        if uploaded_file is not None:
            if "preset_image" in st.session_state:
                del st.session_state["preset_image"]
            try:
                active_image, active_name, active_size = load_image_bytes(uploaded_file)
            except Exception as exc:
                st.error(f"❌ Failed to decode image: {exc}")
                return None

    with input_tab2:
        st.markdown(
            "Quickly test the forensic pipeline using bundled benchmark test images:"
        )
        col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
        with col_s1:
            if st.button("📄 Load Authentic Sample", use_container_width=True):
                if auth_sample_path.exists():
                    st.session_state["preset_image"] = ("authentic", auth_sample_path)
                    st.rerun()
                else:
                    st.warning("Sample authentic.png not found in samples/ directory.")
        with col_s2:
            if st.button("⚠️ Load Forged Sample", use_container_width=True):
                if forged_sample_path.exists():
                    st.session_state["preset_image"] = ("forged", forged_sample_path)
                    st.rerun()
                else:
                    st.warning("Sample forged.png not found in samples/ directory.")
        with col_s3:
            if "preset_image" in st.session_state:
                p_label, p_path = st.session_state["preset_image"]
                st.info(f"Loaded Preset: **{p_path.name}**")

        if uploaded_file is None and "preset_image" in st.session_state:
            _, p_path = st.session_state["preset_image"]
            try:
                active_image, active_name, active_size = load_image_bytes(p_path)
            except Exception as exc:
                st.error(f"Failed to load preset: {exc}")

    # Display active image metadata card & controls
    if active_image is not None:
        h, w = active_image.shape[:2]
        size_kb = active_size / 1024.0

        st.markdown(
            f"""
            <div class="uploaded-info">
                <div>
                    <strong style="color: #00E5FF;">Active File:</strong> <code style="color: #F8FAFC;">{active_name}</code> &nbsp;|&nbsp;
                    <strong style="color: #00E5FF;">Resolution:</strong> {w} × {h} px &nbsp;|&nbsp;
                    <strong style="color: #00E5FF;">Size:</strong> {size_kb:.1f} KB &nbsp;|&nbsp;
                    <strong style="color: #00E5FF;">Channels:</strong> RGB
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_clr1, col_clr2 = st.columns([1, 5])
        with col_clr1:
            if st.button("🗑️ Clear Image", use_container_width=True):
                if "preset_image" in st.session_state:
                    del st.session_state["preset_image"]
                if "detection_result" in st.session_state:
                    del st.session_state["detection_result"]
                st.rerun()

        return active_image, active_name, active_size

    # Empty State Guide
    st.markdown(
        """
        <div style="background: rgba(17, 24, 39, 0.7); border: 2px dashed #334155; border-radius: 12px; padding: 36px 20px; text-align: center; margin-top: 15px;">
            <span style="font-size: 2.5rem; display: block; margin-bottom: 8px;">📷</span>
            <h4 style="margin: 0 0 6px 0; color: #F8FAFC;">No Image Selected for Analysis</h4>
            <p style="color: #CBD5E1; font-size: 0.9rem; margin: 0 auto; max-width: 500px;">
                Upload a JPEG, PNG, or TIFF image above, or choose a benchmark test sample to begin
                tampering localization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return None


# -----------------------------------------------------------------------------
# 6. Analysis Execution
# -----------------------------------------------------------------------------
def run_analysis(image_rgb: np.ndarray, config: Dict[str, Any]) -> Tuple[PreprocessedImage, DetectionResult]:
    """Execute preprocessing and detection using the real DIP algorithms."""
    prep = preprocess_image(image_rgb)
    detector = CopyMoveDetector(
        method=config["method"],
        n_features=config["n_features"],
        ratio_threshold=config["ratio_threshold"],
        min_spatial_distance=config["min_spatial_distance"],
        min_matches_for_ransac=config["min_matches_for_ransac"],
        min_inliers_for_forgery=config["min_inliers_for_forgery"],
        ransac_reproj_threshold=config["ransac_reproj_threshold"],
        mask_dilation_radius=config["mask_dilation_radius"],
    )
    result = detector.detect(prep)
    return prep, result


# -----------------------------------------------------------------------------
# 7. Metrics & Results Dashboard
# -----------------------------------------------------------------------------
def render_results(
    image_name: str,
    prep: PreprocessedImage,
    result: DetectionResult,
    config: Dict[str, Any],
):
    """Render the forensic metric cards, visual tabs, and download buttons."""
    st.markdown("## 📊 Forensic Investigation Results")

    # 1. Verdict Banner
    if result.is_forgery:
        conf_pct = result.confidence_score * 100.0
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.5); border-left: 5px solid #EF4444; border-radius: 10px; padding: 18px 24px; margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.8rem;">🚨</span>
                        <div>
                            <h3 style="margin: 0; color: #FCA5A5; font-size: 1.25rem;">TAMPERING DETECTED: Copy-Move Forgery Confirmed</h3>
                            <p style="margin: 2px 0 0 0; color: #F8FAFC; font-size: 0.9rem;">
                                Found <strong>{result.num_verified_matches}</strong> geometrically verified matching inliers matching an affine transformation.
                            </p>
                        </div>
                    </div>
                    <div class="status-pill status-tamper" style="font-size: 0.9rem; padding: 6px 14px;">
                        CONFIDENCE: {conf_pct:.1f}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Check if warning exists
        has_low_feature = any("Low keypoint count" in w for w in result.diagnostic_warnings)
        if has_low_feature:
            st.markdown(
                """
                <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.5); border-left: 5px solid #F59E0B; border-radius: 10px; padding: 18px 24px; margin-bottom: 24px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.8rem;">⚠️</span>
                        <div>
                            <h3 style="margin: 0; color: #FCD34D; font-size: 1.25rem;">LOW TEXTURE / INCONCLUSIVE</h3>
                            <p style="margin: 2px 0 0 0; color: #F8FAFC; font-size: 0.9rem;">
                                The image lacks sufficient corner textures or distinct gradient features to reliably detect duplication.
                            </p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.5); border-left: 5px solid #10B981; border-radius: 10px; padding: 18px 24px; margin-bottom: 24px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 1.8rem;">🛡️</span>
                            <div>
                                <h3 style="margin: 0; color: #6EE7B7; font-size: 1.25rem;">NO COPY-MOVE TAMPERING DETECTED</h3>
                                <p style="margin: 2px 0 0 0; color: #F8FAFC; font-size: 0.9rem;">
                                    No statistically significant duplicated regions met the geometric verification criteria.
                                </p>
                            </div>
                        </div>
                        <div class="status-pill status-authentic" style="font-size: 0.9rem; padding: 6px 14px;">
                            STATUS: NO FORGERY DETECTED
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 2. Metric Cards Grid
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Verdict</div>
                <div class="metric-value" style="color: {'#EF4444' if result.is_forgery else '#10B981'}; font-size: 1.25rem;">
                    {'FORGERY' if result.is_forgery else 'UNTAMPERED'}
                </div>
                <div class="metric-sub">Geometric verdict</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        conf_display = f"{result.confidence_score * 100:.1f}%" if result.is_forgery else "0.0%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" style="color: #00E5FF;">{conf_display}</div>
                <div class="metric-sub">Inlier consensus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Keypoints</div>
                <div class="metric-value">{result.num_keypoints:,}</div>
                <div class="metric-sub">Points extracted</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Candidates</div>
                <div class="metric-value">{result.num_candidate_matches}</div>
                <div class="metric-sub">Distance-filtered</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Verified Inliers</div>
                <div class="metric-value" style="color: {'#EF4444' if result.is_forgery else '#F8FAFC'};">
                    {result.num_verified_matches}
                </div>
                <div class="metric-sub">RANSAC consensus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        elapsed_ms = result.execution_time_seconds * 1000.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Latency / Engine</div>
                <div class="metric-value" style="font-size: 1.25rem;">{elapsed_ms:.0f} ms</div>
                <div class="metric-sub">Method: {result.selected_method}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3. Generate Visuals
    vis_overlay = create_overlay(prep.original_rgb, result.binary_mask)
    vis_matches = draw_matches(
        prep.original_rgb,
        result.verified_matches,
        candidate_matches=result.candidate_matches,
        show_candidates=False,
    )
    vis_keypoints = draw_keypoints(prep.original_rgb, result.keypoints_coords)

    # 4. Multi-View Workspace Tabs
    tab_view1, tab_view2, tab_view3, tab_view4, tab_view5 = st.tabs(
        [
            "🔍 Forensic Overlay & Comparison",
            "🔗 Feature Match Vectors",
            "🎯 Predicted Binary Mask",
            "📍 Keypoints Distribution",
            "📋 Diagnostic Audit Trail",
        ]
    )

    with tab_view1:
        st.markdown("#### Side-by-Side Forensic Inspection")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.image(prep.original_rgb, caption="Original Input Image", use_container_width=True)
        with col_v2:
            st.image(
                vis_overlay,
                caption="Suspicious Region Overlay (Red Highlight + Yellow Contour)",
                use_container_width=True,
            )

    with tab_view2:
        st.markdown("#### Feature Match Displacement Vectors")
        st.markdown(
            "Green lines indicate verified keypoint pairs matching the same physical displacement vector."
        )
        st.image(vis_matches, caption="Verified Match Lines (RANSAC Inliers)", use_container_width=True)

    with tab_view3:
        st.markdown("#### Predicted Binary Localization Mask")
        st.markdown("White (255) pixels represent localized suspicious regions. Black (0) represents authentic background.")
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            st.image(result.binary_mask, caption="Binary Localization Mask (HxW)", use_container_width=True)
        with col_m2:
            mask_px = int((result.binary_mask > 0).sum())
            total_px = result.binary_mask.size
            pct_tampered = (mask_px / max(total_px, 1)) * 100.0
            st.markdown(
                f"""
                <div class="metric-card" style="margin-top: 10px;">
                    <div class="metric-label">Mask Coverage</div>
                    <div class="metric-value">{mask_px:,} px</div>
                    <div class="metric-sub">{pct_tampered:.2f}% of total canvas</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_view4:
        st.markdown("#### Extracted Keypoints Distribution")
        st.image(
            vis_keypoints,
            caption=f"All {result.num_keypoints} Keypoints Extracted using {result.selected_method}",
            use_container_width=True,
        )

    with tab_view5:
        st.markdown("#### Diagnostic Audit Trail & Parameters")
        if result.diagnostic_warnings:
            st.markdown("##### ⚠️ Diagnostic Notes")
            for w in result.diagnostic_warnings:
                st.info(f"• {w}")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("##### 📐 Estimated Transformation Matrix (Affine Partial)")
            if result.transformation_matrix is not None:
                st.code(str(np.round(result.transformation_matrix, 4)))
            else:
                st.text("No geometric model fitted (insufficient inliers).")

        with col_d2:
            st.markdown("##### ⚙️ Applied Parameters")
            st.json(result.parameters)

    # 5. Downloads Section
    st.markdown("---")
    st.markdown("### 💾 Export Forensic Artifacts")
    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

    # Prepare download payloads
    mask_bytes = image_to_png_bytes(result.binary_mask)
    overlay_bytes = image_to_png_bytes(vis_overlay)
    matches_bytes = image_to_png_bytes(vis_matches)

    # JSON report
    report_dict = {
        "project": "Copy-Move Image Forgery Detection",
        "timestamp": datetime.datetime.now().isoformat(),
        "input_filename": image_name,
        "resolution": {
            "width": prep.original_rgb.shape[1],
            "height": prep.original_rgb.shape[0],
            "channels": prep.original_rgb.shape[2],
        },
        "verdict": {
            "is_forgery": result.is_forgery,
            "confidence_score": round(result.confidence_score, 4),
            "status": "FORGERY_DETECTED" if result.is_forgery else "AUTHENTIC_OR_INCONCLUSIVE",
        },
        "statistics": {
            "keypoints_detected": result.num_keypoints,
            "candidate_matches": result.num_candidate_matches,
            "verified_inliers": result.num_verified_matches,
            "execution_time_seconds": round(result.execution_time_seconds, 4),
            "selected_method": result.selected_method,
        },
        "transformation_matrix": (
            result.transformation_matrix.tolist() if result.transformation_matrix is not None else None
        ),
        "parameters": result.parameters,
        "warnings": result.diagnostic_warnings,
    }
    report_bytes = json.dumps(report_dict, indent=2).encode("utf-8")

    with col_dl1:
        st.download_button(
            label="📥 Download Mask (PNG)",
            data=mask_bytes,
            file_name=f"{image_name}_predicted_mask.png",
            mime="image/png",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            label="📥 Download Overlay (PNG)",
            data=overlay_bytes,
            file_name=f"{image_name}_forensic_overlay.png",
            mime="image/png",
            use_container_width=True,
        )
    with col_dl3:
        st.download_button(
            label="📥 Download Matches (PNG)",
            data=matches_bytes,
            file_name=f"{image_name}_feature_matches.png",
            mime="image/png",
            use_container_width=True,
        )
    with col_dl4:
        st.download_button(
            label="📄 Download Report (JSON)",
            data=report_bytes,
            file_name=f"{image_name}_forensic_report.json",
            mime="application/json",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# 8. Main Application Controller
# -----------------------------------------------------------------------------
def main():
    render_header()
    config = render_sidebar()

    img_data = render_image_input_section()

    if img_data is not None:
        image_rgb, image_name, _ = img_data

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Prominent Analyze Button
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            analyze_clicked = st.button(
                "🚀 Run Forensic Analysis",
                type="primary",
                use_container_width=True,
                help="Execute keypoint matching and RANSAC verification on the active image.",
            )

        # If analyze button clicked or result already stored for this image
        if analyze_clicked:
            with st.spinner("🔍 Performing keypoint extraction, self-matching, and RANSAC verification..."):
                prep, result = run_analysis(image_rgb, config)
                st.session_state["detection_result"] = (image_name, prep, result)

        if "detection_result" in st.session_state:
            cached_name, cached_prep, cached_result = st.session_state["detection_result"]
            # Verify cached result matches current image
            if cached_name == image_name:
                render_results(cached_name, cached_prep, cached_result, config)

    # Footer
    st.markdown(
        """
        <div class="forensic-footer">
            <strong>B.Tech Major Project: Copy-Move Image Forgery Detection Using Digital Image Processing</strong><br>
            Engineered with Python 3.14 • OpenCV 5.0 • SIFT / ORB • RANSAC Consensus • Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
