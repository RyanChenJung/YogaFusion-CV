"""
app.py  ─  YogaMaster AI: Multi-modal Pose Recognition
-------------------------------------------------------
Streamlit inference engine supporting three model variants:
  • A1  – Custom CNN           (YogaMaster_A1_Production.tflite)
  • V2  – MobileNetV2          (YogaMaster_V2_Production.tflite)
  • Hybrid – CNN + Pose        (YogaMaster_Hybrid_Production.tflite)

Run:
    source .venv/bin/activate
    streamlit run app.py
"""

import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "A1 (CNN)":             "../models/YogaMaster_A1_Production.tflite",
    "V2 (MobileNetV2)":     "../models/YogaMaster_V2_Production.tflite",
    "Hybrid (CNN + Pose)":  "../models/YogaMaster_Hybrid_Production.tflite",
}
YOLO_PATH       = "../models/yolo11n-pose.pt"
CLASS_MAP_PATH  = "yoga_class_map.json"
IMG_SIZE        = (224, 224)
POSE_DIM        = 34   # 17 keypoints × (x, y)

# COCO skeleton connections for 17-keypoint model
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12),                          # torso
    (11, 12),                                  # hips
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YogaMaster AI",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1448 40%, #24243e 100%);
    min-height: 100vh;
  }

  /* Hero title */
  .hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.1rem;
    letter-spacing: -0.5px;
  }
  .hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
    font-weight: 400;
  }

  /* Metric cards */
  .metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(139,92,246,0.25); }
  .metric-label { color: #94a3b8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
  .metric-value { color: #f1f5f9; font-size: 1.6rem; font-weight: 700; }
  .metric-unit  { color: #64748b; font-size: 0.75rem; font-weight: 400; }

  /* Pose name badge */
  .pose-badge {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    border-radius: 12px;
    padding: 1rem 2rem;
    text-align: center;
    margin: 1rem 0;
  }
  .pose-badge .pose-label { color: #c4b5fd; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
  .pose-badge .pose-name  { color: #ffffff; font-size: 1.9rem; font-weight: 800; text-transform: capitalize; }

  /* Confidence bar */
  .conf-bar-wrap { margin: 0.8rem 0; }
  .conf-bar-label { display: flex; justify-content: space-between; color: #cbd5e1; font-size: 0.85rem; margin-bottom: 4px; }
  .conf-bar-bg { background: rgba(255,255,255,0.08); border-radius: 999px; height: 10px; }
  .conf-bar-fill {
    height: 10px; border-radius: 999px;
    background: linear-gradient(90deg, #7c3aed, #38bdf8);
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
  }

  /* Section headers */
  .section-header {
    color: #e2e8f0; font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 0.5rem; margin: 1.2rem 0 0.8rem 0;
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background: rgba(15,12,41,0.9) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
  }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* Skeleton overlay caption */
  .overlay-note { color: #64748b; font-size: 0.78rem; text-align: center; margin-top: 0.3rem; }

  /* Upload zone */
  [data-testid="stFileUploader"] {
    border: 2px dashed rgba(139,92,246,0.4) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    background: rgba(139,92,246,0.05) !important;
  }

  /* Spinner accent */
  .stSpinner > div { border-top-color: #7c3aed !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tflite(model_path: str):
    """Load and allocate a TFLite interpreter."""
    interpreter = TFLiteInterpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource(show_spinner=False)
def load_yolo():
    """Load YOLO11 pose model (imported lazily to avoid TF+PyTorch segfault on macOS ARM)."""
    from ultralytics import YOLO  # noqa: PLC0415 — intentional lazy import
    return YOLO(YOLO_PATH)

@st.cache_data(show_spinner=False)
def load_class_map():
    with open(CLASS_MAP_PATH, "r") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize to 224×224, normalise to [0,1], add batch dim."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


def extract_pose(pil_image: Image.Image, yolo_model) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Run YOLO pose estimation on the image.
    Returns:
        pose_vector  – (1, 34) float32 (normalised x,y for 17 kpts)
        keypoints_xy – (17, 2) pixel-space array, or None if no person
    """
    img_np = np.array(pil_image.convert("RGB"))
    h, w   = img_np.shape[:2]

    results = yolo_model(img_np, verbose=False)

    if (results and results[0].keypoints is not None
            and results[0].keypoints.xy is not None
            and len(results[0].keypoints.xy) > 0):
        # Take the first detected person
        kpts = results[0].keypoints.xy[0].cpu().numpy()   # (17, 2) in pixels
        normalised = kpts.copy()
        normalised[:, 0] /= w   # x / W
        normalised[:, 1] /= h   # y / H
        pose_vec = normalised.flatten().astype(np.float32)
        return np.expand_dims(pose_vec, axis=0), kpts     # (1,34), (17,2)

    # Fallback: zero vector
    return np.zeros((1, POSE_DIM), dtype=np.float32), None


def draw_skeleton(pil_image: Image.Image, keypoints_xy: np.ndarray | None) -> np.ndarray:
    """Overlay keypoints and skeleton lines on the image (OpenCV BGR)."""
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if keypoints_xy is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    kpts = keypoints_xy.astype(int)

    # Draw connections
    for (i, j) in SKELETON_CONNECTIONS:
        xi, yi = kpts[i]
        xj, yj = kpts[j]
        if 0 < xi < w and 0 < yi < h and 0 < xj < w and 0 < yj < h:
            cv2.line(img_bgr, (xi, yi), (xj, yj), (0, 230, 180), 2)

    # Draw keypoints
    for x, y in kpts:
        if 0 < x < w and 0 < y < h:
            cv2.circle(img_bgr, (x, y), 5, (180, 130, 255), -1)
            cv2.circle(img_bgr, (x, y), 5, (255, 255, 255), 1)

    return img_bgr


def run_single_input(interpreter, img_array: np.ndarray) -> np.ndarray:
    """Run inference for A1 / V2 (single image input).
    NOTE: All three TFLite models include a built-in Softmax output layer,
    so the returned array is already a probability distribution (sums to 1.0).
    Do NOT apply softmax again in the caller.
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]   # (num_classes,)  already softmaxed


def run_hybrid_input(interpreter, img_array: np.ndarray, pose_array: np.ndarray) -> np.ndarray:
    """Run inference for Hybrid (image + pose inputs). Matches inputs by tensor name.
    Tensor mapping (verified via get_input_details()):
      index=0  'serving_default_pose_input:0'   shape=[1, 34]
      index=1  'serving_default_image_input:0'  shape=[1, 224, 224, 3]
    NOTE: Output is already softmaxed — do NOT apply softmax in the caller.
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail in input_details:
        name = detail["name"].lower()
        if "image" in name:
            interpreter.set_tensor(detail["index"], img_array)
        elif "pose" in name:
            interpreter.set_tensor(detail["index"], pose_array)
        else:
            # Shape-based fallback (4-D → image, 2-D → pose)
            if len(detail["shape"]) == 4:
                interpreter.set_tensor(detail["index"], img_array)
            else:
                interpreter.set_tensor(detail["index"], pose_array)

    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧘 YogaMaster AI")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Model Selection</p>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        label="Choose architecture",
        options=list(MODEL_PATHS.keys()),
        index=0,
        label_visibility="collapsed",
        key="model_selector",
    )

    st.markdown('<p class="section-header">About</p>', unsafe_allow_html=True)
    model_info = {
        "A1 (CNN)": ("Custom CNN", "~9.8 MB", "Image only"),
        "V2 (MobileNetV2)": ("MobileNetV2", "~5.6 MB", "Image only"),
        "Hybrid (CNN + Pose)": ("CNN + YOLO Pose", "~3.2 MB", "Image + 34D pose vector"),
    }
    name, size, inputs = model_info[selected_model]
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:0.9rem 1rem; font-size:0.82rem; line-height:1.9;">
      <b>Architecture:</b> {name}<br>
      <b>Model size:</b> {size}<br>
      <b>Inputs:</b> {inputs}<br>
      <b>Classes:</b> 107 yoga poses
    </div>
    """, unsafe_allow_html=True)

    if selected_model == "Hybrid (CNN + Pose)":
        st.markdown("""
        <div style="margin-top:0.8rem; background:rgba(124,58,237,0.15);
             border:1px solid rgba(124,58,237,0.4); border-radius:10px;
             padding:0.8rem 1rem; font-size:0.8rem; color:#c4b5fd;">
          🦴 <b>Hybrid mode</b>: YOLO11 extracts 17 skeletal keypoints
          which are normalised and concatenated with the image embedding
          before classification.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.caption("CV Assignment 2 · YogaMaster AI")

# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">YogaMaster AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Multi-modal Yoga Pose Recognition · 107 Asanas</p>', unsafe_allow_html=True)

# Divider
st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.07); margin:0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a yoga pose image",
    type=["jpg", "jpeg", "png"],
    key="image_uploader",
    label_visibility="collapsed",
    help="Accepts JPG / PNG. For best results use a clear, full-body image.",
)

# Placeholder hint
if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:2.5rem 1rem; color:#475569; font-size:0.9rem;">
      📸  Drag & drop or click above to upload a yoga pose image.<br>
      <span style="font-size:0.78rem;">Accepts JPG · PNG</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    pil_image  = Image.open(uploaded_file)
    class_map  = load_class_map()
    is_hybrid  = selected_model == "Hybrid (CNN + Pose)"

    with st.spinner(f"Running inference with **{selected_model}** …"):
        t0 = time.perf_counter()

        # Pre-process
        img_array = preprocess_image(pil_image)

        # Load model
        interpreter = load_tflite(MODEL_PATHS[selected_model])

        # Pose extraction (Hybrid only)
        keypoints_xy = None
        if is_hybrid:
            yolo_model   = load_yolo()
            pose_array, keypoints_xy = extract_pose(pil_image, yolo_model)
            logits = run_hybrid_input(interpreter, img_array, pose_array)
        else:
            logits = run_single_input(interpreter, img_array)

        elapsed_ms = (time.perf_counter() - t0) * 1000

    # Post-process
    # Models output softmaxed probabilities directly — no further softmax needed.
    probs      = logits.astype(np.float64)          # already a prob distribution (sum ≈ 1.0)
    probs      = np.clip(probs, 0.0, 1.0)           # safety clamp against tiny negatives
    probs     /= probs.sum()                        # renormalise for floating-point safety
    top5_idx   = np.argsort(probs)[::-1][:5]
    pred_idx   = int(top5_idx[0])
    pred_name  = class_map.get(str(pred_idx), "Unknown").title()
    confidence = float(probs[pred_idx]) * 100

    # ── Layout: image columns ──────────────────────────────────────────────
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    if is_hybrid:
        col_orig, col_skel = st.columns(2, gap="medium")
    else:
        col_orig, col_res = st.columns([1, 1], gap="medium")

    with col_orig:
        st.markdown('<p class="section-header">📷 Input Image</p>', unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)

    if is_hybrid:
        with col_skel:
            st.markdown('<p class="section-header">🦴 Skeletal Overlay</p>', unsafe_allow_html=True)
            overlay_bgr = draw_skeleton(pil_image, keypoints_xy)
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, use_container_width=True)
            if keypoints_xy is None:
                st.markdown('<p class="overlay-note">⚠️ No person detected — zero-vector fallback used.</p>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<p class="overlay-note">17 keypoints · YOLO11n-pose</p>',
                            unsafe_allow_html=True)

    # ── Results panel ──────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pose-badge">
      <div class="pose-label">Predicted Pose</div>
      <div class="pose-name">{pred_name}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3 = st.columns(3, gap="small")
    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Confidence</div>
          <div class="metric-value">{confidence:.1f}<span class="metric-unit"> %</span></div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Inference Time</div>
          <div class="metric-value">{elapsed_ms:.0f}<span class="metric-unit"> ms</span></div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Model</div>
          <div class="metric-value" style="font-size:1rem; padding-top:4px;">{selected_model.split("(")[0].strip()}</div>
        </div>""", unsafe_allow_html=True)

    # Confidence bar for top prediction
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="conf-bar-wrap">
      <div class="conf-bar-label"><span>{pred_name}</span><span>{confidence:.1f}%</span></div>
      <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{min(confidence,100):.1f}%"></div></div>
    </div>""", unsafe_allow_html=True)

    # ── Top-5 breakdown ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📊 Top-5 Predictions</p>', unsafe_allow_html=True)
    for rank, idx in enumerate(top5_idx):
        name_i  = class_map.get(str(idx), "Unknown").title()
        prob_i  = float(probs[idx]) * 100
        bar_w   = min(prob_i, 100)
        opacity = 1.0 if rank == 0 else max(0.35, 1.0 - rank * 0.18)
        st.markdown(f"""
        <div class="conf-bar-wrap" style="opacity:{opacity:.2f}">
          <div class="conf-bar-label">
            <span>#{rank+1} &nbsp; {name_i}</span>
            <span>{prob_i:.2f}%</span>
          </div>
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{bar_w:.1f}%"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Hybrid: pose detection details ────────────────────────────────────
    if is_hybrid and keypoints_xy is not None:
        with st.expander("🦴 Pose keypoint coordinates", expanded=False):
            kpt_names = [
                "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
                "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
                "Left Wrist","Right Wrist","Left Hip","Right Hip",
                "Left Knee","Right Knee","Left Ankle","Right Ankle",
            ]
            kpt_data = {
                "Keypoint": kpt_names,
                "X (px)":   [f"{xy[0]:.1f}" for xy in keypoints_xy],
                "Y (px)":   [f"{xy[1]:.1f}" for xy in keypoints_xy],
            }
            st.table(kpt_data)
