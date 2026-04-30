# YogaMaster AI — CV Assignment 2

Multi-modal yoga pose recognition system supporting three model architectures:  
**A1 (Custom CNN)** · **V2 (MobileNetV2)** · **Hybrid (CNN + Pose)**

---

## Project Structure

```
CV-Assignment2/
├── app.py                             # Streamlit inference engine (main application)
├── convert_model.py                   # Phase 1: Keras → TFLite quantization utility
├── requirements.txt                   # Python dependencies
├── yoga_class_map.json                # Class index → pose name mapping (107 classes)
│
├── YogaMaster_A1_Production.tflite    # Quantized Custom CNN model (~9.8 MB)
├── YogaMaster_V2_Production.tflite    # Quantized MobileNetV2 model (~5.6 MB)
├── YogaMaster_Hybrid_Production.tflite # Quantized Hybrid CNN+Pose model (~3.2 MB)
│
├── yolo11n-pose.pt                    # YOLO11n pose weights (keypoint extraction)
└── yoga_samples/                      # Sample test images for validation
```

---

## Setup

### 1. Create & activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `tensorflow` 2.16+ bundles CPU and GPU support in a single package.  
> `tensorflow-cpu` is no longer a separate distribution for Python 3.10+.

---

## Running the Application

```bash
source .venv/bin/activate
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Model Overview

| Model | Architecture | Input | Size | Notes |
|-------|-------------|-------|------|-------|
| **A1** | Custom CNN | Image (224×224×3) | ~9.8 MB | Baseline model |
| **V2** | MobileNetV2 | Image (224×224×3) | ~5.6 MB | Transfer learning |
| **Hybrid** | CNN + Pose | Image + 34D pose vector | ~3.2 MB | Multi-modal; requires YOLO |

All three TFLite models output a **pre-softmaxed** probability distribution over 107 yoga pose classes. Do **not** apply softmax again at inference time.

---

## Hybrid Model — Dual-Input Inference

The Hybrid model takes two inputs simultaneously:

| TFLite Index | Tensor Name | Shape | Description |
|---|---|---|---|
| 0 | `serving_default_pose_input:0` | `[1, 34]` | 17 keypoints × (x, y), normalised to [0,1] |
| 1 | `serving_default_image_input:0` | `[1, 224, 224, 3]` | RGB image, pixel values in [0,1] |

Pose vectors are extracted via **YOLO11n-pose** → 17 COCO keypoints → normalised to image dimensions (`x/W`, `y/H`) → flattened to 34 floats.

---

## Quantization Script (Phase 1)

To re-convert the source Keras model (requires `hybrid_model_final_best.keras`):

```bash
source .venv/bin/activate
python convert_model.py
```

Outputs `YogaMaster_Hybrid_Production.tflite` (~3.2 MB, dynamic-range quantized).

---

## Preprocessing Spec

| Step | Detail |
|------|--------|
| Resize | `(224, 224)` using PIL |
| Normalise | `pixel / 255.0` → `float32` in `[0, 1]` |
| Batch dim | `np.expand_dims(arr, axis=0)` → `(1, 224, 224, 3)` |
| Pose norm | `x / image_width`, `y / image_height` |

---

## Assignment Checklist

- [x] Custom CNN model quantized to TFLite (A1)
- [x] MobileNetV2 model quantized to TFLite (V2)
- [x] Hybrid CNN+Pose model converted & quantized to TFLite
- [x] Streamlit app with model selector dropdown
- [x] Dual-input inference logic for Hybrid model
- [x] YOLO11 pose extraction (17 keypoints → 34D vector)
- [x] Skeletal overlay visualization for Hybrid mode
- [x] Top-5 prediction display with confidence bars
- [x] Code clean and structured for submission
