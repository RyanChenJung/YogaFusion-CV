# Yoga Pose Identification

Yoga pose recognition system using YOLO11-pose to detect human body keypoints in images and classify yoga poses.

## Features

- 🎯 **Keypoint Detection** — Detects 17 body keypoints using YOLO11-pose
- 🧘 **Pose Classification** — Classifies yoga poses based on keypoint coordinates
- 📊 **Pose Mapping** — Built-in JSON pose class mapping
- 🌐 **Web UI** — Streamlit interface for image upload and real-time recognition

## Quick Start

### 1. Install Dependencies

```bash
cd pose_identification
pip install -r requirements.txt
```

### 2. YOLO11-pose Model

The model is located at the project root: `models/yolo11n-pose.pt`

### 3. Start Streamlit

```bash
python3 -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## File Structure

```
pose_identification/
├── app.py                   # Streamlit Web application
├── requirements.txt         # Python dependencies
├── yoga_class_map.json      # Pose class mapping
└── README.md                # This file
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Pose Estimation | YOLO11n-pose (Ultralytics) |
| Pose Classification | Keypoint coordinate matching |
| Web Framework | Streamlit |
| Image Processing | OpenCV, Pillow |

## Pose Classification

Pose classification is managed via `yoga_class_map.json`, which includes:
- Pose name (English)
- Corresponding keypoint patterns
- Notes

## Notes

- Requires PyTorch environment to run YOLO11
- Recommended image resolution: at least 640×640 for best results
- Single-person yoga poses work best