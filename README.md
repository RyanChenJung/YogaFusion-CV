# YogaFusion-CV

Multi-modal yoga pose recognition and semantic search system. Supports three model architectures: **A1 (Custom CNN)** · **V2 (MobileNetV2)** · **Hybrid (CNN + Pose)**, with vector similarity-based semantic search functionality.

---

## Project Structure

```
YogaFusion-CV/
├── README.md                          # This file
│
├── semantic_search/                   # 🧘 Semantic Search System
│   ├── app.py                         #    Streamlit Web UI
│   ├── embedding_engine.py            #    TFLite feature extraction engine
│   ├── db_initializer.py              #    Qdrant database initializer
│   ├── pose_extractor.py              #    YOLO11 keypoint extraction
│   ├── config.yaml                    #    Configuration
│   ├── docker-compose.yml             #    Qdrant Docker setup
│   └── requirements.txt               #    Python dependencies
│
├── pose_identification/               # 🎯 Pose Identification System
│   ├── app.py                         #    Streamlit Web UI
│   ├── yoga_class_map.json            #    Pose class mapping
│   └── requirements.txt               #    Python dependencies
│
├── models/                            # 🤖 Pre-trained models
│   ├── YogaMaster_A1_Production.tflite    #    A1: Custom CNN (~9.8 MB)
│   ├── YogaMaster_V2_Production.tflite    #    V2: MobileNetV2 (~5.6 MB)
│   ├── YogaMaster_Hybrid_Production.tflite #    Hybrid: CNN+Pose (~3.2 MB)
│   ├── yolo11n-pose.pt                  #    YOLO11n pose estimation (~35 MB)
│   └── hybrid_model_final_best.keras    #    Hybrid original Keras model
│
├── yoga_samples/                      # 📸 Sample test images
├── yoga_dataset/                      # 📚 Yoga Dataset (training data)
└── semantic_search_venv/              # 🐍 Semantic search virtual environment
```

---

## Quick Start

### 1. Semantic Search System

```bash
# Start Qdrant database
cd semantic_search
docker-compose up -d

# Install dependencies
cd ../semantic_search_venv
pip install streamlit qdrant-client pillow numpy pyyaml tensorflow

# Initialize database
cd ../semantic_search
python3 db_initializer.py --model v2

# Start Streamlit
python3 -m streamlit run app.py --server.headless=true
```

Open http://localhost:8501 in your browser.

### 2. Pose Identification System

```bash
cd pose_identification
pip install -r requirements.txt
python3 -m streamlit run app.py
```

---

## Model Overview

| Model | Architecture | Input | Output Dim | Notes |
|-------|-------------|-------|------------|-------|
| **A1** | Custom CNN | Image (224×224×3) | 107 | Baseline model |
| **V2** | MobileNetV2 | Image (224×224×3) | 107 | Lightweight transfer learning |
| **Hybrid** | CNN + Pose | Image + 34D pose | 107 | Multi-modal, requires YOLO |

All TFLite models output 107-dimensional feature vectors for vector similarity search.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Feature Extraction | TensorFlow Lite (Custom CNN, MobileNetV2) |
| Pose Estimation | YOLO11n-pose (Ultralytics) |
| Vector Database | Qdrant |
| Web Framework | Streamlit |
| Containerization | Docker |

---

## Documentation

- [semantic_search/README.md](semantic_search/README.md) — Semantic Search System details
- [pose_identification/README.md](pose_identification/README.md) — Pose Identification System details
- [image-search/README.md](image-search/README.md) — Image Search System (reference)

---

## Notes

- Agent mission files (`AGENT_MISSION.md`, `REHYDRATE_PROMPT.md`) are excluded from Git
- Large model files and data directories are added to `.gitignore`
