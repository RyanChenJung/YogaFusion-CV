# Yoga Pose Semantic Search

Deep learning-based yoga pose semantic search system. Upload a yoga pose image, and the system uses a CNN model to extract feature vectors and finds similar poses in the vector database.

## Features

- 📸 **Image Upload Search** — Upload a yoga pose image and find similar poses in real time
- 🧠 **Multi-Model Support** — Supports three pre-trained models:
  - **A1** — Custom quantized CNN model
  - **V2** — MobileNetV2 lightweight model
  - **Hybrid** — Multi-modal model (CNN + YOLO keypoints)
- 💾 **Vector Database** — Uses Qdrant to store and search feature vectors
- 🌐 **Web UI** — Beautiful interface built with Streamlit

## Quick Start

### 1. Start Qdrant Database

```bash
cd semantic_search
docker-compose up -d
```

### 2. Install Dependencies

```bash
cd semantic_search
python3 -m pip install streamlit qdrant-client pillow numpy pyyaml tensorflow
```

### 3. Initialize Database

```bash
# Initialize with V2 model
python3 db_initializer.py --model v2

# Initialize with A1 model
python3 db_initializer.py --model a1

# Initialize with Hybrid model (requires YOLO pose extraction)
python3 db_initializer.py --model hybrid
```

### 4. Start Streamlit

```bash
python3 -m streamlit run app.py --server.headless=true --server.fileWatcherType=none
```

Open http://localhost:8501 in your browser.

## File Structure

```
semantic_search/
├── app.py                   # Streamlit web application
├── embedding_engine.py      # TFLite feature extraction engine
├── db_initializer.py        # Qdrant database initializer
├── pose_extractor.py        # YOLO11 keypoint extraction
├── config.yaml              # Configuration
├── docker-compose.yml       # Qdrant Docker setup
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Feature Extraction | TensorFlow Lite (Custom CNN, MobileNetV2) |
| Vector Database | Qdrant |
| Web Framework | Streamlit |
| Pose Estimation | YOLO11n-pose |
| Containerization | Docker |

## Model Details

### A1 - Custom CNN
Custom quantized CNN model, takes 224×224 images as input and outputs 107-dimensional feature vectors.

### V2 - MobileNetV2
Lightweight MobileNetV2 model suitable for fast feature extraction.

### Hybrid - Multi-modal
Multi-modal model combining image features with YOLO keypoints. Requires YOLO pose extraction for full functionality.

## Database Collections

| Collection Name | Model | Dimensions |
|-----------------|-------|------------|
| yoga_a1 | A1 | 107 |
| yoga_v2 | V2 | 107 |
| yoga_hybrid | Hybrid | 107 |

## Notes

- Qdrant must be running before starting Streamlit
- The Hybrid model requires the `ultralytics` package to enable YOLO pose extraction
- It is recommended to prepare at least 100+ images for good search results