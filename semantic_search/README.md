# Yoga Pose Semantic Search

基於深度學習的瑜伽姿勢語意搜尋系統。上傳一張瑜伽姿勢圖片，系統會使用 CNN 模型提取特徵向量，並在向量資料庫中尋找相似的瑜伽姿勢。

## 功能

- 📸 **圖片上傳搜尋** — 上傳瑜伽姿勢圖片，即時尋找相似姿勢
- 🧠 **多模型支援** — 支援三種預訓練模型：
  - **A1** — 自訂量化 CNN 模型
  - **V2** — MobileNetV2 輕量級模型
  - **Hybrid** — 多模態模型（CNN + YOLO 關節點）
- 💾 **向量資料庫** — 使用 Qdrant 儲存與搜尋特徵向量
- 🌐 **Web UI** — Streamlit 建置的美觀介面

## 快速開始

### 1. 啟動 Qdrant 資料庫

```bash
cd semantic_search
docker-compose up -d
```

### 2. 安裝依賴

```bash
cd semantic_search
python3 -m pip install streamlit qdrant-client pillow numpy pyyaml tensorflow
```

### 3. 初始化資料庫

```bash
# 使用 V2 模型初始化
python3 db_initializer.py --model v2

# 使用 A1 模型初始化
python3 db_initializer.py --model a1

# 使用 Hybrid 模型初始化（需要 YOLO pose 提取）
python3 db_initializer.py --model hybrid
```

### 4. 啟動 Streamlit

```bash
python3 -m streamlit run app.py --server.headless=true --server.fileWatcherType=none
```

瀏覽器開啟 http://localhost:8501

## 檔案結構

```
semantic_search/
├── app.py                   # Streamlit Web 應用程式
├── embedding_engine.py      # TFLite 特徵提取引擎
├── db_initializer.py        # Qdrant 資料庫初始化
├── pose_extractor.py        # YOLO11 關節點提取
├── config.yaml              # 設定檔
├── docker-compose.yml       # Qdrant Docker 設定
├── requirements.txt         # Python 依賴
└── README.md                # 本檔案
```

## 技術棧

| 元件 | 技術 |
|------|------|
| 特徵提取 | TensorFlow Lite (Custom CNN, MobileNetV2) |
| 向量資料庫 | Qdrant |
| Web 框架 | Streamlit |
| Pose Estimation | YOLO11n-pose |
| 容器化 | Docker |

## 模型說明

### A1 - Custom CNN
自訂量化 CNN 模型，輸入 224×224 圖片，輸出 107 維特徵向量。

### V2 - MobileNetV2
輕量級 MobileNetV2 模型，適合快速特徵提取。

### Hybrid - Multi-modal
結合影像特徵與 YOLO 關節點的多模態模型。需要 YOLO pose 提取才能發揮完整效果。

## 資料庫集合

| 集合名稱 | 模型 | 維度 |
|----------|------|------|
| yoga_a1 | A1 | 107 |
| yoga_v2 | V2 | 107 |
| yoga_hybrid | Hybrid | 107 |

## 注意事項

- Qdrant 必須在啟動 Streamlit 之前運行
- Hybrid 模型需要安裝 `ultralytics` 套件以啟用 YOLO pose 提取
- 建議至少準備 100+ 張圖片以獲得良好的搜尋效果