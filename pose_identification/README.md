# Yoga Pose Identification

瑜伽姿勢識別系統，使用 YOLO11-pose 模型檢測圖片中的人體關節點並分類瑜伽姿勢。

## 功能

- 🎯 **關節點檢測** — 使用 YOLO11-pose 檢測 17 個人體關節點
- 🧘 **姿勢分類** — 根據關節點座標與姿勢對照表分類瑜伽姿勢
- 📊 **姿勢對照表** — 內建 JSON 格式的瑜伽姿勢映射表
- 🌐 **Web UI** — Streamlit 介面支援圖片上傳與即時識別

## 快速開始

### 1. 安裝依賴

```bash
cd pose_identification
pip install -r requirements.txt
```

### 2. 下載 YOLO11-pose 模型

模型位於專案根目錄的 `models/yolo11n-pose.pt`

### 3. 啟動 Streamlit

```bash
python3 -m streamlit run app.py
```

瀏覽器開啟 http://localhost:8501

## 檔案結構

```
pose_identification/
├── app.py                   # Streamlit Web 應用程式
├── requirements.txt         # Python 依賴
├── yoga_class_map.json      # 瑜伽姿勢分類對照表
└── README.md                # 本檔案
```

## 技術棧

| 元件 | 技術 |
|------|------|
| 姿勢檢測 | YOLO11n-pose (Ultralytics) |
| 姿勢分類 | 關節點座標匹配 |
| Web 框架 | Streamlit |
| 影像處理 | OpenCV, Pillow |

## 瑜伽姿勢分類

姿勢分類透過 `yoga_class_map.json` 管理，包含：
- 姿勢名稱（英文）
- 對應的關節點模式
- 備註資訊

## 注意事項

- 需要 PyTorch 環境運行 YOLO11
- 建議圖片解析度至少 640×640 以獲得最佳效果
- 單人瑜伽姿勢識別效果最佳