# Rehydrate Prompt - YogaFusion-CV Semantic Search

> 複製以下內容貼給新的 AI Agent，即可快速恢復上下文。

---

## 專案：Yoga Pose Semantic Search

### 你現在在哪个目錄
```
/Users/ryan/Antigravity/YogaFusion-CV/semantic_search/
```

### 關鍵環境設定
- **必須使用** `.venv` 虛擬環境（位於專案根目錄 `/Users/ryan/Antigravity/YogaFusion-CV/.venv/`）
- **不能使用** base conda (`/opt/anaconda3/bin/python`) — 會 segfault
- 所有指令前面加 `../.venv/bin/python`

### 已完成的工作
1. **Phase 1** — `embedding_engine.py`：TFLite 特徵提取引擎，支援 a1/v2/hybrid 三種模型
2. **Phase 2** — `db_initializer.py`：Qdrant Docker 管理 + 向量資料庫初始化
3. **Streamlit UI** — `app.py`：Web 介面
4. **設定檔** — `config.yaml`, `docker-compose.yml`, `requirements.txt`

### 資料庫狀態
- `yoga_a1` collection: 5 points (已 upsert)
- `yoga_v2` collection: 5 points (已 upsert)
- `yoga_hybrid` collection: 0 points (需要 Phase 3 Pose Estimation 才能使用)

### 已知問題與修正
1. segfault → 用 `.venv` 而非 base conda
2. TFLite 非 thread-safe → `process_single_image()` 為每個執行緒建立獨立 FeatureExtractor
3. Qdrant API 變更 → 用 `delete_collection()` + `create_collection()` 替代 `recreate_collection()`

### 常用指令
```bash
cd semantic_search

# 啟動 Qdrant
../.venv/bin/python db_initializer.py --docker start

# 初始化資料庫
../.venv/bin/python db_initializer.py --model v2

# 啟動 Streamlit
../.venv/bin/python -m streamlit run app.py
```

### 下一步
- Phase 3: 加入 Pose Estimation (YOLO + 關節點提取) 支援 hybrid 模型
- Phase 4: Streamlit 搜尋功能串接 Qdrant

### 參考檔案
- `semantic_search/KNOWLEDGE_BASE.md` — 完整技術文件
- `semantic_search/AGENT_MISSION.md` — 專案任務（被 gitignore）
- `image-search/` — 結構與風格參考