import os
import glob
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

st.set_page_config(page_title="Yoga Semantic Search", page_icon="🧘", layout="wide")

# Custom CSS for beautiful UI
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e1b4b 100%); color: white; }
  .hero-title { font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #a78bfa, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
  .hero-sub { text-align: center; color: #cbd5e1; font-size: 1.1rem; margin-bottom: 2rem; }
  .card { background: rgba(255,255,255,0.1); border-radius: 12px; padding: 1rem; backdrop-filter: blur(10px); }
  [data-testid="stFileUploader"] { border: 2px dashed rgba(167,139,250,0.5) !important; border-radius: 16px !important; background: rgba(167,139,250,0.05) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">Yoga Semantic Search</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Find similar yoga poses based on an image query using deep visual embeddings.</p>', unsafe_allow_html=True)

@st.cache_resource
def load_feature_extractor():
    # Load MobileNetV2 without the classification head for feature extraction
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return base_model

def get_embedding(img: Image.Image, model) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features[0]

@st.cache_data
def load_database_embeddings():
    model = load_feature_extractor()
    db_paths = glob.glob("../yoga_samples/*.jpg")
    db_embeddings = []
    valid_paths = []
    
    for path in db_paths:
        try:
            img = Image.open(path)
            emb = get_embedding(img, model)
            db_embeddings.append(emb)
            valid_paths.append(path)
        except Exception as e:
            pass
            
    if len(db_embeddings) > 0:
        return np.array(db_embeddings), valid_paths
    else:
        return np.empty((0, 1280)), [] # MobileNetV2 avg pooling output size

# Load database
with st.spinner("Loading database embeddings..."):
    db_embeddings, db_paths = load_database_embeddings()
    model = load_feature_extractor()

# Upload query
uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('### Query Image')
        st.image(query_img, use_container_width=True)
        
    with st.spinner("Extracting features and searching..."):
        query_emb = get_embedding(query_img, model)
        
        if len(db_embeddings) > 0:
            similarities = cosine_similarity([query_emb], db_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:6] # top 6 results
            
            with col2:
                st.markdown('### Top Similar Poses')
                cols = st.columns(3)
                for i, idx in enumerate(top_indices):
                    sim_score = similarities[idx]
                    match_path = db_paths[idx]
                    match_img = Image.open(match_path)
                    pose_name = os.path.basename(match_path).replace(".jpg", "").replace("sample_", "").replace("_", " ").title()
                    # Remove number from pose name if present
                    pose_name = ''.join([i for i in pose_name if not i.isdigit()]).strip()
                    
                    with cols[i % 3]:
                        st.markdown(f'<div class="card">', unsafe_allow_html=True)
                        st.image(match_img, use_container_width=True)
                        st.markdown(f"<p style='text-align:center; font-weight:bold; margin-top:0.5rem;'>{pose_name}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align:center; font-size:0.8rem; color:#a78bfa;'>Sim: {sim_score:.2f}</p>", unsafe_allow_html=True)
                        st.markdown(f'</div>', unsafe_allow_html=True)
        else:
            with col2:
                st.warning("No images found in the database (../yoga_samples/) to search against.")
