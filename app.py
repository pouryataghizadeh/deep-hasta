import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import pickle
import joblib
import os
from PIL import Image, ImageEnhance, ImageFilter

# --- TASARIM VE CSS ---
st.set_page_config(page_title="PHOENIX AI Diagnostic", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .result-card {
        padding: 25px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);
        background: rgba(30, 41, 59, 0.8); margin-top: 20px; text-align: center;
    }
    .img-label { color: #3b82f6; font-weight: bold; font-size: 14px; margin-bottom: 10px; text-align: center; }
    .prediction-text { font-size: 24px; font-weight: bold; color: #10b981; }
    h1, h2, h3 { color: #3b82f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL TAHMİN FONKSİYONLARI ---

def predict_chest(img_pil, model):
    img = img_pil.convert('RGB').resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    if prediction >= 0.5:
        return f"Pnömoni Riski Yüksek 🔴", f"%{prediction*100:.2f}"
    else:
        return f"Normal - Pnömoni Riski Düşük 🟢", f"%{(1-prediction)*100:.2f}"

def predict_brain(img_pil, model):
    CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
    img = img_pil.convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    return f"Teşhis: {CLASSES[idx].upper()}", f"%{preds[0][idx]*100:.2f}"

def predict_fracture(img_pil, model):
    img = img_pil.convert('RGB').resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0][0]
    non_frac = preds * 100
    frac = 100 - non_frac
    if frac >= 50:
        return "KIRIK TESPİT EDİLDİ 🔴", f"%{frac:.2f}"
    else:
        return "KIRIKSIZ (NORMAL) 🟢", f"%{non_frac:.2f}"

# --- GÖRSEL ANALİZ FONKSİYONLARI ---

def apply_chest_analysis(img_pil):
    sharp = img_pil.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img_pil)
    contrast = enhancer.enhance(1.8)
    gray = np.array(img_pil.convert('L'))
    r, g, b = np.clip(gray * 2, 0, 255), np.clip(gray * 1.5, 0, 255), 255 - gray
    spect = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return sharp, contrast, spect

def apply_brain_analysis(img_pil):
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    dilation = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
    spect = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    spect = cv2.cvtColor(spect, cv2.COLOR_BGR2RGB)
    return edges, dilation, spect

def apply_fracture_analysis(img_pil):
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 100, 200)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return enhanced, edges, morph

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    # Modellerin app.py ile aynı klasörde olduğundan emin ol
    return {
        "chest": tf.keras.models.load_model("chest_xray_pneumonia_model.h5", compile=False) if os.path.exists("chest_xray_pneumonia_model.h5") else None,
        "brain": tf.keras.models.load_model("brain_tumor_model.h5", compile=False) if os.path.exists("brain_tumor_model.h5") else None,
        "fracture": tf.keras.models.load_model("best_fracture_detector_model.keras", compile=False) if os.path.exists("best_fracture_detector_model.keras") else None
    }

all_models = load_models()

# --- YAN MENÜ ---
st.sidebar.title("🩺 PHOENIX Diagnostic")
app_mode = st.sidebar.selectbox("Hastalık Seçin", 
    ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı", "Diyabet", "Kalp Sağlığı", "Meme Kanseri", "Obezite"])

# --- ANA AKIŞ ---
st.title(f"🚀 {app_mode} Tanı Sistemi")

if app_mode in ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"]:
    uploaded_file = st.file_uploader("Görüntü Yükleyin...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="Orijinal Görüntü", use_container_width=True)
            
        with col2:
            if st.button("AI ANALİZİNİ BAŞLAT"):
                st.info("Yapay Zeka derin doku analizi yapıyor...")
                
                # TAHMİN VE ANALİZ
                if app_mode == "Göğüs (Pnömoni)":
                    res, conf = predict_chest(img, all_models["chest"])
                    out1, out2, out3 = apply_chest_analysis(img)
                    labels = ["KESKİNLEŞTİRME", "KONTRAST ARTIRIMI", "PSEUDO-SPECT"]
                
                elif app_mode == "Beyin Tümörü":
                    res, conf = predict_brain(img, all_models["brain"])
                    out1, out2, out3 = apply_brain_analysis(img)
                    labels = ["KENAR TESPİTİ (CANNY)", "GENİŞLEME (DILATION)", "SPECT RENK HARİTASI"]

                elif app_mode == "Kemik Kırığı":
                    res, conf = predict_fracture(img, all_models["fracture"])
                    out1, out2, out3 = apply_fracture_analysis(img)
                    labels = ["KONTRAST EŞİTLEME", "KENAR TESPİTİ", "MORFOLOJİK KAPANIŞ"]

                # SONUÇ KARTI
                st.markdown(f"""
                <div class="result-card">
                    <p class="prediction-text">{res}</p>
                    <p>Güven Oranı: {conf}</p>
                </div>
                """, unsafe_allow_html=True)

                # 3'LÜ GÖRSEL ANALİZ
                st.divider()
                st.subheader("🔬 Detaylı Görsel Analiz Katmanları")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<p class="img-label">{labels[0]}</p>', unsafe_allow_html=True)
                    st.image(out1, use_container_width=True)
                with c2:
                    st.markdown(f'<p class="img-label">{labels[1]}</p>', unsafe_allow_html=True)
                    st.image(out2, use_container_width=True)
                with c3:
                    st.markdown(f'<p class="img-label">{labels[2]}</p>', unsafe_allow_html=True)
                    st.image(out3, use_container_width=True)

# Diyabet, Kalp vb. form tabanlı kısımlar buraya elif olarak devam eder...
