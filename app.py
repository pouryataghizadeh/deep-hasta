import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance, ImageFilter

# --- 1. SAYFA VE TASARIM AYARLARI (Professional Dark UI) ---
st.set_page_config(page_title="PHOENIX AI Diagnostic", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    
    /* Yan Menü Özelleştirme */
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #3b82f6; }
    
    /* Sonuç Kartı Tasarımı */
    .result-card {
        padding: 30px; border-radius: 20px; border: 2px solid #3b82f6;
        background: rgba(30, 41, 59, 0.9); margin: 20px 0; text-align: center;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
    }
    
    /* Görsel Etiketleri */
    .img-label { 
        color: #60a5fa; font-weight: 800; font-size: 14px; 
        text-align: center; margin-bottom: 15px; text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Başlıklar */
    h1, h2, h3 { color: #3b82f6 !important; font-family: 'Inter', sans-serif; }
    
    /* Buton Tasarımı */
    .stButton>button {
        width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white; border: none; padding: 15px; font-weight: bold;
        border-radius: 12px; transition: 0.4s;
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 15px 30px rgba(59, 130, 246, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODELLERİ GÜVENLİ YÜKLEME ---
@st.cache_resource
def load_all_diagnostic_models():
    base_path = os.path.dirname(__file__)
    # Dosya adlarını senin verdiğin isimlerle eşleştiriyoruz
    model_paths = {
        "chest": "chest_xray_pneumonia_model.h5",
        "brain": "brain_tumor_model.h5",
        "fracture": "best_fracture_detector_model.keras"
    }
    
    loaded = {}
    for key, name in model_paths.items():
        # Hem ana dizinde hem 'models/' klasöründe kontrol et
        full_path = os.path.join(base_path, name)
        alt_path = os.path.join(base_path, "models", name)
        
        target = full_path if os.path.exists(full_path) else alt_path if os.path.exists(alt_path) else None
        
        if target:
            try:
                loaded[key] = tf.keras.models.load_model(target, compile=False)
            except:
                loaded[key] = None
        else:
            loaded[key] = None
    return loaded

models = load_all_diagnostic_models()

# --- 3. GÖRSEL ANALİZ FONKSİYONLARI (Senin Özel Filtrelerin) ---

def process_chest(img_pil):
    # Sharp, Contrast, Pseudo-SPECT
    sharp = img_pil.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
    contrast = ImageEnhance.Contrast(img_pil).enhance(1.8)
    gray = np.array(img_pil.convert('L'))
    spect = np.stack([np.clip(gray*2,0,255), np.clip(gray*1.5,0,255), 255-gray], axis=-1).astype(np.uint8)
    return sharp, contrast, spect

def process_brain(img_pil):
    # Canny, Dilation, SPECT
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    dilation = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
    spect = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return edges, dilation, cv2.cvtColor(spect, cv2.COLOR_BGR2RGB)

def process_fracture(img_pil):
    # Equalize, Canny, Morph
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 100, 200)
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return enhanced, edges, morph

# --- 4. TAHMİN MANTIKLARI ---

def get_prediction(img_pil, mode):
    if mode == "Göğüs (Pnömoni)":
        target_size = (150, 150)
        model = models["chest"]
    elif mode == "Beyin Tümörü":
        target_size = (224, 224)
        model = models["brain"]
    else: # Kemik Kırığı
        target_size = (150, 150)
        model = models["fracture"]

    if model is None: return "MODEL BULUNAMADI", "0"

    # Preprocessing
    img = img_pil.convert('RGB').resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array, verbose=0)
    
    if mode == "Beyin Tümörü":
        CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
        idx = np.argmax(preds[0])
        return CLASSES[idx].upper(), f"{preds[0][idx]*100:.2f}"
    elif mode == "Göğüs (Pnömoni)":
        score = preds[0][0]
        res = "PNÖMONİ RİSKİ YÜKSEK 🔴" if score >= 0.5 else "NORMAL 🟢"
        conf = score if score >= 0.5 else (1-score)
        return res, f"{conf*100:.2f}"
    else: # Fracture
        score = preds[0][0]
        res = "KIRIK TESPİT EDİLDİ 🔴" if (100 - score*100) >= 50 else "KIRIKSIZ 🟢"
        conf = (100 - score*100) if (100 - score*100) >= 50 else (score*100)
        return res, f"{conf:.2f}"

# --- 5. ANA PANEL VE AKIŞ ---

st.sidebar.markdown("### 🩺 PHOENIX AI PANEL")
mode = st.sidebar.selectbox("Protokol Seçin", ["Göğüs (Pnömoni)", "Beyin Tümörü", "Kemik Kırığı"])

st.title(f"🏥 {mode} Analiz İstasyonu")
st.write("Derin Öğrenme Modelleri ile Otomatik Tanı ve Filtreleme")

uploaded_file = st.file_uploader("Görüntü Dosyasını Seçin", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col_img, col_res = st.columns([1.2, 1])
    
    with col_img:
        st.image(img, caption="Analiz Edilen Görüntü", width=500)
        
    with col_res:
        st.subheader("⚙️ Tanı İşlemi")
        if st.button("DERİN ANALİZİ BAŞLAT"):
            with st.spinner("Nöral Ağlar Katmanları İnceliyor..."):
                # Tahmin Al
                res_text, confidence = get_prediction(img, mode)
                
                # Görsel Analiz Al
                if mode == "Göğüs (Pnömoni)":
                    o1, o2, o3 = process_chest(img)
                    labels = ["KESKİNLEŞTİRME", "KONTRAST", "YAPAY SPECT"]
                elif mode == "Beyin Tümörü":
                    o1, o2, o3 = process_brain(img)
                    labels = ["KENAR (CANNY)", "GENİŞLEME", "SPECT HARİTASI"]
                else:
                    o1, o2, o3 = process_fracture(img)
                    labels = ["EŞİTLEME", "KENARLAR", "MORFOLOJİK"]

                # Sonuç Kartı
                st.markdown(f"""
                    <div class="result-card">
                        <h2 style='margin:0;'>{res_text}</h2>
                        <p style='font-size:20px; color:#94a3b8;'>Güven Oranı: %{confidence}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # 3'lü Görsel Analiz (Senin İstediğin)
                st.divider()
                st.subheader("🔬 İleri Filtre Katmanları")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<p class="img-label">{labels[0]}</p>', unsafe_allow_html=True)
                    st.image(o1, width=250)
                with c2:
                    st.markdown(f'<p class="img-label">{labels[1]}</p>', unsafe_allow_html=True)
                    st.image(o2, width=250)
                with c3:
                    st.markdown(f'<p class="img-label">{labels[2]}</p>', unsafe_allow_html=True)
                    st.image(o3, width=250)
