import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Analisis Sentimen Publik: Kasus Amsal Sitepu",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus+Jakarta+Sans', sans-serif;
    }
    
    .stApp {
        background-color: #fcfcfd;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid #f0f0f0;
    }

    .gradient-text {
        background: -webkit-linear-gradient(45deg, #2E3192, #1BFFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0;
    }

    .main-card {
        background: white;
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        border: 1px solid #f0f2f6;
        margin-bottom: 1.5rem;
    }

    .metric-box {
        text-align: center;
        padding: 1.5rem;
        background: #ffffff;
        border-radius: 20px;
        border: 1px solid #f0f2f6;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
    }

    div.stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 15px;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 172, 254, 0.4);
        color: white;
    }

    .sentiment-label {
        font-size: 1.5rem;
        font-weight: 800;
        padding: 10px 20px;
        border-radius: 12px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model_ai():
    model_path = 'model_training/sentiment_model_lstm.h5'
    tokenizer_path = 'model_training/tokenizer.pkl'

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        st.error("❌ File model atau tokenizer tidak ditemukan di folder model_training!")
        st.stop()

    try:
        # Mendefinisikan ulang arsitektur model untuk menghindari bug "quantization_config" pada Keras 3
        model = keras.models.Sequential([
            keras.layers.Input(shape=(100,)),
            keras.layers.Embedding(5000, 128),
            keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        # Hanya memuat bobot (weights) saja dari file model
        model.load_weights(model_path)
        
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        st.success("✅ Model berhasil dimuat!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        st.info("💡 Solusi:")
        st.write("1. Pastikan TensorFlow==2.16.1 dan Keras==3.3.3 sudah terinstall")
        st.write("2. Jalankan: `pip install --upgrade tensorflow==2.16.1 keras==3.3.3`")
        st.write("3. Restart aplikasi")
        st.stop()

model, tokenizer = load_model_ai()

# 4. FUNGSI PREDIKSI
def predict_sentiment(text, model, tokenizer):
    if model is None or tokenizer is None:
        st.error("❌ Model atau tokenizer tidak tersedia")
        return None, None, None
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded, verbose=0)
    labels = ['Negatif', 'Netral', 'Positif']
    result = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return result, confidence, prediction[0]

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.markdown("### 🛠️ System Engine")
    st.success("Model AI: LSTM Active")
    
    st.markdown("---")
    st.markdown("📊 **Performance Metrics**")
    st.write("**Accuracy:** 92.4%")
    st.progress(92)
    st.write("**Dataset:** 9.6K+ Komentar")
    st.write("**Latency:** < 0.5s")

# HEADER
st.markdown('<h1 class="gradient-text">Analisis Sentimen Publik: Kasus Amsal Sitepu</h1>', unsafe_allow_html=True)
st.markdown("Aplikasi ini menggunakan model **Deep Learning (LSTM)** untuk memprediksi sentimen komentar netizen terkait kasus hukum Amsal Sitepu")

# METRICS
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric-box"><small>TOTAL DATASET</small><br><h2 style="color:#2E3192">9.6K+</h2></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-box"><small>MODEL ACCURACY</small><br><h2 style="color:#764ba2">92.4%</h2></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-box"><small>PROCESSING TIME</small><br><h2 style="color:#1BFFFF">Real-time</h2></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# MAIN
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("🔍 Uji Sentimen Real-Time")
st.write("Analisis emosi dan polaritas teks dalam hitungan detik.")

# SESSION STATE
if "text" not in st.session_state:
    st.session_state.text = ""
if "run" not in st.session_state:
    st.session_state.run = False
if "history" not in st.session_state:
    st.session_state.history = []

# QUICK TRY
st.markdown("**COBA CEPAT:**")
q1, q2, q3, _ = st.columns([1,1,1,2])

if q1.button("🌟 Positif"):
    st.session_state.text = "Keputusan yang sangat adil dan transparan!"
    st.session_state.run = True

if q2.button("⚠️ Negatif"):
    st.session_state.text = "Hukum tajam ke bawah, ini tidak adil bagi rakyat."
    st.session_state.run = True

if q3.button("⚖️ Netral"):
    st.session_state.text = "Jaksa penuntut umum menghadirkan bukti baru dalam persidangan."
    st.session_state.run = True

user_input = st.text_area("Ketik komentar di bawah ini:", value=st.session_state.text, height=150)

# 🔥 CHARACTER COUNTER
st.caption(f"Panjang teks: {len(user_input)} karakter")

if st.button("JALANKAN ANALISIS SARAF"):
    st.session_state.run = True

# RUN
if st.session_state.run:
    if user_input:
        with st.spinner('Memproses Jaringan Saraf...'):
            label, score, probs = predict_sentiment(user_input, model, tokenizer)

            st.session_state.history.append((user_input, label))

            st.markdown("---")
            res_col1, res_col2 = st.columns([1,2])

            with res_col1:
                if label == 'Positif':
                    st.markdown(f'<div class="sentiment-label" style="background:#e8f5e9; color:#2e7d32;">✅ {label}</div>', unsafe_allow_html=True)
                elif label == 'Negatif':
                    st.markdown(f'<div class="sentiment-label" style="background:#ffebee; color:#c62828;">🚨 {label}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="sentiment-label" style="background:#e3f2fd; color:#1565c0;">⚖️ {label}</div>', unsafe_allow_html=True)

            with res_col2:
                st.write(f"### Tingkat Keyakinan: **{score:.2f}%**")
                # Dikonversi menjadi /100 agar aman di Streamlit (maksimum 1.0)
                st.progress(min(int(score) / 100.0, 1.0))

                # 🔥 DETAIL PROBABILITAS
                st.write("Detail Probabilitas:")
                st.write(f"Negatif: {probs[0]*100:.2f}%")
                st.write(f"Netral: {probs[1]*100:.2f}%")
                st.write(f"Positif: {probs[2]*100:.2f}%")

                # 🔥 INSIGHT
                if label == "Negatif":
                    st.warning("Teks mengandung opini negatif atau kritik.")
                elif label == "Positif":
                    st.success("Teks menunjukkan sentimen positif.")
                else:
                    st.info("Teks bersifat netral atau informatif.")

        st.session_state.run = False
    else:
        st.warning("Mohon masukkan teks terlebih dahulu.")

# VISUALISASI
st.markdown("### 📈 Eksplorasi Visualisasi")
t1, t2 = st.tabs(["📊 Analisis Komprehensif", "☁️ WordCloud Data"])

with t1:
    if os.path.exists('output_visual/infografis_1x1.png'):
        st.image('output_visual/infografis_1x1.png', caption='Laporan Visualisasi LSTM', use_container_width=True)
    else:
        st.info("Visualisasi belum digenerate.")

with t2:
    if os.path.exists('output_visual/confusion_matrix.png'):
        st.image('output_visual/confusion_matrix.png', width=700)

# Di bagian 🔥 HISTORY
st.markdown("### 🕒 Riwayat Analisis")
if st.session_state.history:
    for h in reversed(st.session_state.history[-5:]): # Gunakan reversed agar yang terbaru di atas
        with st.expander(f"Hasil: {h[1]}"):
            st.write(f"Teks: {h[0]}")
else:
    st.info("Belum ada riwayat analisis.")