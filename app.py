import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
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
        font-family: 'Plus Jakarta Sans', sans-serif;
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
        color: #2E3192;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0;
        text-align: center;
        letter-spacing: 2px;
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
    # 1. Impor library di dalam fungsi untuk efisiensi cache & menghindari error versi
    try:
        import tf_keras as keras
    except ImportError:
        from tensorflow import keras
    
    import pickle
    import os

    # 2. Tentukan path file (sesuai hasil 'dir model_training')
    model_path = 'model_training/sentiment_model_lstm.h5'
    tokenizer_path = 'model_training/tokenizer.pkl'

    # 3. Validasi keberadaan file secara ketat
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan: {model_path}")
        st.stop()
    
    if not os.path.exists(tokenizer_path):
        st.error(f"File tokenizer tidak ditemukan: {tokenizer_path}")
        st.stop()

    # 4. Proses muat model
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
        st.info("Saran: Pastikan file 'requirements.txt' sudah berisi 'tf_keras==2.15.0' dan 'tensorflow==2.15.0'")
        st.stop()
    
    # 5. Proses muat tokenizer
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        st.error(f"Gagal memuat tokenizer: {e}")
        st.stop()
        
    return model, tokenizer

# Eksekusi pemuatan model
model, tokenizer = load_model_ai()

# 4. FUNGSI PREDIKSI
def predict_sentiment(text, model, tokenizer):
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
st.markdown('<h1 class="gradient-text">⚖️ Analisis Sentimen Publik: Kasus Amsal Sitepu</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Aplikasi ini menggunakan model <b>Deep Learning (LSTM)</b> untuk memprediksi sentimen komentar netizen terkait kasus hukum Amsal Sitepu</p>", unsafe_allow_html=True)

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
                st.progress(int(score))

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
st.markdown('</div>', unsafe_allow_html=True)

# VISUALISASI
st.markdown("### 📈 Eksplorasi Visualisasi")
t1, t2 = st.tabs(["📊 Analisis Komprehensif", "☁️ WordCloud Data"])

with t1:
    if os.path.exists('output_visual/infografis_1x1.png'):
        st.image('output_visual/infografis_1x1.png', caption='Laporan Visualisasi LSTM', use_container_width=True)
    else:
        st.info("Visualisasi belum digenerate. Silakan jalankan training terlebih dahulu.")

with t2:
    if os.path.exists('output_visual/confusion_matrix.png'):
        st.image('output_visual/confusion_matrix.png', width=700)
    else:
        st.info("Confusion matrix belum digenerate.")

# HISTORY
st.markdown("### 🕒 Riwayat Analisis")
if st.session_state.history:
    for h in reversed(st.session_state.history[-5:]):
        with st.expander(f"Hasil: {h[1]}"):
            st.write(f"Teks: {h[0]}")
else:
    st.info("Belum ada riwayat analisis.")