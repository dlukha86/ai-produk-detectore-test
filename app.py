import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests

# Unduh model dari Hugging Face jika belum ada
MODEL_PATH = "model_produk.h5"
MODEL_URL = "https://huggingface.co/dlukha/ai-model-produk-test/resolve/main/model_produk.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Mengunduh model..."):
        r = requests.get(MODEL_URL)
        open(MODEL_PATH, 'wb').write(r.content)

# Load model
model = load_model(MODEL_PATH)
labels = ['bakpia', 'burger', 'jamu_tradisional', 'minuman_kopi']

st.title("🧠 AI Pendeteksi Gambar Produk")

uploaded_file = st.file_uploader("Upload gambar produk...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Gambar yang diupload', use_column_width=True)

    # Preprocessing
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    label = labels[np.argmax(pred)]

    st.success(f"✅ Prediksi AI: **{label}**")
