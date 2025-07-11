import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("model_produk.h5")
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
