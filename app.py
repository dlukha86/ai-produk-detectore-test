import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests

# Download model jika belum ada
MODEL_PATH = "model_produk.h5"
MODEL_URL = "https://huggingface.co/dlukha/ai-model-produk-test/blob/main/model_produk.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Mengunduh model..."):
        r = requests.get(MODEL_URL)
        open(MODEL_PATH, 'wb').write(r.content)

# Load model
model = load_model(MODEL_PATH)
labels = ['bakpia', 'burger', 'jamu_tradisional', 'minuman_kopi']
