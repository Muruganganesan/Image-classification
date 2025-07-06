# streamlit_app.py

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import gdown
import os

# Title
st.title("🐟 Fish Species Classifier (CNN Model)")
st.markdown("Upload a fish image and this model will predict the species.")

# Load the trained model from Google Drive
@st.cache_resource
def load_cnn_model():
    model_path = "cnn_fish_model.h5"

    # Check if model already downloaded
    if not os.path.exists(model_path):
        # Replace with your file ID
        file_id = "1lOXRwwEc2H2IJzvwvCLDvMGxv2E96t9F"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, model_path, quiet=False, fuzzy=True)

    model = load_model(model_path)
    return model
#gdown.download(url, model_path, quiet=False, fuzzy=True)
model = load_cnn_model()

# Define class names (update as per your dataset)
class_names = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel',
               'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet',
               'Trout', 'Other 1', 'Other 2', 'Other 3']  # Replace with actual class list

# Image upload
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.markdown("### 🧠 Prediction Result")
    st.success(f"Predicted Fish Species: **{predicted_class}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")
