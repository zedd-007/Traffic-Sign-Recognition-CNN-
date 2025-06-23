# streamlit_app/app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import json
from PIL import Image

# Load model and label map
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("../model/traffic_sign_cnn.h5")
    with open("../model/label_map.json", "r") as f:
        label_map = json.load(f)
    return model, label_map

model, label_map = load_model()

# Streamlit UI
st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")
st.title("🚦 Traffic Sign Recognition")
st.write("Upload an image of a traffic sign and let the model recognize it.")

uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Show result
    st.markdown("### Prediction:")
    st.write(f"**Class:** {label_map[str(predicted_class)]}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")