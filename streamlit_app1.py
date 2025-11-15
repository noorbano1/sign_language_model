import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import cv2
import os

# ---------------------------
# Paths (based on your folder)
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "sign_language_cnn_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.pkl")

# ---------------------------
# Load model and labels
# ---------------------------
model = load_model(MODEL_PATH)

labels = pickle.load(open(LABELS_PATH, "rb"))
labels = {v: k for k, v in labels.items()}  # index -> label mapping

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Sign Language Detection (A-Z, 0-9)")
st.write("Upload an image of a hand gesture to predict the sign:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array
    img_array = np.array(img)

    # Resize to match model's training input (64x64)
    img_array = cv2.resize(img_array, (64, 64))

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred))
    predicted_class = labels[class_index]

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Predicted Sign:** {predicted_class}")
    st.write(f"**Confidence:** {round(confidence * 100, 2)} %")
