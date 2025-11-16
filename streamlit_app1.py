import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import cv2
import os

# ---------------------------
# Load model and labels
# ---------------------------
MODEL_PATH = "sign_language_cnn_model.h5"
LABELS_PATH = "labels.pkl"

model = load_model(MODEL_PATH)

# Load labels.pkl (which you created in Colab)
with open(LABELS_PATH, "rb") as f:
    label_dict = pickle.load(f)

# Convert {label: index} → {index: label}
index_to_label = {v: k for k, v in label_dict.items()}

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🤟 Sign Language Detection (A–Z, 0–9)")
st.write("Upload an image of a hand gesture to predict the sign.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to array
    img_array = np.array(img)

    # Preprocess: Resize → Normalize → Expand dims
    img_array = cv2.resize(img_array, (64, 64))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred))

    predicted_class = index_to_label[class_index]

    # Show prediction
    st.subheader("Prediction Result")
    st.write(f"### Predicted Sign: **{predicted_class}**")
    st.write(f"### Confidence: **{round(confidence * 100, 2)} %**")
