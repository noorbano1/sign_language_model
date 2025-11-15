import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = load_model("sign_language_cnn_model.h5")

# Class labels (adjust according to your dataset)
class_labels = ['A', 'B', 'C', 'D', 'E']  # Example

st.title("Sign Language Detection")
st.write("Upload an image and the model will predict the sign.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = img.resize((64, 64))  # adjust size according to your CNN input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Prediction
    pred = model.predict(img_array)
    predicted_class = class_labels[np.argmax(pred)]
    
    st.write(f"Predicted Sign: **{predicted_class}**")
