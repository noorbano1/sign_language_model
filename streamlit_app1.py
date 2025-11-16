import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import cv2

# ---------------------------
# Load model and labels
# ---------------------------
MODEL_PATH = "sign_language_cnn_model.h5"
LABELS_PATH = "labels.pkl"

model = load_model(MODEL_PATH)

with open(LABELS_PATH, "rb") as f:
    label_dict = pickle.load(f)

index_to_label = {v: k for k, v in label_dict.items()}

# ---------------------------
# Streamlit page config & style
# ---------------------------
st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
    }
    .prediction-box {
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("📖 Instructions")
st.sidebar.write("""
1. Upload a hand gesture image (A–Z or 0–9).  
2. The model will predict the sign and confidence.  
3. Colors indicate different predicted classes.  
4. Optional: See top 3 predictions for comparison.
""")

# ---------------------------
# Main Title
# ---------------------------
st.title("🤟 Sign Language Detection")
st.subheader("Upload a hand gesture image to predict the sign.")

# ---------------------------
# Upload image
# ---------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(img)
    img_array = cv2.resize(img_array, (64,64))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred))
    predicted_class = index_to_label[class_index]

    # Color dictionary for labels
    color_dict = {
        'A':'#FF5733','B':'#33FF57','C':'#3357FF','D':'#FF33A8','E':'#FF8F33',
        'F':'#33FFF0','G':'#DA33FF','H':'#FFC733','I':'#33FF99','J':'#9933FF',
        'K':'#FF3333','L':'#33CCFF','M':'#FF33F6','N':'#33FFBD','O':'#FF8333',
        'P':'#33FF66','Q':'#FF33B5','R':'#FFAA33','S':'#33D1FF','T':'#FF3355',
        'U':'#33FFCC','V':'#FF33CC','W':'#33FF33','X':'#FF9933','Y':'#33B5FF',
        '0':'#FF6666','1':'#66FF33','2':'#6633FF','3':'#FF33AA','4':'#33FF88',
        '5':'#FF3380','6':'#33FF44','7':'#FF9933','8':'#33CC99','9':'#FF6633'
    }

    label_color = color_dict.get(predicted_class, "#000000")

    # ---------------------------
    # Display Prediction
    # ---------------------------
    st.markdown(f"<div class='prediction-box' style='background-color:{label_color}; color:white;'>Predicted Sign: {predicted_class}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='prediction-box' style='background-color:#4682B4; color:white;'>Confidence: {round(confidence*100,2)}%</div>", unsafe_allow_html=True)

    # ---------------------------
    # Top 3 Predictions
    # ---------------------------
    top_indices = pred[0].argsort()[-3:][::-1]
    st.subheader("🏆 Top 3 Predictions")
    cols = st.columns(3)
    for i, idx in enumerate(top_indices):
        cls = index_to_label[idx]
        conf = pred[0][idx]*100
        col_color = color_dict.get(cls, "#000000")
        cols[i].markdown(f"<div class='prediction-box' style='background-color:{col_color}; color:white;'>{cls} : {conf:.2f}%</div>", unsafe_allow_html=True)
