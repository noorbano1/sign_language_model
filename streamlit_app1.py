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

# Load labels.pkl
with open(LABELS_PATH, "rb") as f:
    label_dict = pickle.load(f)

# Convert {label: index} → {index: label}
index_to_label = {v: k for k, v in label_dict.items()}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Sign Language Detection", page_icon="🤟", layout="centered")
st.markdown("""
<style>
body {
    background-color: #FFF8DC;
}
h1 {
    color: #FF6347;
}
h2 {
    color: #4682B4;
}
h3 {
    color: #32CD32;
}
</style>
""", unsafe_allow_html=True)

st.title("🤟 Sign Language Detection (A–Z, 0–9)")
st.write("Upload a hand gesture image, and the model will predict the sign with confidence.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess: Resize → Normalize → Expand dims
    img_array = np.array(img)
    img_array = cv2.resize(img_array, (64, 64))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred))
    predicted_class = index_to_label[class_index]

    # Color dictionary for labels (you can expand/change colors)
    color_dict = {
        'A':'#FF5733','B':'#33FF57','C':'#3357FF','D':'#FF33A8','E':'#FF8F33',
        'F':'#33FFF0','G':'#DA33FF','H':'#FFC733','I':'#33FF99','J':'#9933FF',
        'K':'#FF3333','L':'#33CCFF','M':'#FF33F6','N':'#33FFBD','O':'#FF8333',
        'P':'#33FF66','Q':'#FF33B5','R':'#FFAA33','S':'#33D1FF','T':'#FF3355',
        'U':'#33FFCC','V':'#FF33CC','W':'#33FF33','X':'#FF9933','Y':'#33B5FF',
        '0':'#FF6666','1':'#66FF33','2':'#6633FF','3':'#FF33AA','4':'#33FF88',
        '5':'#FF3380','6':'#33FF44','7':'#FF9933','8':'#33CC99','9':'#FF6633'
    }

    # Get color for predicted class
    label_color = color_dict.get(predicted_class, "#000000")

    # Display prediction nicely
    st.markdown(f"<h2 style='color:{label_color};'>Predicted Sign: {predicted_class}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:#FF4500;'>Confidence: {round(confidence*100,2)} %</h3>", unsafe_allow_html=True)

    # Optional: Show top 3 predictions
    top_indices = pred[0].argsort()[-3:][::-1]
    st.write("### Top 3 Predictions:")
    for i in top_indices:
        cls = index_to_label[i]
        conf = pred[0][i] * 100
        st.markdown(f"<p style='color:{color_dict.get(cls,'#000')}; font-size:18px;'>{cls} : {conf:.2f}%</p>", unsafe_allow_html=True)
