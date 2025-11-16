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
    page_title="🎨 Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS for colors & gradients
# ---------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #FFDEE9, #B5FFFC);
}
.stButton>button {
    background-color: #FF5733;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
}
.prediction-box {
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 15px;
    text-align: center;
    font-weight: bold;
    font-size: 24px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar instructions
# ---------------------------
st.sidebar.header("📖 Instructions")
st.sidebar.markdown("""
- Upload a hand gesture image (A–Z or 0–9).  
- The model predicts the sign with confidence.  
- Prediction boxes are colorful and easy to read.  
- See top 3 predictions for comparison.  
- Enjoy the rainbow of colors! 🌈
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
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(img)
    img_array = cv2.resize(img_array, (64,64))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred))
    predicted_class = index_to_label[class_index]

    # Bright rainbow colors
    color_dict = {
        'A':'#FF0000','B':'#FF7F00','C':'#FFFF00','D':'#7FFF00','E':'#00FF00',
        'F':'#00FF7F','G':'#00FFFF','H':'#007FFF','I':'#0000FF','J':'#7F00FF',
        'K':'#FF00FF','L':'#FF007F','M':'#FF1493','N':'#FF69B4','O':'#DB7093',
        'P':'#8A2BE2','Q':'#4B0082','R':'#9400D3','S':'#FF4500','T':'#FFA500',
        'U':'#FFD700','V':'#ADFF2F','W':'#32CD32','X':'#00FA9A','Y':'#1E90FF',
        '0':'#FF1493','1':'#FF69B4','2':'#BA55D3','3':'#7B68EE','4':'#00CED1',
        '5':'#00FFFF','6':'#7FFFD4','7':'#32CD32','8':'#FFFF00','9':'#FF4500'
    }

    label_color = color_dict.get(predicted_class, "#000000")

    # ---------------------------
    # Show Main Prediction
    # ---------------------------
    st.markdown(f"""
    <div class='prediction-box' style='background: linear-gradient(90deg, #ff6a00, #ee0979);'>
        🎯 Predicted Sign: {predicted_class}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='prediction-box' style='background: linear-gradient(90deg, #00c6ff, #0072ff);'>
        💯 Confidence: {round(confidence*100,2)} %
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------
    # Top 3 Predictions in rainbow cards
    # ---------------------------
    top_indices = pred[0].argsort()[-3:][::-1]
    st.subheader("🏆 Top 3 Predictions")
    cols = st.columns(3)
    gradient_colors = ["#FF5733","#33FF57","#3357FF","#FF33A8","#FF8F33","#33FFF0","#DA33FF"]
    for i, idx in enumerate(top_indices):
        cls = index_to_label[idx]
        conf = pred[0][idx]*100
        col_color = gradient_colors[i % len(gradient_colors)]
        cols[i].markdown(f"""
        <div class='prediction-box' style='background: linear-gradient(135deg, {col_color}, #ffffff); color:black;'>
            {cls} : {conf:.2f}%
        </div>
        """, unsafe_allow_html=True)
