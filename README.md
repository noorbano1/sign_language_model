# Sign Language Detection Using CNN

![Sign Language Detection](placeholder_for_screenshot.png)

This project is a **Convolutional Neural Network (CNN)** based Sign Language Detection system trained on a Kaggle dataset containing **37 classes (A–Z and 0–9)**, each with **1200 images**.

The project includes:

* A trained deep learning model (`model.h5`)
* Label file (`labels.pkl`)
* A Streamlit interface (`app.py`) for real-time prediction
* Dataset splitting, preprocessing, training, and evaluation

---

## 📌 Features

* Automatic train/validation split
* CNN architecture with dropout (prevents overfitting)
* Real-time image upload for prediction
* Supports 37 labels: A–Z and 0–9
* Easy deployment on **Streamlit Cloud**

---

## 📂 Folder Structure

```
your_project/
│
├── model.h5             # Trained CNN model
├── labels.pkl           # Class labels mapping
├── app.py               # Streamlit app
├── requirements.txt     # Required Python packages
└── README.md            # Project documentation
```

---

## 📥 Installation & Usage

### 1️⃣ Install Required Libraries

```
pip install tensorflow streamlit opencv-python-headless numpy pillow
```

### 2️⃣ Run Streamlit App Locally

```
streamlit run app.py
```

### 3️⃣ Upload Your Own Image

* Supported formats: `jpg`, `jpeg`, `png`
* Model outputs predicted class and confidence score

---

## 🧠 Model Training Overview

* Image size: **224x224** pixels
* Normalization: `/255`
* CNN architecture includes:

  * Conv2D + MaxPooling layers
  * BatchNormalization
  * Dropout (to prevent overfitting)
* Callbacks used:

  * EarlyStopping
  * ReduceLROnPlateau
* Model saved as `model.h5`

---

## 🚀 Deployment Guide (Streamlit Cloud)

1. Upload project folder to GitHub
2. Go to **[Streamlit Cloud](https://share.streamlit.io)**
3. Connect your GitHub repo
4. Select `app.py`
5. Add `requirements.txt`
6. Click **Deploy**

Your app will be live in seconds.

---

## 🏷️ GitHub Badges (Optional)

```
![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Yes-green)
```

---

## 💡 Author

Sign Language Detection Project
Developed for educational & real-world applications.

---

*Note: Replace `placeholder_for_screenshot.png` with an actual screenshot of your app.*
