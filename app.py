# GET 324 - Mini-Project (Group EE8)
# Healthy Tomato vs Tomato Yellow Leaf Curl Virus (YLCV) Classifier
# Structure follows Laboratory Exercise 10 of the GET 324 Practical Guide.

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# STEP 1: Configure the page
# ------------------------------------------------------------
st.set_page_config(page_title="Tomato Leaf Classifier", page_icon="🍅", layout="centered")

# Class order MUST match class_names printed at the end of train_model.py
# (Keras infers classes alphabetically from folder names: Healthy, then
# Yellow_Leaf_Curl_Virus)
CLASS_NAMES = ["Healthy", "Yellow_Leaf_Curl_Virus"]
DISPLAY_NAMES = {
    "Healthy": "Healthy",
    "Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus (YLCV)",
}
IMAGE_SIZE = (224, 224)

# ------------------------------------------------------------
# STEP 2: Load the trained model (cached so it only loads once)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/tomato_ylcv_model.keras")

# ------------------------------------------------------------
# STEP 3: Prediction function
# ------------------------------------------------------------
def predict(model, pil_image):
    """Make a prediction and return the label plus per-class probabilities."""
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    probs = model.predict(arr, verbose=0)[0]      # softmax output, shape (2,)
    pred_idx = int(np.argmax(probs))
    label = DISPLAY_NAMES[CLASS_NAMES[pred_idx]]

    pct = {DISPLAY_NAMES[c]: float(p) * 100 for c, p in zip(CLASS_NAMES, probs)}
    return label, pct

# ------------------------------------------------------------
# STEP 4: Build the user interface
# ------------------------------------------------------------
st.title("🍅 Tomato Leaf Health Classifier")
st.write(
    "Upload a photo of a tomato leaf to check whether it is **healthy** or "
    "shows signs of **Tomato Yellow Leaf Curl Virus (YLCV)**."
)

model = load_model()
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

# ------------------------------------------------------------
# STEP 5: Run prediction and display results
# ------------------------------------------------------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300, caption="Uploaded leaf image")

    label, pct = predict(model, img)
    st.write(f"### Prediction: **{label}**")

    for class_label, value in pct.items():
        st.progress(int(value), text=f"{class_label}: {value:.1f}%")

    if label == "Healthy":
        st.success("This leaf appears healthy.")
    else:
        st.warning("This leaf shows signs of Yellow Leaf Curl Virus. Consider consulting an agricultural extension officer.")
else:
    st.info("Please upload an image to get a prediction.")
