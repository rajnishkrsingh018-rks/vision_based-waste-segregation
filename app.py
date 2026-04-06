import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageFile
import tensorflow as tf
import tempfile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    model_lvl1 = tf.keras.models.load_model("model_level1_fixed.h5", compile=False)
    model_bio = tf.keras.models.load_model("model_bio_fixed.h5", compile=False)
    model_nonbio = tf.keras.models.load_model("model_nonbio_fixed.h5", compile=False)
    return model_lvl1, model_bio, model_nonbio

model_lvl1, model_bio, model_nonbio = load_models()

bio_classes = ["food_waste","leaf_waste","paper_waste","wood_waste"]
nonbio_classes = ["ewaste","metal_cans","plastic_bags","plastic_bottles"]

# ---------------- PREDICTION ----------------
def predict(frame):
    img = cv2.resize(frame, (224,224)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred1 = model_lvl1.predict(img)

    if np.argmax(pred1) == 0:
        pred2 = model_bio.predict(img)
        label = bio_classes[np.argmax(pred2)]
        conf = np.max(pred2)
        return f"Biodegradable → {label} ({conf*100:.1f}%)"
    else:
        pred2 = model_nonbio.predict(img)
        label = nonbio_classes[np.argmax(pred2)]
        conf = np.max(pred2)
        return f"Non-Biodegradable → {label} ({conf*100:.1f}%)"

# ---------------- UI ----------------
st.set_page_config(page_title="Waste Segregation", layout="centered")

st.title("♻️ Vision-Based Waste Segregation System")

option = st.radio("Choose Input Type", ["Image", "Video", "Webcam"])

# ---------------- IMAGE ----------------
if option == "Image":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        frame = np.array(img)

        st.image(img, width=300)

        result = predict(frame)
        st.success(result)

# ---------------- VIDEO ----------------
elif option == "Video":
    file = st.file_uploader("Upload Video", type=["mp4","avi"])

    if file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        cap = cv2.VideoCapture(path)
        frame_window = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            label = predict(frame)

            cv2.putText(frame, label, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            frame_window.image(frame, channels="BGR")

        cap.release()

# ---------------- WEBCAM ----------------
elif option == "Webcam":
    st.write("Live Detection")

    run = st.checkbox("Start Webcam")
    frame_window = st.image([])

    # 🔥 Change index if needed (0 / 1 / 2)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("❌ Camera not detected. Try changing index (0/1/2)")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                break

            label = predict(frame)

            cv2.putText(frame, label, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            frame_window.image(frame, channels="BGR")

        cap.release()