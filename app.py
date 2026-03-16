import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Get class names from dataset folders
class_names = sorted(os.listdir("raw/color"))

st.title("🌿 Plant Disease Detection AI")
st.write("Upload a leaf image and the AI will predict the disease.")

file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if file is not None:

    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)*100

    disease_name = class_names[class_index]

    st.success(f"Predicted Disease: {disease_name}")
    st.info(f"Confidence: {confidence:.2f}%")