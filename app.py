import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load trained model
model = load_model("plant_disease_model.h5")

# Class names (dataset classes)
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

st.title("🌿 Plant Disease Detection AI")
st.write("Upload a leaf image and the AI will predict the disease.")

file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

if file is not None:

    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    disease_name = class_names[class_index]

    st.success(f"Predicted Disease: {disease_name}")
    st.info(f"Confidence: {confidence:.2f}%")
