import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("cnn_mnist.h5")

st.title("🖼️ CNN Digit Classifier")

uploaded = st.file_uploader("Upload an image of a digit", type=["jpg", "jpeg", "png"])

if uploaded:
    # Convert to grayscale & resize
    image = Image.open(uploaded).convert("L").resize((28, 28))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
    pred = np.argmax(model.predict(img_array, verbose=0), axis=1)[0]

    st.success(f"✅ Predicted Digit: {pred}")

