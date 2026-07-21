import streamlit as st
import numpy as np
import pandas as pd
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)

model = load_model("digit_recognition_model.keras")

st.title("Handwritten Digit Recognition")



with st.expander("📖 Instructions"):
    st.write("""
    1. Draw a digit between **0 and 9**
    2. Draw it in the center
    3. Click **Predict**
    4. The model will identify the digit
    """)

    
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=10,
    stroke_color="#FFFFFF",      # White pen
    background_color="000000",  # Black canvas
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    st.write("Predicting...")

    
    img = canvas_result.image_data.astype(np.uint8)

    #Convert image to greyscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        # Resize the image to 28x28 pixels
    grey_img = cv2.resize(grey_img, (28, 28))    

        # Normalize the pixel values to be between 0 and 1
    grey_img = grey_img / 255.0

        # Reshape the image to match the input shape of the model
    grey_img = grey_img.reshape(1,784)

    result = model.predict(grey_img)    # Predict the digit using the pre-trained model

    index = np.argmax(result)   # Get the index of the highest probability digit

    st.success(f"🎯 Predicted Digit : {index}")

