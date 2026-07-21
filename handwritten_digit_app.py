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


st.markdown("""
<style>

.main{
    background-color:#F5F7FA;
}

h1{
    text-align:center;
    color:#1565C0;
}

.stButton>button{
    width:100%;
    height:45px;
    border-radius:10px;
    background-color:#1565C0;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#0D47A1;
}

div[data-testid="stMetric"]{
    border:2px solid #1565C0;
    border-radius:10px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

with st.expander("📖 Instructions"):
    st.write("""
    1. Draw a digit between **0 and 9**
    2. Draw it in the center
    3. Click **Predict**
    4. The model will identify the digit
    """)

    

canvas_result = st_canvas(
    fill_color = "#00000000",  # Canvas background color -> black
    stroke_width = 10,
    stroke_color ="#FFFFFF",  # Stroke color -> white
    background_color ="#FFFFFF",
    width = 280,
    height = 280,
    drawing_mode = "freedraw",
    key = "canvas",
)
if st.button("Predict"):
    st.write("Predicting...")

    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)

        #Convert image to greyscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize the image to 28x28 pixels
        grey_img = cv2.resize(grey_img, (28, 28))    

        # Normalize the pixel values to be between 0 and 1
        grey_img = grey_img / 255.0

        # Reshape the image to match the input shape of the model
        grey_img = grey_img.reshape(1,784)

        result = model.predict(grey_img)    # Predict the digit using the pre-trained model

        index = np.argmax(result)   # Get the index of the highest probability digit

        st.success(f"🎯 Predicted Digit : {index}")

