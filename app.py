import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import torch
from ultralytics import YOLO

class_names = {
    0: 'BED-DOUBLE',
    1: 'BED-SINGLE',
    2: 'DISHWASHER',
    3: 'DOOR-DOUBLE',
    4: 'DOOR-SINGLE',
    5: 'DOOR-WINDOWED',
    6: 'REFRIGERATOR',
    7: 'SHOWER',
    8: 'SINK',
    9: 'SOFA-CORNER',
    10: 'SOFA-ONE',
    11: 'SOFA-THREE',
    12: 'SOFA-TWO',
    13: 'STOVE-OVEN',
    14: 'TABLE-DINNER',
    15: 'TABLE-STUDY',
    16: 'TELEVISION',
    17: 'TOILET',
    18: 'WARDROBE',
    19: 'WASHBASIN',
    20: 'WASHBASIN-CABINET',
    21: 'WASHINGMACHINE',
    22: 'WINDOW'
}

def identify(img, model):
    img = Image.open(img).convert('L')
    img=img.resize((112,112))
    img=np.array(img) /255.0
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    predicted_class_index = np.argmax(res)
    predicted_class_name = class_names.get(predicted_class_index, 'Unknown')
    return predicted_class_name

def identify_yolov8(img, model):
    img = Image.open(img).convert('L')
    result = model(img)
    names_dict = result[0].names
    prob = result[0].probs.numpy().top1
    prediction = (names_dict[prob])
    return prediction


# Streamlit App
st.title("CAD Drawing Prediction")

uploaded_file = st.file_uploader("Upload your drawing", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.subheader("Choose Model for Sentiment Classification")
# Choose between tasks
    model_option = st.selectbox('Select Model', ['Choose one','CNN', 'YOLO'])
    if model_option == "CNN":
        model = load_model('model.h5')
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=False, width=200)
        st.write("")

        if st.button("Predict"):
            result = identify(uploaded_file, model)
            st.subheader("Image Result") 
            st.write(f"**{result}**")

    if model_option == "YOLO":
        # Load the model
        model = YOLO(r"C:\Users\HP\Desktop\CAD classification\runs\classify\train6\weights\best.pt")  # Load the model configuration
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=False, width=200)
        st.write("")

        if st.button("Predict"):
            results = identify_yolov8(uploaded_file, model)
            st.subheader("Image Result")
            st.write(results)