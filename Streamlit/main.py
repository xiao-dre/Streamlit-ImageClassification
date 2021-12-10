from PIL.Image import NONE
from numpy.core.fromnumeric import resize
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

st.title('Corona Disease Detector')
st.write('This is a prototype of the Corona Disease Detector.')
st.write('Please upload an x-ray to diagnose.')
st.write('The classification result will be displayed below.')

model = load_model('../Model/myModel.hdf5')

#load file
uploaded_file = st.file_uploader('Choose an image file', type=['jpeg', 'jpg', 'png'])

map_dict = {
    0: 'NEGATIVE',
    1: 'POSITIVE',
}

# print(cv2.__version__)

if(uploaded_file is not None):
    # convert file into openCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (224, 224))

    #do something to the image, lets say we display it
    st.image(opencv_image, channels='RGBA')
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    Generate_pred = st.button('Generate Prediction')
    if(Generate_pred):
        prediction = model.predict(img_reshape).argmax()
        st.info('Predicted Covid Status is {}'.format(map_dict[prediction]))
