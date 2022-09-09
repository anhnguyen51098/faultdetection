import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import faultdetection_m2.extract_object_module as ext

st.title("""FAULT DETECTION - MODULE 2: OBJECT EXTRACTION""")
st.subheader("Extracting object in the order")

input_image = st.file_uploader(label='Insert image')

if input_image is not None:
    img = Image.open(input_image)
    temp_dir = "img_temp.png"
    img = img.save(temp_dir)
    st.image(input_image, caption="Sample image")

    list_image = ext.extract_image(temp_dir)

    st.subheader("Step 1: Rotation")
    rotated_img_dir = 'rotated_obj.jpg'
    rotated_img = Image.open(rotated_img_dir)
    st.image(rotated_img, caption='Image after rotation')

    st.subheader("Step 2: Extraction")
    st.image(list_image, use_column_width='always', caption=list_image)
