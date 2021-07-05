import cv2
#from  PIL import Image, ImageOps
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf
#from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
#from keras.models import load_model

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Digital Image Processing lab- Mid Term 2</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Operations on Image
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png"))
file2= st.file_uploader("Please upload second image", type=("jpg", "png"))
operation = st.selectbox("Operations: ",
                     ['Lgical XOR',"NOT"])


def import_and_predict(image,image2,operation):
  #img = image.load_img(image_data, target_size=(224, 224))
  #image = image.img_to_array(img)
  #img_reshap= np.expand_dims(image, axis=0)
  #img_reshap = preprocess_input(img_reshap)
  image=cv2.resize(image,(512,512))
  image2=cv2.resize(image2,(512,512))
  if operation=='Logical XOR':
    img = cv2.bitwise_xor(image,image2)
    st.image(img, use_column_width=True)
  else:
    img = cv2.bitwise_not(image)
    st.image(img, use_column_width=True)
    img2 = cv2.bitwise_not(image2)
    st.image(img2, use_column_width=True)
  
  return 0
if file is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)

if file2 is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image2 = cv2.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Perform Operation"):
  result=import_and_predict(image,image2,operation)
  
if st.button("About"):
  st.header("Bhavesh Kumawat")
  st.subheader("Student, Department of Computer Engineering,PIET")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Digital Image processing Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
