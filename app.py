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
   <center><p style="font-size:30px;color:white;margin-top:10px;">Digital Image Processing lab- EndTerm</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Operations on Image
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png" , "tif"))
kind = st.selectbox("Operations: ",
                     ["Reflect-X", "Reflect-Y", "Translation", "Rotation", "Cropping"])
an = st.text_input("Enter Angle of rotation", "Type Here ...")


def import_and_predict(image,operation,an):
  #img = image.load_img(image_data, target_size=(224, 224))
  #image = image.img_to_array(img)
  #img_reshap= np.expand_dims(image, axis=0)
  #img_reshap = preprocess_input(img_reshap)
  img2=cv2.resize(image,(512,512))
  an= int(an)
  if kind == "Reflect-X":
   reflected = cv.flip(img2, 0)
   st.image(reflected, use_column_width=True)

  elif kind == "Reflect-Y":
   reflected = cv.warpPerspective(img2, ymat , (int(cols),int(rows)))
   st.image(reflected, use_column_width=True)

  elif kind== "Translation":
   M1 = np.float32([[1, 0, 20], 
                [0, 1, 100], 
                [0, 0, 1]])
   img3 = cv.warpPerspective(img2, M1, (img2.shape[1], img2.shape[0]))
   M = np.float32([[1, 0, 100], [0, 1, 20], [0, 0, 1]])
   img3 = cv.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]))
   st.image(img3, use_column_width=True)
  elif kind=="Cropping":
   cropped_img = img2[25:100, 50:200]
   st.image(cropped_img, use_column_width=True)
  elif kind=="Rotation":
   angle = np.radians(an)
   #transformation matrix for Rotation
   M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	[np.sin(angle), np.cos(angle), 0],
            	[0, 0, 1]])
   # apply a perspective transformation to the image
   rotated_img = cv.warpPerspective(img2, M, ((cols),(rows)))
   st.image(rotated_img, use_column_width=True)
  
  return 0
if file is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)


    
if st.button("Perform Operation"):
  result=import_and_predict(image,operation,an)
  
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
