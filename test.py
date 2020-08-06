import numpy as np 
import pandas as pd 
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
st.set_option('deprecation.showfileUploaderEncoding', False)
import cv2
from PIL import Image, ImageOps
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.keras.models.load_model('my_model.hdf5')



st.write("""
         # Architectural heritage elements Prediction
         """
         )

st.write("This is a image classification web app to predict architectural heritage elements")



file = st.file_uploader("Please upload an image file", type=["jpg", "png"])



def import_and_predict(image_data, model):
    
        size = (128,128)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(128, 128),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a altar!")
    elif np.argmax(prediction) == 1:
        st.write("It is a apse!")
    elif np.argmax(prediction) == 2:
        st.write("It is a bell_tower!")
    elif np.argmax(prediction) == 3:
        st.write("It is a column!")
    elif np.argmax(prediction) == 4:
        st.write("It is a dome(inner)!")
    elif np.argmax(prediction) == 5:
        st.write("It is a dome(outer)!")
    elif np.argmax(prediction) == 6:
        st.write("It is a flying_buttress!")
    elif np.argmax(prediction) == 7:
        st.write("It is a gargoyle!")
    elif np.argmax(prediction) == 8:
        st.write("It is a stained_glass!")
    else:
        st.write("It is a vault!")
    
    st.text("Probability (0: altar, 1: apse, 2: bell_tower, 3: column , 4: dome(inner), 5: dome(outer) , 6: flying_buttress , 7: gargoyle , 8: stained_glass, 9: vault")
    st.write(prediction)




