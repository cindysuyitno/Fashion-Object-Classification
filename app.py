#^ to write/rewrite app.py everytime this cell runs

#importing libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

model = tf.keras.models.load_model('model.h5')

st.write("""#Fashion Picture Prediction""")
st.write('This is a image classification web app to predict fashion image')

file = st.file_uploader('Please upload an image file', type = ['jpg','png'])

classify = st.button('Classify Picture')
if classify:
  if file is None:
    st.text('Please upload an image file')
  else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    prediction = model.predict(np.array(cv2.resize(image, dsize = (150,150)))))
    x_label = ['T-shirt', 'Bag', 'Dress', 'Shoes', 'Trousers']
    y_label = prediction[0]

    fgr = plt.figure(figsize=(10, 6))
    ax = fgr.add_subplot()
    plt.bar(x_label, height = y_label)
    plt.xticks(rotation=90)
    st.pyplot(fig) 