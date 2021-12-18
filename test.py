import numpy as np
import pandas as pd
import os
import tensorflow as tf
import cv2
import random
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Input
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from skimage.transform import resize

# Se carga el modelo
model2 = tf.keras.models.load_model('trained75')

# Definimos los nombres de las clases
folder = 'images2/'
classification = []
for label in os.listdir(folder):
    classification.append(label)
for image in os.listdir('testImages/'):
  # Read image
  new_image = plt.imread('testImages/'+image)
  # Resize image
  resize_image = resize(new_image, (100,100,3))
  # Get model prediction
  predictions = model2.predict(np.array([resize_image]))
  # Print
  print('\n', image)
  for i in range(4):
    print(classification[i], ':', round(predictions[0][i]*100,2), '%')

  