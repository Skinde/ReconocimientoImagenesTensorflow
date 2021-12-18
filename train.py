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

#Folder en donde las imagenes estan etiquetadas
folder = 'images2/'
#Para que todas las imagenes tengan el mismo tamano
IMG_WIDTH = 100
IMG_HEIGHT = 100

def create_dataset(folder, N):
  listNames3D = []
  listNamesRe = []
  x_y_train = []          # Lista de tuples que contiene la imagen y su clase
  x_train = []            # Lista de imagenes normalizadas y formato numpy
  y_train = []            # Lista de clases en formato numpy
  i = 0                   # Iterador de clases
  # A travez de todas las etiquetas
  for label in os.listdir(folder):
    # Sobre todos los archivos (aka las fotos)
    src = os.listdir(os.path.join(folder, label))
    srcA = os.listdir(os.path.join(folder,label,"3D"))  # imagenes 3D
    srcB = os.listdir(os.path.join(folder,label,"Real"))  # imagenes Reales
    A = len(srcA)
    B = len(srcB)
    if N == 0:
      imgReduce = [A, 0]
    elif N == 1:
      imgReduce = [0, B]
    else:
      p = (1 - N) / N
      ratio = B / A
      if p == ratio:
        imgReduce = [0,0]
      else:
        imgReduce = [A-B/p, 0] if p > ratio else [0, B-A*p]
    reducedA = A-round(imgReduce[0])
    reducedB = B-round(imgReduce[1])
    listNames3D = random.sample(srcA, reducedA)
    listNamesRe = random.sample(srcB, reducedB)
    print("{}: 3D: {} Reales: {} Ratio = {}%\n".format(label, reducedA, reducedB, 100*reducedA/(reducedA + reducedB)))

    for imgName in listNames3D:
      # Carga la imagen
      image_path=os.path.join(folder, label, src[0], imgName)
      image= cv2.imread(image_path)
      image_resize = cv2.resize(image, (IMG_HEIGHT,IMG_WIDTH))
      image_resize = np.array(image_resize).astype('uint8')
      # Normalizar
      image_resize = image_resize/255
      # Anade a la lista de tuples para posterior shuffle
      x_y_train.append((image_resize,[i]))
    for imgName in listNamesRe:
      # Carga la imagen
      image_path=os.path.join(folder, label, src[1], imgName)
      image= cv2.imread(image_path)
      image_resize = cv2.resize(image, (IMG_HEIGHT,IMG_WIDTH))
      image_resize = np.array(image_resize).astype('uint8')
      # Normalizar
      image_resize = image_resize/255
      # Anade a la lista de tuples para posterior shuffle
      x_y_train.append((image_resize,[i]))
    # Iterador de clases
    i += 1
  # Random shuffle
  random.shuffle(x_y_train)
  # Separar x_train y y_train
  for x,y in x_y_train:
    x_train.append(x)
    y_train.append(y)
  return np.array(x_train), np.array(y_train)

# Obtenemos los datos de la carpeta con las imagenes
simuImages = 0.75
x_train, y_train = create_dataset(folder, simuImages)
# Convertimos a un vector binario -> '2' = [0 0 1]
y_train_catego = to_categorical(y_train)

# Create model architecture
model2 = Sequential()
model2.add(Conv2D(32, (5,5), activation = 'relu', input_shape = (100,100,3))) 
model2.add(MaxPooling2D(pool_size = (2,2)))
model2.add(Conv2D(32, (5,5), activation = 'relu'))
model2.add(MaxPooling2D(pool_size = (2,2)))
model2.add(Flatten())
model2.add(Dense(100, activation = 'relu'))
model2.add(Dropout(0.5))
model2.add(Dense(50, activation = 'relu'))
model2.add(Dropout(0.5))
model2.add(Dense(25, activation = 'relu'))
model2.add(Dense(4, activation = 'softmax')) # Ultima capa que coincide con el numero de clases '4'

# Compile model
model2.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics  = ['accuracy'])

# Train the model
hist = model2.fit(x_train, y_train_catego,
                 batch_size = 256,
                 epochs= 50,
                 validation_split = 0.25)

print(hist)

# Se guarda el modelo para posterior verificacion
model2.save('trained{}'.format(round(100*simuImages)))