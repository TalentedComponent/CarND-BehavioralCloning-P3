# Import math libs
import numpy as np
import json
import pickle
import os
import csv
import matplotlib.pyplot as plt

import cv2
from skimage.exposure import equalize_adapthist

from sklearn.utils import shuffle

# Import the keras layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Convolution2D, Activation
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils import np_utils
from keras import backend as K


def preprocess_img(img):
    """
    Pre-processes the image and returns it
    :param img: The image to be pre-processed
    :return: Returns the pre-processed image
    """
    #fig = plt.figure()
    img = cv2.resize(img, None, fx=0.1, fy=0.1)
    #fig.add_subplot(1, 5, 1),plt.imshow(img)

    #blur = cv2.medianBlur(img, 1)
    #fig.add_subplot(1, 5, 3), plt.imshow(blur, cmap='gray')

    equ = equalize_adapthist(img)
    #fig.add_subplot(1, 5, 5), plt.imshow(equ, cmap='gray')

    #plt.show()
    return equ


def data_generator(path='/home/karti/data/data/'):
    x = []
    y = []
    while 1:
        f = open(os.path.join(path,'driving_log.csv'), 'r')
        rows = f.readlines()
        rows = shuffle(rows)
        for row in rows:
            row = row.split(', ')
            img_path = os.path.join(path,row[0])
            if os.path.exists(img_path):
                img = cv2.imread(os.path.join(path,row[0]))
                x.append(img)
                y.append(float(row[3]))
                x.append(np.fliplr(img))
                y.append(-float(row[3]))

            if len(y) >= 400:
                x = np.resize(x, (400, 160, 320, 3))
                y = np.array(y)
                yield (x, y)
                x = []
                y = []
        f.close()

ch, row, col = 3, 160, 320

# for x,y in data_generator():
#     fig = plt.figure()
#     fig.add_subplot(1, 2, 1), plt.imshow(x[0])
#     fig.add_subplot(1, 2, 2), plt.imshow(x[1])
#     plt.show()

# Define the model
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Activation('relu'))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
#model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(512))
#model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(128))
#model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(64))
#model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(32))
#model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

# Compile and run the model
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(data_generator(), samples_per_epoch=16000, nb_epoch=100, verbose=1)

# Save the model and weights
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)
model.save_weights('model.h5')
print ('Model saved')