# Import math libs
import numpy as np
import json
import pickle
import os
import csv

import cv2
from skimage.exposure import equalize_adapthist

from sklearn.utils import shuffle

# Import the keras layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
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
    # fig = plt.figure()

    img = cv2.resize(img, None, fx=0.1, fy=0.1)
    # fig.add_subplot(1, 5, 1),plt.imshow(img)

    #blur = cv2.medianBlur(img, 1)
    # fig.add_subplot(1, 5, 3), plt.imshow(blur, cmap='gray')

    equ = equalize_adapthist(img)
    # fig.add_subplot(1, 5, 5), plt.imshow(equ, cmap='gray')

    # plt.show()
    return equ


def data_generator():
    x = []
    y = []
    while 1:
        f = open('/home/karti/simulator-linux-50/track1/driving_log.csv', 'r')
        rows = f.readlines()
        rows = shuffle(rows)
        for row in rows:
            row = row.split(', ')
            if float(row[6]) > 5:
                img = cv2.imread(row[0], 0)
                x.append(preprocess_img(img))
                y.append(float(row[3]))

            if len(y) >= 256:
                x = np.resize(x, (256, 16, 32, 1))
                y = np.array(y)
                yield (x, y)
                x = []
                y = []
        f.close()


# Define the model
model = Sequential()
model.add(Conv2D(24, 3, 3, input_shape=(16, 32, 1)))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Conv2D(48, 3, 3))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

# Compile and run the model
model.compile(loss='mse',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])
try:
    history = model.fit_generator(data_generator(), samples_per_epoch=19200, nb_epoch=3, verbose=1)
except:
    print ('')

# Save the model and weights
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)
model.save_weights('model.h5')