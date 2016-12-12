# Import generic utilities
import csv
import os
import numpy as np
import cv2
import pickle
from progress.bar import Bar

# Image utilities
from matplotlib import pyplot as plt

# Data manipulation
from sklearn.model_selection import train_test_split
from skimage.exposure import equalize_adapthist

class DrivingData:
    """
    Class with the responsibility of augmenting all the data collected so far,
    preprocess the data and make it ready for input
    """
    def __init__(self):
        """
        Initializes the class
        """
        self.n_train = 0
        self.n_val = 0
        self.n_test = 0
        self.img_sz = (160, 320)
        self.num_ch = 3

        self.X = {}
        self.y = {}

    def load_data(self, _path, _train):
        """
        loads a training / testing data file (csv) along with it's corresponding data collection
        :param _path: Path to the csv file
        :param _train: Boolean to say if this is training data or test data
        :return: None
        """
        # Preconditions
        assert os.path.exists(_path) == True, "File does not exist"

        # Initialize
        features = []
        outputs = []

        print('##### Started loading {} #####'.format(_path))

        # Read the csv file
        with open(_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            print ('Processing: ',end='')
            for row in csv_reader:

                """
                Only take in data if the driving speed is beyond a threshold,
                this is to ensure that we don't record the samples when the car
                was stationary (when starting to record, etc...
                """
                if float(row[6]) > 5:
                    # Read the center image
                    #img = plt.imread(row[0])
                    #img = self.preprocess_img(img)

                    # Add to the corresponding collection
                    #features.append(img)
                    outputs.append(float(row[3]))

                #if len(outputs) % 200:
                    #print ('#',end='')

                #if len(outputs) > 10000:
                    #print('')
                   # break

        fig = plt.figure()
        plt.bar(np.arange(len(outputs)), outputs)
        plt.show()

        # If the variable does not exist (i.e.: first time we are loading the data)
        if not hasattr(self, 'X_train'):
            self.X_train = np.array(features)
            self.y_train = np.array(outputs)
        else:
            self.y_train = np.append(self.y_train, np.array(outputs))
            self.X_train = np.append(self.X_train, np.array(features))

        self.X_train.resize(self.y_train.shape[0], self.img_sz[0], self.img_sz[1], self.num_ch)

        # Update the variables
        self.n_train = self.y_train.shape[0]

        # Print out success
        print('##### Loaded {} #####'.format(_path))
        print('Total training samples = {}'.format(self.n_train))
        input('Press any key to continue: ')

    def split_data(self, split=0.0):
        """
        Splits the data into training, cross validation and testing data
        :return: None
        """
        # Split the data
        self.X['train'], self.X['val'], self.y['train'], self.y['val'] = \
            train_test_split(self.X_train, self.y_train, test_size=split, random_state=42)

        # Update the variables
        self.n_train = self.X['train'].shape[0]
        self.n_val = self.X['val'].shape[0]

        # Print confirmation
        print('##### Training, validation data generated #####')
        print('Total training samples = {}'.format(self.n_train))
        print('Total validation samples = {}'.format(self.n_val))
        input('Press any key to continue: ')

    def pickle_the_data(self):
        """
        Pickles the data for future use
        :return: None
        """
        try:
            with open('data.pickle', 'wb') as pfile:
                pickle.dump(
                    {
                        'X_train': self.X['train'],
                        'y_train': self.y['train'],
                        'X_test': self.X['val'],
                        'y_test': self.y['val']
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', 'data.pickle', ':', e)
            raise
        input('Press any key to continue: ')

    def load_from_pickle(self):
        pickle_file = 'data.pickle'
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
            self.X['train'] = pickle_data['X_train']
            self.y['train'] = pickle_data['y_train']
            self.X['val'] = pickle_data['X_test']
            self.y['val'] = pickle_data['y_test']
            del pickle_data  # Free up memory

        print('Training and test data loaded')
        input('Press any key to continue: ')


    @staticmethod
    def preprocess_img(img):
        """
        Pre-processes the image and returns it
        :param img: The image to be pre-processed
        :return: Returns the pre-processed image
        """
        #fig = plt.figure()

        img = cv2.resize(img, None, fx=0.2,fy=0.2)
        #fig.add_subplot(1, 5, 1),plt.imshow(img)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #fig.add_subplot(1, 5, 2), plt.imshow(gray, cmap='gray')

        blur = cv2.medianBlur(gray, 3)
        #fig.add_subplot(1, 5, 3), plt.imshow(blur, cmap='gray')

        equ = equalize_adapthist(blur)
        #fig.add_subplot(1, 5, 5), plt.imshow(equ, cmap='gray')

        #plt.show()
        return equ