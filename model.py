# Import generic utils
import os

# Import math libs
import numpy as np
import pandas as pd

# Import image processing libs
import cv2
from scipy.misc import imresize

# Import the keras layers
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16

"""
CONSTANTS
"""
PATH = '/home/karti/sdc-live-trainer/data'  # Data path

# Data augmentation constants
ANGLE_BIAS = 0.4  # Bias for choosing a good distribution of angles for training
TRANS_X_RANGE = 50  # Number of translation pixels up to in the X direction for augmented data (-RANGE/2, RANGE/2)
TRANS_Y_RANGE = 40  # Number of translation pixels up to in the Y direction for augmented data (-RANGE/2, RANGE/2)
ANGLE_PER_TRANS = .05  # Maximum angle change when translating in the X direction
OFF_CENTER_IMG_ANGLE = .15  # Angle change when using off center images
BRIGHTNESS_RANGE = .1  # The range of brightness changes
ANGLE_SMOOTH_DIV = 3.  # A divisor to smooth the angles in the training set to make the drive smoother

# Training constants
BATCH = 64  # Number of images per batch
TRAIN_BATCH_PER_EPOCH = 320  # Number of batches per epoch for training
TRAIN_VAL_CHECK = 1e-3  # The maximum increase in validation loss during re-training
RETRAIN_LR = 1e-5  # The learning rate used when re-training

# Image constants
IMG_ROWS = 140  # Number of rows in the image
IMG_COLS = 200  # Number of cols in the image
IMG_CH = 3  # Number of channels in the image


def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """

    # Remove the unwanted top scene and retain only the track
    roi = img[60:140, :, :]

    # Resize the image
    resize = imresize(roi, (IMG_ROWS, IMG_COLS))

    # Normalize the image to -1 to 1
    norm = ((resize / 255.) - 0.5 * 2.)
    return np.resize(norm, (1, IMG_ROWS, IMG_COLS, IMG_CH))


def img_change_brightness(img):
    # Convert the image to HSV
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute a random brightness value (only to decrease brightness) and apply it to the image
    brightness = BRIGHTNESS_RANGE + np.random.uniform() - 0.5
    temp[:, :, 2] = temp[:, :, 2] * brightness

    # Convert back to RGB and return
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)


def img_translate(img, x_translation):
    # Randomly compute a Y translation
    y_translation = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)

    # Form the translation matrix
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

    # Translate the image
    return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))


def data_augment(img_path, angle, threshold, augment):
    x_translation = 0.
    if augment:
        # Randomly form the X translation distance and compute the resulting steering angle change
        x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
        new_angle = angle + ((x_translation / TRANS_X_RANGE) * 2) * ANGLE_PER_TRANS

        # Check if the new angle does not meets the threshold requirements
        if (abs(new_angle) + ANGLE_BIAS) < threshold:
            return None, None
    else:
        new_angle = angle

    # Hurray, the newly generated angle matches the threshold
    img = cv2.imread(img_path)
    if augment:
        img = img_change_brightness(img)
        img = img_translate(img, x_translation)
    img = img_pre_process(img)

    # Finally, lets' decide if we want to flip the image or not
    if np.random.randint(2) == 0:
        return img, new_angle
    return np.fliplr(img), -new_angle


def val_data_generator(df):
    """
    Validation data generator
    :param df: Pandas data frame consisting of all the validation data
    :return: (x[BATCH, IMG_ROWS, IMG_COLS, NUM_CH], y)
    """
    # Preconditions
    assert len(df) == BATCH, 'The length of the validation set should be batch size'

    while 1:
        _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
        _y = np.zeros(BATCH, dtype=np.float)

        for idx in np.arange(BATCH):
            _x[idx] = img_pre_process(cv2.imread(os.path.join(PATH, df.center.iloc[idx].strip())))
            _y[idx] = df.steering.iloc[idx] / ANGLE_SMOOTH_DIV

        yield _x, _y


def train_data_generator(df, augment):
    """
    Training data generator
    :param df: Pandas data frame consisting of all the training data
    :param augment: Boolean to state if data augmentation should be used or not
    :return: (x[BATCH, IMG_ROWS, IMG_COLS, NUM_CH], y)
    """
    _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
    _y = np.zeros(BATCH, dtype=np.float)
    out_idx = 0
    while 1:
        # Get a random line and get the steering angle
        idx = np.random.randint(len(df))
        angle = df.steering.iloc[idx]

        """
        ANGLE SMOOTHING
        ---------------
        The human driven angle data is generally quite extreme. Empirically,
         softening the angles by dividing them by a constant and making them
         smoother allows the car to drive around the track much more smoothly.

        But note, this counter acts our idea that we want to bias the model
        towards bigger angles and not let it bias to 0. So we need to walk a
        fine line and this is a step to be taken near the end of the training
        session
        """
        angle /= ANGLE_SMOOTH_DIV

        # If augmentation is enabled, pick one of the images, left, right or center
        if augment:
            img_choice = np.random.randint(3)
        else:
            img_choice = 1

        if img_choice == 0:
            img_path = os.path.join(PATH, df.left.iloc[idx].strip())
            angle += OFF_CENTER_IMG_ANGLE
        elif img_choice == 1:
            img_path = os.path.join(PATH, df.center.iloc[idx].strip())
        else:
            img_path = os.path.join(PATH, df.right.iloc[idx].strip())
            angle -= OFF_CENTER_IMG_ANGLE

        """
        Randomly distort the (img, angle) to generate new data
        Here, we want to bias towards not selecting low angles, so we generate a random number
        and if that number were less than the absolute value of the newly coined angle + a known bias,
        only then do we accept the transformation.
        """
        threshold = np.random.uniform()
        img, angle = data_augment(img_path, angle, threshold, augment)

        # Check if we've got valid values
        if img is not None:
            _x[out_idx] = img
            _y[out_idx] = angle
            out_idx += 1

        # Check if we've enough values to yield
        if out_idx >= BATCH:
            yield _x, _y

            # Reset the values back
            _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
            _y = np.zeros(BATCH, dtype=np.float)
            out_idx = 0


def get_model():
    """
    Defines the model
    :return: Returns the model
    """
    """
    Check if a model already exists
    """
    if os.path.exists(os.path.join('.', 'model.json')):
        ch = input('A model already exists, do you want to reuse? (y/n): ')
        if ch == 'y' or ch == 'Y':
            with open(os.path.join('.', 'model.json'), 'r') as in_file:
                json_model = in_file.read()
                model = model_from_json(json_model)

            weights_file = os.path.join('.', 'model.h5')
            model.load_weights(weights_file)
            print('Model fetched from the disk')
            model.summary()
            return model

    """
    Get the pre-trained convolutional layers as a feature extractor
    from VGG16
    """
    input_layer = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CH))
    base_model = VGG16(input_tensor=input_layer,
                       weights='imagenet',
                       include_top=False)

    """
    Add spatial global average pooling
    """
    temp = base_model.output
    temp = GlobalAveragePooling2D()(temp)

    """
    Add fully connected layers
    """
    temp = Dense(1024, activation='elu')(temp)
    temp = Dropout(0.5)(temp)
    temp = Dense(512, activation='elu')(temp)
    temp = Dropout(0.5)(temp)
    temp = Dense(128, activation='elu')(temp)
    temp = Dropout(0.5)(temp)
    temp = Dense(32, activation='elu')(temp)
    temp = Dropout(0.5)(temp)
    predictions = Dense(1, init='zero')(temp)

    """
    Create the full model
    """
    model = Model(input=base_model.input, output=predictions)

    """
    Print the summary
    """
    model.summary()
    return model


def train_model(model, train_data, val_data, init=False):
    """
    Trains the given model
    :param model: A keras model
    :param train_data: Training data as a pandas data frame
    :param val_data: The validation data as a pandas data frame
    :param init: Are we training from scratch of re-training
    :return: The history of the model
    """

    # If we are training the FC layers from scratch
    if init:
        # Freeze the feature extractor (it's 19 layers)
        for layer in model.layers[:19]:
            layer.trainable = False
        for layer in model.layers[19:]:
            layer.trainable = True

        """
        Compile and train the model naturally for 1 epoch so that
        the fully connected layer can settle down
        Note:
        1. We train with the default learning rate, we want the data to overfit
        slightly. This is because this is a complex hypothesis. So asking the model
        to generalize directly is like asking a child to learn complex linear algebra
        on the first standard. We always ask children to memorize simple formulas and
        then slowly eventually, they learn to generalize.
        2. To support the above process, we also do not feed augmented data to the
        model in the first training run
        """
        model.compile(loss='mse', optimizer='adam')

        # Get an evaluation on the validation set
        print('Initial evaluation loss = {}'.format(
            model.evaluate_generator(val_data_generator(val_data), val_samples=BATCH)))

        # Try some predictions before we start..
        test_predictions(model, train_data)

        # Train over some epochs without data augmentation to train the fully connected layers
        model.fit_generator(
            train_data_generator(train_data, False),
            samples_per_epoch=TRAIN_BATCH_PER_EPOCH * BATCH,
            nb_epoch=1,
            validation_data=val_data_generator(val_data),
            nb_val_samples=BATCH,
            verbose=1)

    """
    Now that the fully connected layer is fully settled, let's get the full training
    started.
    Note:
    1. We allow more of the VGG16 convnet to be fine-tuned
    2. When we are retraining, we'll start directly from here
    3. we start with a smaller learning rate so as to not over-fit the data
    4. We enable data augmentation so that the data generalizes now
    """
    # Make the bottom two Conv Layers trainable in VGG16
    for layer in model.layers[:11]:
        layer.trainable = False
    for layer in model.layers[11:]:
        layer.trainable = True

    # Recompile the model with a finer learning rate
    model.compile(optimizer=Adam(lr=RETRAIN_LR), loss='mse')

    # Get an evaluation on the validation set
    val_loss = model.evaluate_generator(val_data_generator(val_data), val_samples=BATCH)
    print('Pre-trained evaluation loss = {}'.format(val_loss))

    # Try some predictions before we start..
    test_predictions(model, train_data)

    num_runs = 0
    while True:
        print('Run {}'.format(num_runs+1), end=': ')

        history = model.fit_generator(
            train_data_generator(train_data, True),
            samples_per_epoch=TRAIN_BATCH_PER_EPOCH * BATCH,
            nb_epoch=1,
            validation_data=val_data_generator(val_data),
            nb_val_samples=BATCH,
            verbose=1)
        num_runs += 1

        # Print out the test predictions
        test_predictions(model, train_data)

        # Save the model and the weights so far as checkpoints so we can manually terminate when things
        # go south...
        # Think that statement is very offensive to the south though, let's call it
        # when things go north :P
        save_model(model)

        # If the validation loss starts to increase, it's time for us to stop training...
        if history.history['val_loss'][-1] > (val_loss + TRAIN_VAL_CHECK):
            break

        # Store the validation loss for the next epoch
        val_loss = history.history['val_loss'][-1]


def test_predictions(model, df, num_tries=5):
    """
    Tries some random predictions
    :param model: The keras model
    :param df: The validation data as a pandas data frame
    :param num_tries: Number of images to try on
    :return: None
    """
    print('Predictions: ')
    for i in np.arange(num_tries):
        topset = df.loc[df.steering < (i * .4) - .6]
        subset = topset.loc[topset.steering >= (i * .4) - 1.]
        idx = int(len(subset)/2)
        img = img_pre_process(cv2.imread(os.path.join(PATH, subset.center.iloc[idx].strip())))
        img = np.resize(img, (1, IMG_ROWS, IMG_COLS, IMG_CH))
        org_angle = subset.steering.iloc[idx] / ANGLE_SMOOTH_DIV
        pred_angle = model.predict(img, batch_size=1)
        print(org_angle, pred_angle[0][0])


def save_model(model):
    """
    Saves the model and the weights to a json file
    :param model: The mode to be saved
    :return: None
    """
    json_string = model.to_json()
    with open('model.json', 'w') as outfile:
        outfile.write(json_string)
    model.save_weights('model.h5')
    print('Model saved')


if __name__ == '__main__':
    # Set the seed for predictability
    np.random.seed(200)

    # Load the data
    total_data = pd.read_csv(os.path.join(PATH, 'driving_log.csv'))

    # Shuffle and split the dataset
    validate, train = np.split(total_data.sample(frac=1), [BATCH])
    del total_data

    # Create a model
    steering_model = get_model()

    # Train the model
    train_model(steering_model, train, validate, False)

    # Save the model
    save_model(steering_model)

    exit(0)
