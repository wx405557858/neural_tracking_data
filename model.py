from tensorflow import keras
from keras.models import load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input, Concatenate
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    Convolution2D,
    UpSampling2D,
    Add,
    AveragePooling2D,
)
from keras.models import Model


def build_model_small():
    padding = "same"
    ksize = (5, 5)

    #     input_img = Input(shape=(WIDTH, HEIGHT, 6))
    input_img = Input(shape=(None, None, 6))

    conv1 = Conv2D(
        16, ksize, activation="relu", padding=padding, input_shape=(None, None, 6)
    )(input_img)
    conv1 = Conv2D(16, ksize, activation="relu", padding=padding)(conv1)
    conv1 = Conv2D(16, ksize, activation="relu", padding=padding)(conv1)

    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, ksize, activation="relu", padding=padding)(pool1)
    conv2 = Conv2D(32, ksize, activation="relu", padding=padding)(conv2)
    conv2 = Conv2D(32, ksize, activation="relu", padding=padding)(conv2)

    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, ksize, activation="relu", padding=padding)(pool2)
    conv3 = Conv2D(64, ksize, activation="relu", padding=padding)(conv3)
    conv3 = Conv2D(64, ksize, activation="relu", padding=padding)(conv3)

    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, ksize, activation="relu", padding=padding)(pool3)
    conv4 = Conv2D(128, ksize, activation="relu", padding=padding)(conv4)
    conv4 = Conv2D(128, ksize, activation="sigmoid", padding=padding)(conv4)

    output = Conv2D(2, (5, 5), padding=padding)(conv4)

    model = Model(input_img, output)

    return model
