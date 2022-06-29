import tensorflow.compat.v1 as tf
from tensorflow import keras
import keras
from keras.models import Sequential
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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Model

import random
import numpy as np
import matplotlib.pyplot as plt

import cv2

import h5py
import os
import argparse
import shutil
import random

from generate_data import generate_img

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-p", "--prefix", default="test")
parser.add_argument("-lr", "--lr", type=float, default=0.00001)


args = parser.parse_args()
prefix = args.prefix
lr = args.lr

print(prefix, lr)

try:
    os.mkdir("models/" + prefix)
except:
    pass

print(tf.__version__)


shutil.copy("train.py", "models/" + prefix + "/train.py")
shutil.copy("generate_data.py", "models/" + prefix + "/generate_data.py")


# train
train_graph = tf.Graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
train_sess = tf.Session(graph=train_graph, config=config)

keras.backend.set_session(train_sess)


def build_model():

    # 2 Convo Layer
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(
        Conv2D(
            32, (5, 5), activation="relu", padding="same", input_shape=(None, None, 6)
        )
    )
    model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))

    #     model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    #     model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))

    #     model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    #     model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))

    #     model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    #     model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(Conv2D(64, (5, 5), activation="sigmoid", padding="same"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(2, (5, 5), padding="same"))

    return model


def build_model_ae():
    padding = 0

    padding = "same"
    ksize = (5, 5)

    #     input_img = Input(shape=(WIDTH, HEIGHT, 6))
    input_img = Input(shape=(None, None, 6))

    conv1 = Conv2D(
        32, ksize, activation="relu", padding=padding, input_shape=(None, None, 6)
    )(input_img)
    conv1 = Conv2D(32, ksize, activation="relu", padding=padding)(conv1)

    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, ksize, activation="relu", padding=padding)(pool1)
    conv2 = Conv2D(64, ksize, activation="relu", padding=padding)(conv2)

    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, ksize, activation="relu", padding=padding)(pool2)
    conv3 = Conv2D(128, ksize, activation="relu", padding=padding)(conv3)
    #     conv3 = Conv2D(128, ksize, activation='relu', padding = padding)(conv3)
    #     conv3 = Conv2D(128, ksize, activation='relu', padding = padding)(conv3)

    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, ksize, activation="relu", padding=padding)(pool3)
    conv4 = Conv2D(256, ksize, activation="sigmoid", padding=padding)(conv4)

    output = Conv2D(2, (5, 5), padding=padding)(conv4)

    model = Model(input_img, output)

    return model


def build_model_small():
    padding = 0

    padding = "same"
    ksize = (5, 5)

    #     input_img = Input(shape=(WIDTH, HEIGHT, 6))
    input_img = Input(shape=(None, None, 6))

    conv1 = Conv2D(
        16, ksize, activation="relu", padding=padding, input_shape=(None, None, 6)
    )(input_img)
    conv1 = Conv2D(16, ksize, activation="relu", padding=padding)(conv1)

    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, ksize, activation="relu", padding=padding)(pool1)
    conv2 = Conv2D(32, ksize, activation="relu", padding=padding)(conv2)

    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, ksize, activation="relu", padding=padding)(pool2)
    conv3 = Conv2D(128, ksize, activation="relu", padding=padding)(conv3)
    conv3 = Conv2D(128, ksize, activation="relu", padding=padding)(conv3)
    conv3 = Conv2D(128, ksize, activation="relu", padding=padding)(conv3)

    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, ksize, activation="relu", padding=padding)(pool3)
    conv4 = Conv2D(256, ksize, activation="sigmoid", padding=padding)(conv4)

    output = Conv2D(2, (5, 5), padding=padding)(conv4)

    model = Model(input_img, output)

    return model


with train_graph.as_default():
    #     model = build_model()
    #     model = build_model_ae()
    model = build_model_small()

    #     tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=200)
    train_sess.run(tf.global_variables_initializer())

    #     model = load_model('models/random_ae/tracking_029_0.631.h5')
    #     model = load_model('models/random_ae_color/tracking_000_0.514.h5')
    #     model = load_model('models/random_ae_grid_abs_k5_2/tracking_012_1.790.h5')
    #     model = load_model('models/random_ae_random_multi_scratch/tracking_047_2.604.h5')
    #     model = load_model('models/random_ae_var_grid/tracking_025_0.627.h5')
    #     model = load_model('models/random_ae_fix_2/tracking_010_0.297.h5')
    #     model = load_model('models/random_ae_fix_block/tracking_022_0.274.h5')
#     model = load_model("models/random_ae_fix_subpixel_color/tracking_099_0.055.h5")
    #     model = load_model('models/random_ae_fix_small/tracking_099_0.329.h5')
    #     model = load_model('models/random_ae_fix_subpixel_bound/tracking_002_0.343.h5')

    optimizer = keras.optimizers.Adam(
        lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    )
    model.compile(optimizer=optimizer, loss="mean_squared_error")


def preprocessing(img):
    global WIDTH, HEIGHT
    #     Brightness
    ret = img.copy()

    blur = cv2.GaussianBlur(img[3:], (31, 31), 0)

    sz = int(3 + random.random() * 15)
    x = int(random.random() * (WIDTH - sz))
    y = int(random.random() * (HEIGHT - sz))
    ret[x : x + sz, y : y + sz, 3:] = blur[x : x + sz, y : y + sz, :]

    ret = ret * (0.9 + random.random() * 0.2)

    return ret


with train_graph.as_default():

    datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        #         rotation_range=5,
        #     width_shift_range=0.05,
        #     height_shift_range=0.05,
        #         zoom_range=0.05,
        #         preprocessing_function=preprocessing
    )

    min_loss = 100
    #     datagen_flow = datagen.flow(X_train, Y_train, batch_size=32)

    #     X_test, Y_test = next(generate_img(10000, setting=(48, 48, 6, 6)))
    X_test, Y_test = next(generate_img(100, setting=(80, 112, 10, 14)))
    #     X_test, Y_test = next(generate_img(100, setting=(80, 112, 10, 14)))

    for i in range(100):
        print("epoch", i)

        #         model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=2)
        model.fit_generator(
            generate_img(32),
            validation_data=None,
            steps_per_epoch=20,
            epochs=1,
            workers=16,
            use_multiprocessing=True,
        )
        #         model.fit_generator(generate_img(), validation_data = None, steps_per_epoch=2, epochs = 1)

        pred = model.predict(X_test)
        loss = ((pred - Y_test) ** 2).mean()
        #         loss = ((pred[0] - Y_test[0])**2).mean()
        print(loss)

        # #         # save graph and checkpoints
        # #         saver = tf.train.Saver()
        # #         saver.save(train_sess, "models/{}/checkpoints_{:03d}_{:.3f}".format(prefix, i, loss))
        if loss < min_loss:
            min_loss = loss
            model.save("models/{}/tracking_{:03d}_{:.3f}.h5".format(prefix, i, loss))
