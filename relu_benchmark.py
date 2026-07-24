# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:14:57 2022

@author: ar2806
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split


# load train and test dataset
def load_dataset1():
    (X_train, y_train), (testX, testY) = cifar10.load_data()
    trainX, valX, trainY, valY = train_test_split(X_train, y_train, test_size=5000, random_state=1234)
    trainY = to_categorical(trainY)
    valY = to_categorical(valY)
    testY = to_categorical(testY)
    return trainX, trainY, valX, valY, testX, testY


# scale pixels
def prep_pixels1(train, val, test):
    train_norm = train.astype('float32')
    val_norm = val.astype('float32')
    test_norm = test.astype('float32')

    train_norm = train_norm / 255.0
    val_norm = val_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, val_norm, test_norm


# define cnn model
def define_model1():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, valX, valY, testX, testY = load_dataset1()
    # prepare pixel data
    trainX, valX, testX = prep_pixels1(trainX, valX, testX)
    # define model
    model = define_model1()
    # load model weights for training further
    # model.load_weights('GA_weights.h5')
    # fit model
    model.fit(trainX, trainY, epochs=15, batch_size=64, validation_data=(valX, valY), verbose=1)
    # save model weights
    # model.save_weights('GA_weights.h5')
    # evaluate model
    loss, acc = model.evaluate(testX, testY, verbose=0)
    return loss, acc


loss, acc = run_test_harness()
print("Test loss:", loss)
print("Test accuracy:", acc)
