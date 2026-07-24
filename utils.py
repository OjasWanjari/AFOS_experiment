# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:23:13 2022

@author: ar2806
"""
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# load train and test dataset
def load_dataset():
    (X_train, y_train), (testX, testY) = cifar10.load_data()
    trainX, valX, trainY, valY = train_test_split(X_train, y_train, test_size=5000, random_state=1234)
    trainY = to_categorical(trainY)
    valY = to_categorical(valY)
    testY = to_categorical(testY)
    return trainX, trainY, valX, valY, testX, testY
 
# scale pixels
def prep_pixels(train, val, test):
    train_norm = train.astype('float32')
    val_norm = val.astype('float32')
    test_norm = test.astype('float32')
    
    train_norm = train_norm / 255.0
    val_norm = val_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, val_norm, test_norm