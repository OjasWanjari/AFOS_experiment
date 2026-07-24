# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 08:26:08 2022

@author: ar2806
"""
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
import math

from utils import load_dataset, prep_pixels
from activation import get_activation


def _last_history_value(history, key):
    values = history.history.get(key, [])
    if not values:
        return math.inf
    value = values[-1]
    if math.isnan(value):
        return math.inf
    return value


def fitness(population, on_evaluation=None):
    scores =[]
    losses = []
    train_losses = []
    validation_losses = []
    test_accuracies = []
    train_accuracies = []
    validation_accuracies = []
    trainX, trainY, valX, valY, testX, testY = load_dataset()
    #repare pixel data
    trainX, valX, testX = prep_pixels(trainX, valX, testX)
    
    for serial, i in enumerate(population):
    #custom activation function
        
        def custom_activation(x):
            value = get_activation(i,x)
            return value
        model = Sequential()
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Dense(32, activation=custom_activation))
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(32, activation=custom_activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(64, activation=custom_activation))
        model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(64, activation=custom_activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(128, activation=custom_activation))
        model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(Dense(128, activation=custom_activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, kernel_initializer='he_uniform'))
        model.add(Dense(128, activation=custom_activation))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))

        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        


        print("Choromosome Number: ",serial, i)
        history = model.fit(trainX, trainY, epochs=1, batch_size=64, validation_data=(valX, valY), verbose=1)
        
        loss_check, acc_check = model.evaluate(testX, testY, verbose=0)
        if math.isnan(loss_check):
            scores.append(0.0)
            losses.append(math.inf)
            train_losses.append(_last_history_value(history, "loss"))
            validation_losses.append(_last_history_value(history, "val_loss"))
            test_accuracies.append(acc_check)
            train_accuracies.append(_last_history_value(history, "accuracy"))
            validation_accuracies.append(_last_history_value(history, "val_accuracy"))
            if on_evaluation is not None:
                on_evaluation(
                    serial=serial,
                    chromosome=i,
                    score=scores[-1],
                    test_loss=losses[-1],
                    train_loss=train_losses[-1],
                    validation_loss=validation_losses[-1],
                    test_accuracy=test_accuracies[-1],
                    train_accuracy=train_accuracies[-1],
                    validation_accuracy=validation_accuracies[-1],
                    model=model,
                )
            continue
        elif acc_check<0.25:
            scores.append(acc_check/loss_check)
            losses.append(loss_check)
            train_losses.append(_last_history_value(history, "loss"))
            validation_losses.append(_last_history_value(history, "val_loss"))
            test_accuracies.append(acc_check)
            train_accuracies.append(_last_history_value(history, "accuracy"))
            validation_accuracies.append(_last_history_value(history, "val_accuracy"))
            if on_evaluation is not None:
                on_evaluation(
                    serial=serial,
                    chromosome=i,
                    score=scores[-1],
                    test_loss=losses[-1],
                    train_loss=train_losses[-1],
                    validation_loss=validation_losses[-1],
                    test_accuracy=test_accuracies[-1],
                    train_accuracy=train_accuracies[-1],
                    validation_accuracy=validation_accuracies[-1],
                    model=model,
                )
            continue

        history = model.fit(trainX, trainY, epochs=14, batch_size=64, validation_data=(valX, valY), verbose=1)


        loss, acc = model.evaluate(testX, testY, verbose=0)
        
        scores.append(acc/loss)
        losses.append(loss)
        train_losses.append(_last_history_value(history, "loss"))
        validation_losses.append(_last_history_value(history, "val_loss"))
        test_accuracies.append(acc)
        train_accuracies.append(_last_history_value(history, "accuracy"))
        validation_accuracies.append(_last_history_value(history, "val_accuracy"))
        print("Test Score", acc/loss)

        if on_evaluation is not None:
            on_evaluation(
                serial=serial,
                chromosome=i,
                score=scores[-1],
                test_loss=losses[-1],
                train_loss=train_losses[-1],
                validation_loss=validation_losses[-1],
                test_accuracy=test_accuracies[-1],
                train_accuracy=train_accuracies[-1],
                validation_accuracy=validation_accuracies[-1],
                model=model,
            )
    return (
        population,
        scores,
        losses,
        train_losses,
        validation_losses,
        test_accuracies,
        train_accuracies,
        validation_accuracies,
    )
