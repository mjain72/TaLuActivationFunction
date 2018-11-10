#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 12:35:58 2018

@author: mohit
"""
"""
Benchmarking Talu with Relu, Elu and Leaku_Relu
"""
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tqdm import tqdm #to show progress
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
import keras.activations

#customized activation



def talu(x): 
    cond = tf.less_equal(x, x*0.0)
    t = tf.tanh(x)
    tanH = tf.tanh(-0.5)
    cond1 = tf.less_equal(x, -0.5*(1 - x*0.0))
    y = tf.where(cond1, tanH*(1 - x*0.0), t)
    return tf.where(cond, y, x)

keras.activations.custom_activation = talu

#seed 
np.random.seed(24)

image_size = 32
n_channel = 3


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train_train, X_train_dev, y_train_train, y_train_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=24)

def cnn_model():
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), input_shape=(image_size, image_size, n_channel), activation=talu))
    classifier.add(Conv2D(64, kernel_size=(3,3),  strides=(1,1), activation=talu))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv2D(128, kernel_size=(3,3), strides=(1,1),  activation=talu))  
    classifier.add(MaxPool2D(pool_size=(2,2)))
  
    
    classifier.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), activation=talu))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation=talu))
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(units=10, activation='softmax'))
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()    
    return classifier

def cnn_model_relu():
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), input_shape=(image_size, image_size, n_channel), activation='relu'))
    classifier.add(Conv2D(64, kernel_size=(3,3),  strides=(1,1), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv2D(128, kernel_size=(3,3), strides=(1,1),  activation='relu'))  
    classifier.add(MaxPool2D(pool_size=(2,2)))
  
    
    classifier.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation='relu'))
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(units=10, activation='softmax'))
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()    
    return classifier

def leaky_relu(z):
    return tf.maximum(0.01*z, z)

def cnn_model_leaky_relu():
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), input_shape=(image_size, image_size, n_channel), activation=leaky_relu))
    classifier.add(Conv2D(64, kernel_size=(3,3),  strides=(1,1), activation=leaky_relu))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv2D(128, kernel_size=(3,3), strides=(1,1),  activation=leaky_relu))  
    classifier.add(MaxPool2D(pool_size=(2,2)))
  
    
    classifier.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), activation=leaky_relu))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation=leaky_relu))
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(units=10, activation='softmax'))
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()    
    return classifier

def cnn_model_elu():
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), input_shape=(image_size, image_size, n_channel), activation='elu'))
    classifier.add(Conv2D(64, kernel_size=(3,3),  strides=(1,1), activation='elu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Conv2D(128, kernel_size=(3,3), strides=(1,1),  activation='elu'))  
    classifier.add(MaxPool2D(pool_size=(2,2)))
  
    
    classifier.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), activation='elu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation='elu'))
    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(units=10, activation='softmax'))
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()    
    return classifier


callbacksBoard = TensorBoard(log_dir='my_log_dir')
checkpoint = ModelCheckpoint('modelTalu_cifar_10_talu50.h5', verbose=1, save_best_only=True)
earlStopping = EarlyStopping(min_delta=0.001, patience=3)

#Train model

model = cnn_model()
model.fit(X_train_train/255.0, to_categorical(y_train_train), batch_size=128, shuffle=True, epochs=25, validation_data=(X_train_dev/255.0, to_categorical(y_train_dev)), callbacks=[checkpoint, callbacksBoard])

#evaluate the model performance

performance = model.evaluate(X_test/255.0, to_categorical(y_test))
print('Loss: %0.3f' % performance[0])
print('Accuracy: %0.3f' % performance[1])

    
    
    