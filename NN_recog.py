# -*- coding: utf-8 -*-
"""
Created on Tue Nov  10 4:07:56 2021

@author: Sandesh Jain
"""


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

# All global variables

class_names = ['s', 'b'] #can add character names here
c = len(class_names) #used for output of the neural network
video = ""  #name of the video to perform detection on

# feed the nn with 100x100 images, hence resize function:
def resizeImage(image):
    dim = (100,100)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


#currently takes 2 paths for +ve and -ve data folders, can place input arg to 
# receive custom paths, additionally for loop can be added to reduce boilerplate code
def dataset_proc():
    
    # handle any size input data and build the respective lists of training set
    path = glob.glob("./Train_dataset/positive/*.jpg")
    train_s = []
    train_s_labels = []
    train_b = []
    train_b_labels = []
    
    for img in path:
        n = cv2.imread(img)
        image = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        image = resizeImage(image)
        train_s.append(image)
        train_s_labels.append(0) 
    
    
    path = glob.glob("./Train_dataset/negative/*.jpg")
    
    for img in path:
        n = cv2.imread(img)
        image = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        image = resizeImage(image)
        train_b.append(image)
        train_b_labels.append(1) 
    
    # Experiment block to see in the data set is built properly
    plt.figure()
    plt.imshow(train_b[1])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    
    train_all = train_s
    train_all.extend(train_b)
    train_all_labels = train_s_labels
    train_all_labels.extend(train_b_labels)
    train_all = np.array(train_all)
    train_all_labels = np.array(train_all_labels)
    
    
    # divide by 255 so that the values are between 0 and 1 for ease neural network processing
    train_all = train_all / 255.0    
    plt.figure(figsize=(25,25))
    for i in range(32):
        plt.subplot(20,20,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_all[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_all_labels[i]])
    plt.show()
    
    # ready to use dataset for training
    return train_all, train_all_labels


# Takes in the training set and returns the fit model
def model_dev(train_all, train_all_labels):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(100, 100)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(c, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_all, train_all_labels,epochs=50, batch_size = 50)
    return model
    
    
# supposed to be used with the kcft input and displays the result
# currently handles binary input but can be scaled easily

def pred_loop(frame):    

    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img_cut = resizeImage(gray_img)
    img_cut = img_cut / 255.0
    train_all, train_all_labels = dataset_proc()
    model = model_dev(train_all, train_all_labels)
    oncam_pred = model.predict(img_cut)
    if oncam_pred[0, 0] > 0.7:
        gray_img = cv2.putText(gray_img, "Some Character", (1,1), cv2.FONT_ITALIC, 0.75, (0,0,255), 3)
        gray_img = cv2.putText(gray_img, "Prediction Confidence = "+str(oncam_pred[0, 0]),
                          (1,1), cv2.FONT_ITALIC, 0.75, (0,0,255), 3)
    else:
        gray_img = cv2.putText(gray_img, "Else", (1,1), cv2.FONT_ITALIC, 1.5, (0,0,255), 3)            
    cv2.imshow('gray_img',gray_img)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed ==27:
        cv2.destroyAllWindows()


