# -*- coding: utf-8 -*-
"""
Created on Tue Nov  10 4:07:56 2021

@author: Sandesh Jain
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import glob
import os
import random
# from tensorflow.keras.utils.np_utils import to_categorical
# All global variables

class_names = ['block', 'crouch', 'idle', 'walking', 'jump', 'kick', 'punch'] #can add character names here
c = len(class_names) #used for output of the neural network
#video = ""  #name of the video to perform detection on
prefix = "assets/training/"
all_paths  = [prefix + x for x in class_names]
# feed the nn with 100x100 images, hence resize function:
def resizeImage(image):
    dim = (70,70)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


#currently takes 2 paths for +ve and -ve data folders, can place input arg to
# receive custom paths, additionally for loop can be added to reduce boilerplate code
def dataset_proc():

    train_all = []
    train_all_labels = []
    idx = 0

    for path in all_paths:
        files = os.listdir(path)
        if len(files)<1000:
            index = random.sample(files, len(files))
        else:
            index = random.sample(files, 1000)
        for img in index:

            image = cv2.imread(path+"/" + img, 0)
            image = resizeImage(image)
            train_all.append(np.reshape(image, (image.shape[0]*image.shape[1], 1)))
            train_all_labels.append(idx)
        idx+=1

    Y_train = keras.utils.to_categorical(train_all_labels)
    return np.array(train_all)[:,:,0], Y_train

# import os
# import random

# path ='C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM2/CV/Project/training/training/backroll'
# files = os.listdir(path)
# index = random.sample(files, 10)



dataset_proc()





#     # handle any size input data and build the respective lists of training set
#     path = glob.glob("./Train_dataset/positive/*.jpg")
#     train_s = []
#     train_s_labels = []
#     train_b = []
#     train_b_labels = []

#     for img in path:
#         n = cv2.imread(img)
#         image = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
#         image = resizeImage(image)
#         train_s.append(image)
#         train_s_labels.append(0)


#     path = glob.glob("./Train_dataset/negative/*.jpg")

#     for img in path:
#         n = cv2.imread(img)
#         image = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
#         image = resizeImage(image)
#         train_b.append(image)
#         train_b_labels.append(1)

#     # Experiment block to see in the data set is built properly
#     # plt.figure()
#     # plt.imshow(train_b[1])
#     # plt.colorbar()
#     # plt.grid(False)
#     # plt.show()

#     train_all = train_s
#     train_all.extend(train_b)
#     train_all_labels = train_s_labels
#     train_all_labels.extend(train_b_labels)
#     train_all = np.array(train_all)
#     train_all_labels = np.array(train_all_labels)


#     # divide by 255 so that the values are between 0 and 1 for ease neural network processing
#     train_all = train_all / 255.0
#     # plt.figure(figsize=(25,25))
#     # for i in range(32):
#     #     plt.subplot(20,20,i+1)
#     #     plt.xticks([])
#     #     plt.yticks([])
#     #     plt.grid(False)
#     #     plt.imshow(train_all[i], cmap=plt.cm.binary)
#     #     plt.xlabel(class_names[train_all_labels[i]])
#     # plt.show()

#     # ready to use dataset for training
#     return train_all, train_all_labels


# Takes in the training set and returns the fit model
def model_dev(train_all, train_all_labels):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape = (4900,)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(c, activation=tf.nn.softmax) #s, c, f =(0.7,0.1,0.2)
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_all, train_all_labels,epochs=100, batch_size = 128)
    return model


# train_all, train_all_labels = dataset_proc()

# model = model_dev(train_all, train_all_labels)

# supposed to be used with the kcft input and displays the result
# currently handles binary input but can be scaled easily

def pred_loop(frame, model):

    gray_img = frame
    img_cut = resizeImage(gray_img)
    img_cut = img_cut / 255.0
    # train_all, train_all_labels = dataset_proc()
    # model = model_dev(train_all, train_all_labels)
    img_cut = np.reshape(img_cut, (1,4900))
    oncam_pred = model.predict(img_cut)[0]
    pred_class = np.argmax(oncam_pred)
    # gray_img = cv2.putText(gray_img, class_names[pred_class], (1,1), cv2.FONT_ITALIC, 0.75, (0,0,255), 3)
    # gray_img = cv2.putText(gray_img, "Prediction Confidence = "+str(oncam_pred[pred_class]),
    #                   (1,1), cv2.FONT_ITALIC, 0.75, (0,0,255), 3)

    # cv2.imshow('gray_img',gray_img)
    # key_pressed = cv2.waitKey(1) & 0xFF
    # if key_pressed ==27:
    #     cv2.destroyAllWindows()

    return class_names[pred_class]
#print(pred_loop(cv2.imread("C:/Users/Sandesh Jain/OneDrive/Documents/Acads_VT/SEM2/CV/Project/training/training/backroll/backroll84340.png", 0)))


train_all, train_all_labels = dataset_proc()

model = model_dev(train_all, train_all_labels)
train_all = []
train_all_labels = []
idx = 0

for path in all_paths:
    files = os.listdir(path)
    if len(files)<10:
        index = random.sample(files, len(files))
    else:
        index = random.sample(files, 10)
    for img in index:

        image = cv2.imread(path+"/" + img, 0)
        image = resizeImage(image)
        train_all.append(image)
        train_all_labels.append(class_names[idx])
    idx+=1

test_labels=[]
for test in train_all:
    test_labels.append(pred_loop(test, model))
