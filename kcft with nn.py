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

class_names = ['walking', 'block', 'idle'] #can add character names here
c = len(class_names) #used for output of the neural network
#video = ""  #name of the video to perform detection on
prefix = "./training/"
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
        if len(files)<100:
            index = random.sample(files, len(files))
        else:
            index = random.sample(files, 100)
        for img in index:

            image = cv2.imread(path+"/" + img, 0)
            image = resizeImage(image)
            train_all.append(np.reshape(image, (image.shape[0]*image.shape[1], 1)))
            train_all_labels.append(idx)
        idx+=1

    Y_train = keras.utils.to_categorical(train_all_labels)
    return np.array(train_all)[:,:,0], Y_train


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

    model.fit(train_all, train_all_labels,epochs=200, batch_size = 128)
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







train_all, train_all_labels = dataset_proc()

model = model_dev(train_all, train_all_labels)

def tracker(vid_name):
    kcft = cv2.TrackerKCF_create()
    curve=[]
    vid = cv2.VideoCapture(vid_name)
    init_kcft, frame = vid.read()
    roi = cv2.selectROI(frame, False)
    init_kcft = kcft.init(frame, roi)
    height, width, layers = frame.shape
    size = (250,200)
    out = cv2.VideoWriter('project_mod.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

    while True:
            # next frame
            init_kcft, frame = vid.read()
            if not init_kcft:
                break
            # Update
            init_kcft, roi = kcft.update(frame)
            if init_kcft:
                #  success
                corner_1 = (int(roi[0]), int(roi[1]))
                corner_3 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
                cv2.rectangle(frame, corner_1, corner_3, (0,0,0), 3, 2)
                gr_frame = cv2.cvtColor(frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])], cv2.COLOR_BGR2GRAY)
                put = "ROI & Action:" + str(pred_loop(gr_frame, model))
            else :
                # In case of failure
                cv2.putText(frame, "Target Undetectable", (50,100),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255),1)
                put = "ROI & Action: Undefined"
            cv2.putText(frame, "KCF Tracker", (50,50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255,255,255),1);


            cv2.putText(frame, put, corner_1, cv2.FONT_HERSHEY_PLAIN, 2,
                        (0,0,255),1);

            # Display result
            cv2.imshow("Tracking", frame)
            dim = (250,200)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            out.write((frame))
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 :
                out.release()
                break
    cv2.destroyAllWindows()
    out.release()
tracker("./assets/game1.webm")
