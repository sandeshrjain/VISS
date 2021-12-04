# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:22:45 2021

@author: Sandesh Jain
"""

import cv2
import numpy as np
import glob
import pandas as pd

#Names of the characters fought in each episode
characters = ['blanka']
#characters = ['blanka','chunli','ehonda','ryu','zangief']
#Data we're extracting
states = ['x_position', 'y_position', 'status', 'health', 'round_timer']

#Set up dataframe to store information in
state_df = pd.DataFrame(columns=states)
episodes = 1
kcft = cv2.TrackerKCF_create()

for e in range(0,episodes):
    for c in characters:
        #Load images from numpy matrix
        title = './data/' + 'episode' + str(e) + '_' + c + '_images.npy'
        images = np.load(title)
height, width, layers = images[0].shape
size = (width, height)

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 40, size)
 
for i in range(len(images)):
    out.write(np.uint8(images[i]))
out.release()

