# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:02:43 2021
@author: Administrator
"""

import numpy as np
import pickle
import cv2
import os
import sys
import matplotlib.pyplot as plt
import time
import tensorflow as tf


saved_model = tf.keras.models.load_model('weights_CNN_new6.h5')

height, weight = 1080, 1920
start_row,start_col= 1085,446
end_row,end_col= 1203,590

CNN=[]
for i in range (2500):
    arr=[]
    path = "img/img (" + str(i+1) + ").jpg"
    img = cv2.imread(path)

    img = img.astype('float')/255.0
    arr.append(img)
    arr = np.array(arr)
    ypred = saved_model.predict(arr)
    if (ypred[0] > 0.5):
        ypred[0] = 1
    else:
        ypred[0] = 0
    CNN.append(str(ypred[0]))

with open('CNN6.txt', 'w') as f:
    for line in CNN:
        f.write(line)
        f.write('\n')

with open('CNN6.txt', 'w') as f:
    for line in CNN:
        f.write(line)
        f.write('\n')

        
        
