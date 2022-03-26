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
import tensorflow as tf
import time

cap_check = cv2.VideoCapture(2) # Khai báo USB Camera Check Config
cap_check.set(3, 1920)
cap_check.set(4, 1080)

height, weight = 1080, 1920
start_row,start_col= 1085,446
end_row,end_col= 1203,590

def check_align(mean):
    arr = []
    image = cv2.resize(mean,(1920,1080))
    cropped=image[start_col:end_col,start_row:end_row]

    save_img = cropped
    # plt.imshow(cropped)
    # plt.show()
    cropped = cropped.astype('float')/255.0
    arr.append(cropped)


    arr = np.array(arr)
    saved_model = tf.keras.models.load_model('weights_jig_1.h5')
    ypred = saved_model.predict(arr)
    # print(ypred)
   
    if (ypred[0] > 0.5):
        ypred[0] = 1
    else:
        ypred[0] = 0
    # cv2.imshow("ing", cropped)
    # cv2.waitKey(0)
    return ypred[0], save_img
    
for i in range (10):
    ret, test_imgs = cap_check.read()
#################
index_false = 1051

##############
for i in range(50):
    ret, test_imgs = cap_check.read()
    start = time.time()
    result, img = check_align(test_imgs)
    # print("Predict: ", result)
    end = time.time()
    result = result.tolist()
#t_tag la bien thoi gian, tag_t la tag true
    t_tag = end - start
    tag_t = str(index_false)+ "_kq_"+str(result[0])+"_" +str(round(t_tag,3))
    cv2.imwrite("result/CNN/machine_result/"+tag_t+".jpg", img) # thay gia tri img thanh anh
    index_false += 1
    time.sleep(0.5)
    