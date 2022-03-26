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


with open('clf_logistic.pkl', 'rb') as f:
    clf = pickle.load(f)

height, weight = 1080, 1920
start_row,start_col= 1085,446
end_row,end_col= 1203,590

def check_align(mean):

    image = cv2.resize(mean,(1920,1080))
    cropped=image[start_col:end_col,start_row:end_row]
    sample_img = cropped.reshape(-1,144*118)

    ypred = clf.predict(sample_img)
   

    # cv2.imshow("ing", cropped)
    # cv2.waitKey(0)
    return ypred[0], cropped

###############################       
index_false = 1
############################

for i in range(300):
    test_imgs = cv2.imread("test_data/image (" + str(i+1) + ").jpg")
    cv2.imwrite('checkjig.jpg', test_imgs)
    img_check = cv2.imread('checkjig.jpg', cv2.IMREAD_GRAYSCALE)
    start = time.time()
    result, img = check_align(img_check)
    # print("Predict: ", result)
    end = time.time()
    result = result.tolist()
    t_tag = end - start
    tag_t = str(index_false)+ "_kq_"+str(result)+"_" +str(round(t_tag,5))
    cv2.imwrite("Mai_check_data/LOGISTIC/"+tag_t+".jpg", img) # thay gia tri img thanh anh
    index_false += 1
    time.sleep(0.5)