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
arr=[]
for i in range (2500):
    path = "img/img (" + str(i+1) + ").jpg"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    sample_img = img.reshape(-1,144*118)

    ypred = clf.predict(sample_img)
    arr.append(str(ypred[0]))

with open('Tan.txt', 'w') as f:
    for line in arr:
        f.write(line)
        f.write('\n')

        
        
