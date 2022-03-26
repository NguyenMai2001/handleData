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

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

#1
def find_location_crop(event, x, y, flags, param):
    f = open(resource_path('data/config/location_crop.txt'), 'a')
    if event == cv2.EVENT_LBUTTONDOWN:
        f.write(str(x) + "\n")
        f.write(str(y) + "\n")
    f.close()

#2
def crop_image(image):
    a = []
    f = open(resource_path('data/config/location_crop.txt'), 'r+')
    x1 = int(f.readline())
    y1 = int(f.readline())
    x2 = int(f.readline())
    y2 = int(f.readline())
    crop = image[y1:y2, x1:x2, :]
    # cv2.imwrite(resource_path('data/img_check_crop/{}.jpg'.format(i)), crop)
    a.append(crop)
    f.close()
    return a

#3
# def calc_mean(image):
#     return np.mean(image, axis=(0, 1))

# def calc_mean_all(image_list):
#     a = []
#     for i in range(4):
#         # img = cv2.imread(resource_path('data/img_check_crop/{}.jpg'.format(i)))
#         img = image_list[i]
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         a.append(calc_mean(img))
#     return a
with open('clf_log.pkl', 'rb') as f:
    clf = pickle.load(f)

height, weight = 1080, 1920
start_row,start_col= 1085,446
end_row,end_col= 1203,590

def check(mean):

    image = cv2.resize(mean,(1920,1080))
    cropped=image[start_col:end_col,start_row:end_row]
    sample_img = cropped.reshape(-1,144*118)

    ypred = clf.predict(sample_img)
   

    # cv2.imshow("ing", cropped)
    # cv2.waitKey(0)
    return ypred[0]

# demo chek lech
img=cv2.imread("check_lech/lech/image (100).jpg", cv2.IMREAD_GRAYSCALE)

print(type(img))
dem = check(img)
print(dem)
if dem:
    print("vi tri dung")
else:
    print("lech")

# plt.imshow(img_cr, cmap='gray')
# plt.show()


#gray_tray_1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)