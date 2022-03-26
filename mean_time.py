from types import new_class
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

directory = 'Mai_check_data'
classes = ['CNN', 'HISTOGRAM', 'LOGISTIC', 'SVM']
# classes = ['CNN']

time = []
for i in classes:
    path = os.path.join(directory, i) # path = F:\python\CODE_thu\directory\i
    sum = 0
    for img in os.listdir(path): #liet ke anh trong duong dan path
        string = os.path.join(path,img)
        # print(string)
        num = len(string)
        index = 0
        for char in range(num-1, -1, -1):
            if string[char] == '_':
                index = index+1
                if index == 1:
                    name_time = float(string[int(char+1):(len(string)-4)])
                    sum = sum+name_time
    print("===========")
    print(classes[classes.index(i)])
    print(sum/1400)
    time.append(sum/1400)