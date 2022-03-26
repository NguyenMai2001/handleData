from connectPLC import PLC
from matplotlib import pyplot as plt

import cv2
import numpy as np
import sys
import time
import os
import math

from official import check_align


#Khai báo cam detect
cap_detect = cv2.VideoCapture(2)
cap_detect.set(3, 1920)
cap_detect.set(4, 1080)

#Khai báo cam Check
cap_check = cv2.VideoCapture(1)
cap_check.set(3, 1920)
cap_check.set(4, 1080)

############
Controller = PLC()
Controller.testConnection()
pre_command =''
index_true = 14
index_false = 66
while True: 
    command = Controller.queryCommand()
    print(Controller.queryCommand())

    if command == 'Done_detect':
        pre_command = 'Done_detect'
        # print("pre_command = " , pre_command)
    # status_cam_checked =Controller.status_cam_checked()
    # status_cam_inJig = Controller.status_cam_in_jig()
    jig_signal = Controller.jig_Signal()

    if command == 'Detect':
        print("Done_detect")
    elif command == 'Check' and pre_command == 'Done_detect':

        for i in range (10):
            ret, test_imgs = cap_check.read()

        ret, test_imgs = cap_check.read()
        cap_done = True

        start = time.time()
        check, img_save = check_align(test_imgs)
        end = time.time()
        print("Processing time = ",  end - start )

        if check :
            #Check khong lech
            tag_t = 'K_lech_'+str(index_true)
            index_true+=1
            Controller.send_status_cam_check('Ok_for_jig')
            cv2.imwrite("dtan_1/"+tag_t+".jpg", img_save) # thay gia tri img thanh anh
            
            # Controller.send_status_cam_inJig('Ok')

            # jig_signal = Controller.jig_Signal()
            # print("Jig_signal", jig_signal)
            # if jig_signal:
                
            #     Controller.send_status_cam_inJig('Ok')
        else:
            tag_t = 'lech_'+str(index_false)
            index_false+=1
            cv2.imwrite("dtan_0/"+tag_t+".jpg", img_save) # thay gia tri img thanh anh
            Controller.send_status_cam_check('Skeff')
            # print(Controllecr.status_cam_in_jig())

        pre_command = 'check'
   
    
       



