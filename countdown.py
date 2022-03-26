import time 
import cv2

def make_1080(cap):
    cap.set(3,1920)
    cap.set(4,1080)
######################################
cam_port = 2
count_down_time = 1 #seconds
number_of_image = 100
label = 'img_'
index = 1101 # tag for index of image

#####################################3
tmp = 0 # index of countdown
tmp1 = 0 # count image taken

cam = cv2.VideoCapture(cam_port)
make_1080(cam)
for i in range (10):
    ret, test_imgs = cam.read()
while True :
    if tmp1 > number_of_image - 1:
        # cam.release()
        break
    else:
        print("Remain %0.2f till next shot" %(count_down_time-tmp))
        time.sleep(1)
        tmp+= 1
        if tmp > count_down_time -1 :
            result, image = cam.read()
            print("This is %dth image taken !!!"%index)
            if result:
                pkg = label+str(index)
                # cv2.imshow("dtan", image)
                cv2.imwrite("samples/false/"+pkg+".jpg", image)
                # cv2.waitKey(1000)
                # cv2.destroyWindow("dtan")
                index += 1
            else:
                print("No image detected. Please! try again")
            tmp1 += 1
            tmp = 0
            