import cv2
from cv2 import sqrt
from matplotlib import pyplot as plt
import numpy as np
import time

cap_check = cv2.VideoCapture(2) # Khai báo USB Camera Check Config
cap_check.set(3, 1920)
cap_check.set(4, 1080)

def check_val(mean):

    image = cv2.resize(mean,(1920,1080))
    start_row,start_col= 1080,446
    end_row,end_col= 1210,590
    cropped=image[start_col:end_col,start_row:end_row]
    gray_tray_1 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # ret, thresh1 = cv2.threshold(gray_tray_1, 120, 255, cv2.THRESH_BINARY)

    # cv2.imshow("ing", cropped)
    # cv2.waitKey(10000)
    
    histr1 = cv2.calcHist([gray_tray_1], [0], None, [256], [0, 256])
    # print(type(histr1))
    t = sum(histr1)
    dem = 0
    for i in range(len(histr1)):
        dem = dem + histr1[i]*(i+1)

    value = dem//t
    # print(histr1)
    # value = (min(histr1)+max(histr1))//2
    
    # print(value)

    # plt.subplot(121)
    # plt.imshow(thresh1)
    # plt.subplot(122)
    # plt.plot(histr1)
    # plt.show()
    return value

# 9122.96
#9145.55
#lech 18245
18291

count_l = 0
count_cr = 0
tb=0
his_color=[]
for i in range(300):
    # file_name = "check_lech/lech/image (" + str(i+1) + ").jpg"
    file_name = "samples/true/image (" + str(i+1) + ").jpg"
    img=cv2.imread(file_name)

    value_out = check_val(img)
    # print(value_out)
    his_color.append(value_out)
    # print(his_color)

mean_value = sum(his_color)//len(his_color)
# print(mean_value)

var_value = sqrt(sum((his_color - mean_value)*(his_color - mean_value))//299)
print(var_value)


def check_align(img):
    CI = 1.96
    value = check_val(img)
    if abs(value - mean_value) <= CI*var_value:
        return 1
    return 0

for i in range (10):
    ret, test_imgs = cap_check.read()
#################
index_false = 1051

##############
for i in range(50):
    ret, test_imgs = cap_check.read()
    start = time.time()
    result = check_align(test_imgs)
    # print("Predict: ", result)
    end = time.time()
    # result = result.tolist()
#t_tag la bien thoi gian, tag_t la tag true
    t_tag = end - start
    tag_t = str(index_false)+ "_kq_"+str(result)+"_" +str(round(t_tag,5))
    cv2.imwrite("result/Histogram/machine_result/"+tag_t+".jpg", test_imgs) # thay gia tri img thanh anh
    index_false += 1
    time.sleep(0.5)




