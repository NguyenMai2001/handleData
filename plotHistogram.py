import matplotlib.pyplot as plt
import cv2
cam_dung = 'dtan (1).jpg' #1,30
cam_lech ='dtan (31).jpg' 
img_true = cv2.imread(cam_dung)
img_false = cv2.imread(cam_lech)

img_true = cv2.resize(img_true,(1920,1080))
img_false = cv2.resize(img_false,(1920,1080))
start_row,start_col= 1080,446
end_row,end_col= 1210,590
crop1=img_true[start_col:end_col,start_row:end_row]
crop2=img_false[start_col:end_col,start_row:end_row]
gray_tray_1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
gray_tray_2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

# histr1 = cv2.calcHist([gray_tray_1],[0],None,[256],[0,256])
# histr2 = cv2.calcHist([gray_tray_2],[0],None,[256],[0,256])

plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.hist(gray_tray_1.ravel(),256,[0,256],color='g')
plt.axis([0, 255, 0, 500])
plt.xlabel('Cường độ',fontsize = 18)
plt.ylabel('Tần suất',fontsize = 18)
plt.title('Camera không lệch',fontsize = 20)
plt.tick_params(labelsize=12)
plt.subplot(1,2,2)
plt.hist(gray_tray_2.ravel(),256,[0,256],color='r')
plt.axis([0, 255, 0, 500])
plt.xlabel('Cường độ',fontsize = 18)
plt.ylabel('Tần suất',fontsize = 18)
plt.title('Camera lệch',fontsize = 20)
plt.tick_params(labelsize=12)
plt.show()

# histr1 = cv2.calcHist([gray_tray_2], [0], None, [256], [0, 256])
# # print(type(histr1))
# t = sum(histr1)
# dem = 0
# for i in range(len(histr1)):
#     dem = dem + histr1[i]*(i+1)

# value = dem/t
# print(value)
