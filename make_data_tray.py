import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import glob
import sklearn
import pickle
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from numpy.core.fromnumeric import resize

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Detect(object):
    def __init__(self) -> None:
        super().__init__()

        self.image = []
        self.crop_tray_1 = []
        self.crop_tray_2 = []
        
        f = open(resource_path('data/config/location_crop_yn.txt'))
        self.t1_x_begin = int(f.readline()) # 202
        self.t1_y_begin = int(f.readline()) #321   
        self.t1_x_end = int(f.readline()) #429
        self.t1_y_end = int(f.readline())  #515    
        self.t2_x_begin = int(f.readline())
        self.t2_y_begin = int(f.readline())
        self.t2_x_end = int(f.readline())
        self.t2_y_end = int(f.readline())
       

    def find_location_crop(self, event, x, y, flags, param):
        f = open(resource_path('data/config/location_crop_yn.txt'), 'w')
        if event == cv2.EVENT_LBUTTONDOWN:
            f.write(str(x) + "\n")
            f.write(str(y) + "\n")
    
    def get_coord(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.find_location_crop)
        while True:
            cv2.imshow("image", self.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def thresh(self):
        
        self.crop_tray_1 = self.image[self.t1_y_begin:self.t1_y_end, self.t1_x_begin:self.t1_x_end]
        self.crop_tray_2 = self.image[self.t2_y_begin:self.t2_y_end, self.t2_x_begin:self.t2_x_end]


    def rotated(self, image):
        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-1, scale=1)
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        # cv2.imshow('Rotated image', rotated_image)
        # cv2.waitKey(0)
        # cv2.imwrite('rotated_image.png', rotated_image)
        return rotated_image


def add_crop_yes(crop_img, position):
    # cv2.imwrite('img_y.jpg', crop_img)
    # img = cv2.imread('img_y.jpg', cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    img = crop_img
    # plt.imshow(img)
    # plt.show()

    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros(48)
    arr = [position-1, position, position+1]
    for i in arr:
        k = int(i / 8)
        j = i % 8
        cut = img[int(height / 8 * (8 - j - 1)):int(height / 8 * (8 - j)), int(width / 6 * k):int(width / 6 *(k + 1))]
        cut1 = cv2.resize(cut,(31,27))
        train_tray.append([cut1, classes.index('yes')])
    # plt.imshow(cut, cmap='gray')
    # plt.show()

def add_crop_no(crop_img):
    # cv2.imwrite('img_n.jpg', crop_img)
    # img = cv2.imread('img_n.jpg', cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    img = crop_img

    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros(48)
    for i in range(48):
        k = int(i / 8)
        j = i % 8
        cut = img[int(height / 8 * (8 - j - 1)):int(height / 8 * (8 - j)), int(width / 6 * k):int(width / 6 *(k + 1))]
        cut1 = cv2.resize(cut,(31,27))
        train_tray.append([cut1, classes.index('no')])
        # plt.imshow(cut, cmap='gray')
        # plt.show()

classes = ['no', 'yes']
train_tray = []
X = []
y = []


if __name__ == "__main__":
    for i in range(10):
        path = "cam_init/ok (" + str(i+1) + ").png"
        img=cv2.imread(path)
        detect = Detect()
        detect.image = detect.rotated(img)
        detect.thresh()
        crop1 = detect.crop_tray_1
        add_crop_no(crop1)
        crop2 = detect.crop_tray_2
        add_crop_no(crop2)

    # print(len(train_tray))
    while(1):
        a = int(input())
        if (a==0):
            break   
        else:
            for i in range(73):
                path = "cam_check/case (" + str(i+1) + ")"
                for j in range(22):
                    img =path + "/state (" + str(j+1) + ").png"
                    img=cv2.imread(img)
                    if (j==0):
                        plt.imshow(img)
                        plt.show()
                        print("====nhap vi tri cat======")
                        position = int(input())
                        print('====nhap tray cat====')
                        tray = int(input())
                        print('===================================')
                    detect = Detect()
                    detect.image = detect.rotated(img)
                    detect.thresh()
                    if (tray == 1):
                        crop = detect.crop_tray_1
                        add_crop_yes(crop,position)

                    else:
                        crop= detect.crop_tray_2
                        add_crop_yes(crop,position)
                        
                img, label = train_tray[-1]
                plt.imshow(img)
                plt.show()
        
    # print(np.shape(train_tray))
    import random
    random.shuffle(train_tray)


    for img, label in train_tray:
        X.append(img)
        y.append(label)

    # print(np.shape(X[-1]))
    # print(type(X))
    # print(np.shape(X[-1]))

    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)



    def draw_sample_label(X,y,ypred=None):
        X = X[:12]
        y = y[:12]
        plt.subplots(3,4)
        for i in range(len(X)):
            plt.subplot(3,4,i+1)
            plt.imshow(X[i])
            if ypred is None:
                plt.title(f'y={y[i]}')
            else:
                plt.title(f'y={y[i]} ypred={ypred[i]}')
        plt.show()

    draw_sample_label(X,y)

   
    with open('X.pkl', 'wb') as f:
        pickle.dump(X, f)  

    with open('y.pkl', 'wb') as f:
        pickle.dump(y, f) 