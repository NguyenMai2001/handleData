
from pickletools import read_uint1
from sklearn.metrics import confusion_matrix
import vocabulary_helpers
from sklearn.model_selection import train_test_split
import pickle
import cv2
import time
import numpy as np


# test_imgs = cv2.imread("G:\\NCKH\\Xu_ly_anh\\1.jpg")
def check_align(test_imgs):

    start_row, start_col = 444 , 1086  #tọa độ để cắt ảnh
    end_row, end_col = 585, 1200
    new_img = test_imgs[start_row:end_row,start_col:end_col] #cắt ảnh
    # cv2.imshow("greh", new_img)
    # cv2.waitKey(0)

    X=[]
    X.append(new_img)

    (t_f_vocabulary, t_i_vocab) = vocabulary_helpers.generate_vocabulary(X)

    n_clusters = 1000

    kmeans = pickle.load(open("bov_pickle_1000.sav", 'rb'))

    test_data = vocabulary_helpers.generate_features(t_i_vocab, kmeans, n_clusters)

    with open('clf.pkl', 'rb') as f: #mở file kết quả từ mô hình học máy lưu vào biến clf
        clf = pickle.load(f)

    predict = clf.predict(test_data)

    # print("predicted:")
    # print(predict)

    # if predict[0] ==0:
    #     print("Lech")
    # else:
    #     print("ko lech")
 
    return predict[0],new_img

# 1 la khong lech, 0 la lech 

#################################

index_false = 1
#################################
for i in range(300):
    test_imgs = cv2.imread("test_data/image (" + str(i+1) + ").jpg")
    start = time.time()
    result, img = check_align(test_imgs)
    end = time.time()
    result = result.tolist()
#t_tag la bien thoi gian, tag_t la tag true
    t_tag = end - start
    tag_t = str(index_false)+ "_kq_"+str(result)+"_" +str(round(t_tag,3))
    cv2.imwrite("Mai_check_data/SVM/"+tag_t+".jpg", img) # thay gia tri img thanh anh
    index_false += 1
    time.sleep(0.5)