from types import new_class
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn import metrics
import tensorflow as tf
from pickletools import read_uint1
from sklearn.metrics import confusion_matrix
import vocabulary_helpers
from sklearn.model_selection import train_test_split
import pickle



with open('ROC/Label.txt', 'r') as f:
    CNN_lb = f.readlines() 

label_CNN=[]
for i in range(len(CNN_lb)):
    label_CNN.append(CNN_lb[i][0])

label_CNN = np.array(label_CNN)
label_CNN = label_CNN.astype('float')
# print(type(label_CNN))
# print(label_CNN.shape)

# data_CNN = []
# for i in range(2500):
#     path = "img/img (" + str(i+1) + ").jpg"
#     img = cv2.imread(path)
#     # img = img.astype('float')/255.0
#     data_CNN.append(img)

# data_LOG = []
# for i in range(2500):
#     path = "img/img (" + str(i+1) + ").jpg"
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     # img = img.astype('float')/255.0
#     # sample_img = img.reshape(-1,144*118)
#     data_LOG.append(img)

# with open('clf_SVM.pkl', 'rb') as f: #mở file kết quả từ mô hình học máy lưu vào biến clf
#         clf = pickle.load(f)
X=[]
for i in range(2500):
    path = "img/img (" + str(i+1) + ").jpg"
    img = cv2.imread(path)
    X.append(img)

(t_f_vocabulary, t_i_vocab) = vocabulary_helpers.generate_vocabulary(X)

n_clusters = 1000

kmeans = pickle.load(open("bov_pickle_1000.sav", 'rb'))

test_data = vocabulary_helpers.generate_features(t_i_vocab, kmeans, n_clusters)

 
print(len(test_data))

# pickle_out = open('X_ROC.pickel', 'wb')
# pickle.dump(data_CNN, pickle_out)
# pickle_out.close()

# data_LOG = np.array(data_LOG).reshape(-1,144*118)
# print(data_LOG.shape)
# pickle_out = open('X_ROC_LOG.pickel', 'wb')
# pickle.dump(data_LOG, pickle_out)
# pickle_out.close()

pickle_out = open('X_ROC_SVM.pickel', 'wb')
pickle.dump(test_data, pickle_out)
pickle_out.close()


