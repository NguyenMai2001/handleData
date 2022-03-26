import tensorflow as tf
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer


#doc file
pickle_in = open('X.pkl', 'rb')
X = pickle.load(pickle_in)

pickle_in = open('y.pkl', 'rb')
y = pickle.load(pickle_in)

X = X.astype('float')/255.0

# lb  = LabelBinarizer()
# y = lb.fit_transform(y)

saved_model = tf.keras.models.load_model('weights.h5')
result = saved_model.predict(X[:100])

# print(result)
# result = result.tolist()
# y = y.tolist()
print(X.shape)
print(result[10])
print(y[10])
print(type(result[10]))
print(type(y[10]))


num = 0
for i in range(len(X[:100])):
  if (result[i] > 0.5):
    result[i] = 1
  else:
    result[i] = 0
  if (result[i] != y[i]):
    num +=1

print(num)