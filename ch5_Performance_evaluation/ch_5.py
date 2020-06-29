# -*- coding: utf-8 -*-
"""ch_5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13BTnb47nF7gBkgxex9qn4VHA_agkUmD_
"""

cd drive/My\ Drive/ML_hw

pwd

import numpy as np
import copy
import random
import matplotlib.pyplot as plt

np.random.seed(0)

def gen_n_data():
  cov = np.array([[1., 0.], [0., 1.]])

  x_p = np.random.multivariate_normal([2., 3.], cov, 200)
  x_n = np.random.multivariate_normal([5., 6.], cov, 800)
  
  x = np.vstack((x_p, x_n))
  y = np.vstack(([[1.] for _ in range(200)], [[0.] for _ in range(800)]))

  return np.hstack((x, y))

xy = gen_n_data()
np.random.shuffle(xy)
xy
# y = y.reshape((len(y), 1))

def draw(x, y, m='.'):
  p = []
  n = []
  for i in range(len(x)):
    if y[i] > 0.5:
      p.append(x[i])
    else:
      n.append(x[i])
  p = np.asarray(p)
  n = np.asarray(n)
  if p.size > 0:
    plt.scatter(p[:, 0], p[:, 1], c='b', marker=m)
  if n.size > 0:
    plt.scatter(n[:, 0], n[:, 1], c='r', marker=m)

test_x, test_y = xy[:200, :2], xy[:200, 2:]
val_x, val_y = xy[200:400, :2], xy[200:400, 2:]
x, y = xy[400:, :2], xy[400:, 2:]

draw(x, y, '.')
draw(val_x, val_y, '.')
draw(test_x, test_y, '.')
plt.savefig('5_raw.svg')

draw(val_x, val_y, '.')

x.shape, y.shape



import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD,Adam

from keras import backend as K

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
 
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)

def t(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return true_positives
def tt(y_true, y_pred):
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return possible_positives

model1 = Sequential()

model1.add(Dense(10, input_shape=((2,)), activation='sigmoid'))
model1.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.001, decay=1e-6)
model1.compile(loss='mse', optimizer=sgd, metrics=[t, tt, 'accuracy', recall, precision, f1])

print(model1.summary())

history = model1.fit(x, y, validation_data=(val_x, val_y), epochs=200, batch_size=64)

epochs=range(len(history.history['accuracy']))

fig = plt.figure(figsize=(16, 4))
# ax = fig.add_subplot(111)
# ax.set_aspect('equal', adjustable='box')


plt.subplot(141)

plt.plot(epochs,history.history['accuracy'],'b',label='Training acc')
plt.plot(epochs,history.history['val_accuracy'],'r',label='Validation acc')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.tight_layout()
# ax = plt.gca()
# ax.set_aspect(1)

plt.subplot(142)
plt.plot(epochs,history.history['recall'],'b',label='Training recall')
plt.plot(epochs,history.history['val_recall'],'r',label='Validation recall')
plt.title('Traing and Validation recall')
plt.legend()
plt.tight_layout()
# ax = plt.gca()
# ax.set_aspect(1)

plt.subplot(143)
plt.plot(epochs,history.history['precision'],'b',label='Training precision')
plt.plot(epochs,history.history['val_precision'],'r',label='Validation precision')
plt.title('Traing and Validation precision')
plt.legend()
plt.tight_layout()
# ax = plt.gca()
# ax.set_aspect(1)

plt.subplot(144)
plt.plot(epochs,history.history['f1'],'b',label='Training f1-score')
plt.plot(epochs,history.history['val_f1'],'r',label='Validation f1-score')
plt.title('Traing and Validation f1-score')
plt.legend()
plt.tight_layout()
# ax = plt.gca()
# ax.set_aspect(1)
plt.savefig('5_1_result.svg')




plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.savefig('5_1_loss.svg')

def convert(a):
  return np.asarray(a > 0.5, dtype=np.float64)

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')

y_pr = model1.predict(x)

plt.subplot(131)
plt.title('train data')
plt.xlim([-2, 9])
plt.ylim([0, 9])
draw(x,y)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

y_pr_val = model1.predict(val_x)
plt.subplot(132)
plt.xlim([-2, 9])
plt.ylim([0, 9])
plt.title('val data')
draw(val_x, y_pr_val)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

y_pr_test = model1.predict(test_x)
plt.subplot(133)
plt.xlim([-2, 9])
plt.ylim([0, 9])
plt.title('test data')
draw(test_x, y_pr_test)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

plt.savefig('5_1_s.svg')

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

def roc(y_true, y_pr_):
  # 为每个类别计算ROC曲线和AUC

  fpr, tpr, _ = roc_curve(y_true.flatten(), y_pr_.flatten(), pos_label=1)
  print(fpr, tpr, _)
  roc_auc = auc(fpr, tpr)
  # fpr[0].shape==tpr[0].shape==(21, ), fpr[1].shape==tpr[1].shape==(35, ), fpr[2].shape==tpr[2].shape==(33, ) 
  # roc_auc {0: 0.9118165784832452, 1: 0.6029629629629629, 2: 0.7859477124183007}

  # plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
          lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc="lower right")
  # plt.show()

y_pr

print(y.shape, y_pr.shape)
print(val_y.shape, y_pr_val.shape)

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')


plt.subplot(131)
plt.title('ROC of train data')
roc(y, y_pr)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

plt.subplot(132)
plt.title('ROC of val data')
roc(val_y, y_pr_val)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

plt.subplot(133)
plt.title('ROC of test data')
roc(test_y, y_pr_test)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)
plt.savefig('5_1_auc.svg')



y_pr_test

model1 = Sequential()

model1.add(Dense(10, input_shape=((2,)), activation='sigmoid'))
model1.add(Dense(10, activation='sigmoid'))
model1.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.001, decay=1e-6)
model1.compile(loss='mse', optimizer=sgd, metrics=[t, tt, 'accuracy', recall, precision, f1])

print(model1.summary())

history = model1.fit(x, y, validation_data=(val_x, val_y), epochs=200, batch_size=64)

epochs=range(len(history.history['accuracy']))

fig = plt.figure(figsize=(16, 4))
# ax = fig.add_subplot(111)
# ax.set_aspect('equal', adjustable='box')


plt.subplot(141)

plt.plot(epochs,history.history['accuracy'],'b',label='Training acc')
plt.plot(epochs,history.history['val_accuracy'],'r',label='Validation acc')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.tight_layout()
# ax = plt.gca()
# ax.set_aspect(1)

plt.subplot(142)
plt.plot(epochs,history.history['recall'],'b',label='Training recall')
plt.plot(epochs,history.history['val_recall'],'r',label='Validation recall')
plt.title('Traing and Validation recall')
plt.legend()
plt.tight_layout()
# ax = plt.gca()
# ax.set_aspect(1)

plt.subplot(143)
plt.plot(epochs,history.history['precision'],'b',label='Training precision')
plt.plot(epochs,history.history['val_precision'],'r',label='Validation precision')
plt.title('Traing and Validation precision')
plt.legend()
plt.tight_layout()
# ax = plt.gca()
# ax.set_aspect(1)

plt.subplot(144)
plt.plot(epochs,history.history['f1'],'b',label='Training f1-score')
plt.plot(epochs,history.history['val_f1'],'r',label='Validation f1-score')
plt.title('Traing and Validation f1-score')
plt.legend()
plt.tight_layout()
# ax = plt.gca()
# ax.set_aspect(1)
plt.savefig('5_2_result.svg')




plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.savefig('5_2_loss.svg')

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')

y_pr = model1.predict(x)

plt.subplot(131)
plt.title('train data')
plt.xlim([-2, 9])
plt.ylim([0, 9])
draw(x,y)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

y_pr_val = model1.predict(val_x)
plt.subplot(132)
plt.xlim([-2, 9])
plt.ylim([0, 9])
plt.title('val data')
draw(val_x, y_pr_val)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

y_pr_test = model1.predict(test_x)
plt.subplot(133)
plt.xlim([-2, 9])
plt.ylim([0, 9])
plt.title('test data')
draw(test_x, y_pr_test)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)
plt.savefig('5_2_s.svg')
print(y.shape, y_pr.shape)
print(val_y.shape, y_pr_val.shape)

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')


plt.subplot(131)
plt.title('ROC of train data')
roc(y, y_pr)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

plt.subplot(132)
plt.title('ROC of val data')
roc(val_y, y_pr_val)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

plt.subplot(133)
plt.title('ROC of test data')
roc(test_y, y_pr_test)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)
plt.savefig('5_2_roc.svg')

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')


plt.subplot(131)
plt.title('ROC of train data')
roc(y, y_pr)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

plt.subplot(132)
plt.title('ROC of val data')
roc(val_y, y_pr_val)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

plt.subplot(133)
plt.title('ROC of test data')
roc(test_y, y_pr_test)
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)
plt.savefig('5_2_roc.svg')

