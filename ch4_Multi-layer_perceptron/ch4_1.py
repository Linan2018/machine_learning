# -*- coding: utf-8 -*-
"""ch4_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZbQFxSGvSWW1l4nNEgMVuTsyS0Iye5Pp
"""

cd drive/My\ Drive/ML_hw

pwd

import numpy as np
import copy
import random
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD

from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)



model = Sequential()

model.add(Dense(3, input_shape=((3,)), activation='tanh'))
model.add(Dense(3, activation='tanh'))
model.add(Dense(1, activation='tanh'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
print(model.summary())

x = np.array([[0, 0 ,0],[0, 0 ,1],[0, 1 ,0],[0, 1 ,1],[1, 0 ,0],[1, 0 ,1],[1, 1 ,0],[1, 1 ,1]])
y = np.array([[1], [-1], [-1], [1], [-1], [1], [1], [-1]])

model.fit(x, y, 8, epochs=100)

y_pr = model.predict(x)
print(y_pr,'\n',y)

xx = np.arange(-0.5,1.51,0.01)
yy = np.arange(-0.5,1.51,0.01)
zz = np.arange(-0.5,1.51,0.01)
grid = np.asarray(np.meshgrid(xx, yy, zz))

print(grid.shape)
grid = grid.transpose(1,2,3,0)
print(grid.shape)
t=np.concatenate(grid,axis=0) 
t=np.concatenate(t,axis=0) 
t.shape

t

y_t = model.predict(t).flatten()



t.shape

x = np.array([[0, 0 ,0],[0, 0 ,1],[0, 1 ,0],[0, 1 ,1],[1, 0 ,0],[1, 0 ,1],[1, 1 ,0],[1, 1 ,1]])
y = np.array([[1], [-1], [-1], [1], [-1], [1], [1], [-1]])

index = np.where((y_t > -6e-2) * (y_t < 6e-2))
# grid = grid[index][0]
# yy = yy[index]
# zz = zz[index]
fig = plt.figure()
ax = Axes3D(fig)
print(index[0].shape)
ax.scatter(t[index[0]][:, 0], t[index[0]][:, 1], t[index[0]][:, 2], c='g', s=2, alpha=0.05)

y

dir(ax)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_zlim(-0.5, 1.5)
for i, p in enumerate(x):
  # print(p, y[i][0])
  if y[i][0] == 1:
    ax.scatter(p[0], p[1], p[2], c='b')
  else:
    ax.scatter(p[0], p[1], p[2], c='r')
ax.scatter(t[index[0]][:, 0], t[index[0]][:, 1], t[index[0]][:, 2], c='g', s=2, alpha=0.05)
plt.savefig('4_1.png')

plt.subplot(131)

plt.scatter(t[index[0]][:, 0], t[index[0]][:, 1], c='g', s=2, alpha=0.2)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

plt.subplot(132)
plt.scatter(t[index[0]][:, 0], t[index[0]][:, 2], c='g', s=2, alpha=0.2)
plt.xlabel('x')
plt.ylabel('z')
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

plt.subplot(133)
plt.scatter(t[index[0]][:, 1], t[index[0]][:, 2], c='g', s=2, alpha=0.2)
plt.xlabel('y')
plt.ylabel('z')
plt.tight_layout()
ax = plt.gca()
ax.set_aspect(1)

ax.contourf(x[:, 0], x[:, 1], x[:, 2], rstride = 1)      # 渐变颜色

x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)

z.shape

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x,y,z)
plt.show()

