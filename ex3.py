# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week4 exercise.

Use logistic regression to classify MNIST.
Created on Sat Sep 26 12:20:13 2020
@author: yooji
"""

import numpy as np
from scipy.io import loadmat
from myLogReg import onevsAll, predictOnevsAll, displayData


# %%
num_labels = 10         # 10 labels, from 1 to 10

FN = '..\\machine-learning-ex3\\ex3\\ex3data1.mat'
data = loadmat(FN)
X = data['X']
y = data['y']
y[y == 10] = 0

# Show some data first
m = X.shape[0]
ind = np.random.permutation(m)
X_sel = X[ind[:100]]
displayData(X_sel, 20)

# Train the model
lam = 0.1
all_theta = onevsAll(X, y, num_labels, lam)

# %%
# Show results
pred = predictOnevsAll(all_theta, X)
print('Training Accuraty: %0.2f' % np.mean(pred == y.flatten()))
