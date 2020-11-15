# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week4 exercise.

Use logistic regression to classify MNIST.
Created on Sat Sep 26 12:20:13 2020
@author: yooji
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from myLogReg import onevsAll, predictOnevsAll, displayData
from sklearn.linear_model import LogisticRegression


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
plt.show()

wait = input('Program paused. Press enter to continue.\n')

# %% Train the model
print('Training my logistic regression.\n')
lam = 0.1
all_theta = onevsAll(X, y, num_labels, lam, maxiter=500)

# Show results
pred = predictOnevsAll(all_theta, X)
y = y.reshape(-1,)
print('Training Accuraty: %0.2f\n' % np.mean(pred == y))
wait = input('Program paused. Press enter to continue.\n')

# %% Train using sci-kit learn
print('Training scikit-learn model.\n')
reg = LogisticRegression(multi_class='ovr', C=1/lam, max_iter=500)
reg = reg.fit(X, y)
print('Training Accuraty: %0.2f\n' % reg.score(X, y))
