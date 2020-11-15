# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week3 exercise.

Run logistic regression with regularization.
Created on Sat Sep 26 15:13:12 2020
@author: yooji
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from sklearn.linear_model import LogisticRegression
import util
import myLogReg


def mapFeature(X1, X2):
    """Feature mapping function to polynomial features."""
    degree = 6
    out = np.ones([X1.shape[0], 1])
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.power(X1, i-j)*np.power(X2, j)))
    return out


def plotDecisionBoundary(theta, X, y, lam):
    """Plot the data points & decision boundary defined by theta."""
    # util.plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 1])-2,  max(X[:, 1])+2])
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])
        plt.plot(plot_x, plot_y)
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros([len(u), len(v)])
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(np.array([[ui]]),
                                            np.array([[vj]])), theta.T)
        plt.contour(u, v, z.T, levels=0)
        plt.xlim([-1, 1.25])
        plt.ylim([-1, 1.25])
        plt.title('lambda = %0.4f' % lam)
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')
        plt.legend(['y = 1', 'y = 0', 'Decision boundary'])


# %% Load and look at initial cost
FN = '..\\machine-learning-ex2\\ex2\\ex2data2.txt'
data = np.loadtxt(FN, delimiter=',')

# Print out some data points
print('First 10 examples from the dataset:')
print(data[:10])
print('')

# Reformat data & plot
X = data[:, :-1]
y = data[:, -1, None]
util.plotData(X, y)
wait = input('Program paused. Press enter to continue.\n')

# %% Optimize using fmin_bfgs
# Initial condition
print('Training my logistic regression.\n')
Xmap = mapFeature(X[:, [0]], X[:, [1]])
theta_init = np.zeros([Xmap.shape[1], 1])
lam = 1
cost = myLogReg.costFunc(theta_init, Xmap, y, lam)
grad = myLogReg.gradFunc(theta_init, Xmap, y, lam)

# Optimize
theta1 = fmin_bfgs(myLogReg.costFunc,
                   theta_init,
                   myLogReg.gradFunc,
                   args=(Xmap, y, lam),
                   maxiter=400)

# %% Show results
plotDecisionBoundary(theta1, Xmap, y, lam)
print('Trained theta - first 5 values: \n', theta1[:5])
p1 = myLogReg.predict(theta1, Xmap)
print('Train Accuracy: %0.2f\n' % np.mean(p1 == y))
plt.show()
wait = input('Program paused. Press enter to continue.\n')

# %% Try using sci-kit learn & show results
print('Training scikit-learn model.\n')
util.plotData(X, y)

reg = LogisticRegression(C=1/lam, max_iter=400)
y = y.reshape(-1,)
reg = reg.fit(Xmap, y)
theta2 = np.copy(reg.coef_.reshape(-1,))
theta2[0] = reg.intercept_
p2 = reg.predict(Xmap)

print('Trained theta - first 5 values: \n', theta2[:5])
plotDecisionBoundary(theta2, Xmap, y, lam)
# print('Train Accuracy: %0.2f\n' % reg.score(Xmap, y))
print('Train Accuracy: %0.2f\n' % np.mean(p2 == y))
plt.show()
