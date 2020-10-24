# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week3 exercise.

Functions for logistic regression with regularization.
Created on Wed Sep 23 11:57:51 2020
@author: yooji
"""

import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt


def costFunc(theta, X, y, lam):
    """Compute cost for logistic regression w/ regularization."""
    m, n = X.shape
    theta = theta.reshape([n, 1])
    y = y.reshape([m, 1])
    h = sigmoid(X.dot(theta))
    # J = 1/m*(-(y.T).dot(np.log(h))-((1-y).T).dot(np.log(1-h)))
    J = 1/m*(-(y.T).dot(np.log(h))-((1-y).T).dot(np.log(1-h))) +\
        lam/(2*m)*(theta[1:].T).dot(theta[1:])
    return J


def gradFunc(theta, X, y, lam):
    """Compute gradient for logistic regression w/ regularization."""
    m, n = X.shape
    theta = theta.reshape([n, 1])
    y = y.reshape([m, 1])
    h = sigmoid(X.dot(theta))
    # grad = 1/m*((X.T).dot(h-y))
    grad = 1/m*((X.T).dot(h-y)) + lam/m*np.vstack(([0], theta[1:]))
    return grad.flatten()


def sigmoid(z):
    """Compute sigmoid."""
    g = 1/(1+np.exp(-z))
    return g


def sigmoidGrad(z):
    """Return the gradient of the sigmoid function."""
    g = sigmoid(z)*(1-sigmoid(z))
    return g


def predict(theta, X):
    """Predict the outcome based on the learned theta."""
    p = sigmoid(np.dot(X, np.array([theta]).T)) >= 0.5
    return p


def onevsAll(X, y, num_labels, lam):
    """
    Train multiple logistic regression classifiers.

    Returns a matrix all_theta, where the i-th row corresponds to the
    classifier for label i
    """
    m, n = X.shape
    X = np.hstack((np.ones([m, 1]), X))
    all_theta = np.zeros([num_labels, n + 1])
    for k in range(num_labels):
        theta_init = np.zeros([n+1, 1])
        theta = fmin_cg(costFunc,
                        theta_init,
                        gradFunc,
                        args=(X, y == k, lam),
                        maxiter=200)
        all_theta[k, :] = theta
    return all_theta


def predictOnevsAll(all_theta, X):
    """Predict label for a trained one-vs-all classifier."""
    X = np.hstack((np.ones([X.shape[0], 1]), X))
    y = sigmoid(np.dot(X, all_theta.T))
    p = np.argmax(y, axis=1)
    return p


def displayData(X, width):
    """Display 2D data in a grid."""
    m, n = X.shape
    height = int(n / width)
    disp_rows = int(np.floor(np.sqrt(m)))
    disp_cols = int(np.ceil(m / disp_rows))
    pad = 1

    disp_array = np.ones([pad + disp_rows * (height + pad),
                          pad + disp_cols * (width + pad)])
    cnt = 0

    for j in range(disp_rows):
        for i in range(disp_cols):
            if cnt >= m:
                break
            max_val = max(abs(X[cnt, :]))
            h_start = pad + j * (height + pad)
            w_start = pad + i * (width + pad)

            disp_array[h_start:h_start + height, w_start:w_start + width] =\
                X[cnt, :].reshape(height, width) / max_val
            cnt += 1
    disp_array = disp_array.T

    plt.figure()
    plt.imshow(disp_array)
    return disp_array
