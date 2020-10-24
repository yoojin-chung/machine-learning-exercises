# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week2 exercise.

Functions for linear regression with gradient descent.
Created on Sun Sep 20 08:16:04 2020
@author: yooji
"""

import numpy as np


def computeCost(theta, X, y):
    """Compute cost function."""
    m = len(y)
    J = 1/(2*m)*((X.dot(theta)-y).T).dot((X.dot(theta)-y))
    return float(J)


def gradientDescent(X, y, theta, alpha, num_iters):
    """Perform gradient descent to learn theta."""
    m = len(y)
    J_hist = []
    for n in range(num_iters):
        theta = theta - alpha*1/m*(X.T).dot((X.dot(theta)-y))
        J_hist.append(computeCost(theta, X, y))
    return theta, J_hist


def featureNormalize(X):
    """Normalize features."""
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma


def estimate(test, theta, mu, sigma):
    """Calculate predicted value from test data and theta."""
    X_test = (test-mu)/sigma
    X_test = np.hstack((np.ones([test.shape[0], 1]), X_test))
    y_hat = X_test.dot(theta)
    return y_hat
