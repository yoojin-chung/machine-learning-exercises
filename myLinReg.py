# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week2 exercise.

Functions for linear regression with gradient descent.
Created on Sun Sep 20 08:16:04 2020
@author: yooji

Oct 29 2020
Add regularization to compustCost
Add computeGrad
"""

import numpy as np
from util import *
from scipy.optimize import fmin_bfgs


def computeCost(theta, X, y, lam=0):
    """Compute cost function."""
    m, n = X.shape
    theta = theta.reshape([n, 1])
    y = y.reshape([m, 1])
    h = X.dot(theta)
    J = 1/(2*m)*((h-y).T).dot((h-y)) + lam/(2*m)*(theta[1:].T).dot(theta[1:])
    return J


def computeGrad(theta, X, y, lam=0):
    """Compute gradient."""
    m, n = X.shape
    theta = theta.reshape([n, 1])
    y = y.reshape([m, 1])
    h = X.dot(theta)
    grad = 1/m*((X.T).dot(h-y)) + lam/m*np.vstack(([0], theta[1:]))
    return grad.flatten()


def trainLinearReg(X, y, lam):
    theta_init = np.zeros([X.shape[1], 1])
    theta = fmin_bfgs(computeCost,
                      theta_init,
                      computeGrad,
                      args=(X, y, lam),
                      maxiter=200)
    return theta


def gradientDescent(X, y, theta, alpha, num_iters):
    """Perform gradient descent to learn theta."""
    m = len(y)
    J_hist = []
    for n in range(num_iters):
        theta = theta - alpha*1/m*(X.T).dot((X.dot(theta)-y))
        J_hist.append(computeCost(theta, X, y))
    return theta, J_hist


def predict(test, theta, mu, sigma):
    """Calculate predicted value from test data and theta."""
    X_test = (test-mu)/sigma
    X_test = addOnes(X_test)
    y_hat = X_test.dot(theta)
    return y_hat


def learningCurve(X, y, Xval, yval, lam):
    m = len(X)
    error_train = np.zeros([m, 1])
    error_val = np.zeros([m, 1])

    for i in range(m):
        Xi_0 = addOnes(X[:i+1])
        theta = trainLinearReg(Xi_0, y[:i+1], lam)
        error_train[i] = computeCost(theta, Xi_0, y[:i+1], lam)
        error_val[i] = computeCost(theta, addOnes(Xval), yval, lam)
    return error_train, error_val


def validationCurve(X, y, Xval, yval, lambda_vec):   
    m = len(lambda_vec)
    error_train = np.zeros([m, 1])
    error_val = np.zeros([m, 1])
    
    for i in range(m):
        lam = lambda_vec[i]
        X0 = addOnes(X)
        theta = trainLinearReg(X0, y, lam)
        error_train[i] = computeCost(theta, X0, y, lam)
        error_val[i] = computeCost(theta, addOnes(Xval), yval, lam)
    return error_train, error_val