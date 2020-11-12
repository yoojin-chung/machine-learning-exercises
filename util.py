# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:46:45 2020

@author: yooji
"""

import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y, ax=None):
    """Plot data points."""
    pos = (y == 1).flatten()
    neg = (y == 0).flatten()
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.plot(X[pos, 0], X[pos, 1], 'k+')
    ax.plot(X[neg, 0], X[neg, 1], 'g.')


def featureNormalize(X):
    """Normalize features."""
    mu = np.mean(X, 0)
    sigma = np.std(X, 0, ddof=1)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma


def randInitWe(x, y):
    """Randomly initialize weights."""
    eps = 1.e-4
    W = 2*eps*np.random.randn(x, y) - 2*eps
    return W


def polyFeatures(X, p):
    """Feature mapping function to polynomial features."""
    X = X.reshape(-1, 1)
    Xp = np.tile(X, [1, p])
    pn = np.tile(np.arange(1, p+1), [len(X), 1])
    out = Xp**pn
    return out


def polyFit(min_x, max_x, mu, sigma, theta, p):
    theta = theta.reshape(-1, 1)
    step = (max_x - min_x)/500
    buf = (max_x - min_x)/10
    X = np.arange(min_x-buf, max_x+buf, step)
    Xp = (polyFeatures(X, p)-mu)/sigma
    Xp = addOnes(Xp)
    return X, Xp.dot(theta)


def addOnes(X):
    return np.hstack([np.ones([len(X), 1]), X])


def divideTrainSet(X, y, frac=0.8):
    m = len(X)
    ind = np.random.permutation(m)
    Xtrain = X[:ind[int(m*frac)]]
    ytrain = y[:ind[int(m*frac)]]
    Xval = X[ind[int(m*frac)]:]
    yval = y[ind[int(m*frac)]:]
    return Xtrain, ytrain, Xval, yval
