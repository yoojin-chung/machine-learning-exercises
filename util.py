# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:46:45 2020

@author: yooji
"""

import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    """Plot data points."""
    pos = (y == 1).flatten()
    neg = (y == 0).flatten()
    plt.plot(X[pos, 0], X[pos, 1], 'k+')
    plt.plot(X[neg, 0], X[neg, 1], 'ko')
    

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
    X = X.reshape([len(X), 1])
    Xp = np.tile(X, [1, p])
    pn = np.tile(np.arange(1, p+1), [len(X), 1])
    out = Xp**pn
    return out


def polyFit(min_x, max_x, mu, sigma, theta, p):
    theta = theta.reshape([len(theta), 1])
    step = (max_x - min_x)/500
    buf = (max_x - min_x)/10
    X = np.arange(min_x-buf, max_x+buf, step)
    Xp = (polyFeatures(X, p)-mu)/sigma
    Xp = addOnes(Xp)
    return X, Xp.dot(theta)
     

def addOnes(X):
    return np.hstack([np.ones([len(X), 1]), X])