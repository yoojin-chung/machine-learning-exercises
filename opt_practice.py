# -*- coding: utf-8 -*-
"""
Playing around with scipy.optimize.

Created on Wed Sep 23 21:17:41 2020
@author: yooji
"""

import numpy as np
from scipy.optimize import fmin_bfgs


def f1(X):
    """Test func takes in one array."""
    out = X[0]**3 + X[1]**2 + np.dot(X[1:].T, X[1:])
    return out.flatten()


init_X = np.ones([5, 1])
fmin_bfgs(f1, init_X, maxiter=400)


# %%
def f2(X, param):
    """Test func takes in two array."""
    X = X.reshape([len(X), 1])
    X = X*param
    out = X[0]**3 + X[1]**2 + np.dot(X[1:].T, X[1:])
    return out.flatten()


def f3(X):
    """Remove other args from f2."""
    return f2(X, param)


fmin_bfgs(f3, init_X, maxiter=400)  # This works

# %%
param = np.array([1, 2, 3, 4, 5])
param = param.reshape([len(param), 1])
init_X = np.ones([5, 1])

fmin_bfgs(f2, init_X, args=(param, ), maxiter=400)  # Pass other args
