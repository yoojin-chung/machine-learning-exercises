# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week9 exercise.

Anomaly Detection
Created on Sat Nov 14 15:00:46 2020
@author: yooji
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


def estimateGaussian(X):
    """Estimate the parameters of a Gaussian distribution in X."""
    m = len(X)
    mu = np.sum(X, axis=0)/m
    sigma2 = np.sum((X-mu)**2, axis=0)/m
    return mu, sigma2


def multivariateGaussian(X, mu, sigma2):
    """Compute the probability density function of the multivariate Gaussian distribution."""
    k = len(mu)
    # sigma2 = sigma2.reshape(-1, 1)
    # if sigma2.shape[1] == 1 | sigma2.shape[0] == 1:
    sigma2 = np.diag(sigma2)

    X = X - mu
    p = (2*np.pi)**(-k/2) * np.linalg.det(sigma2)**(-0.5) *\
        np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(sigma2))*X, axis=1))
    return p.reshape(-1, 1)


def visualizeFit(X,  mu, sigma2):
    """ Visualize the dataset and its estimated distribution."""
    x = np.arange(0, 35, 0.5)
    X1, X2 = np.meshgrid(x, x)
    Z = multivariateGaussian(
        np.hstack([X1.reshape(X1.size, 1), X2.reshape(X2.size, 1)]),
        mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')
    levels = 10 ** np.arange(-20, 0, 3).astype(float)
    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=levels)


def selectThreshold(yval, pval):
    """Find the best threshold (epsilon) to use for selecting outliers."""
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    steps = np.linspace(min(pval), max(pval), num=1000)
    for epsilon in steps[1:]:
        pred = pval < epsilon
        prec = np.sum(yval*pred)/np.sum(pred)
        rec = np.sum(yval*pred)/np.sum(yval)
        F1 = 2*prec*rec/(prec+rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1


# %% ================== Part 1: Load Example Dataset  ===================

print("Visualizing example dataset for outlier detection.\n")
data = loadmat('..\\machine-learning-ex8\\ex8\\ex8data1.mat')
print([keys for keys in data.keys() if keys[0] != '_'])
X = data['X']
Xval = data['Xval']
yval = data['yval']

# Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bx')
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

wait = input("Program paused. Press enter to continue.\n")

# %% ================= Part 2: Estimate the dataset statistics ==============

print('Visualizing Gaussian fit.\n')

#  Estimate my and sigma2
mu, sigma2 = estimateGaussian(X)

#  Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2)

#  Visualize the fit
visualizeFit(X,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

wait = input("Program paused. Press enter to continue.\n")

# %% ================== Part 3: Find Outliers ===================

pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)
print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set:  %f' % F1)
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)\n')

#  Find the outliers in the training set and plot the
outliers = np.argwhere(p < epsilon)

#  Draw a red circle around those outliers
plt.plot(X[outliers, 0], X[outliers, 1], 'ro')

# %% ================== Part 4: Multidimensional Outliers ===================

data = loadmat('..\\machine-learning-ex8\\ex8\\ex8data2.mat')
print([keys for keys in data.keys() if keys[0] != '_'])
X = data['X']
Xval = data['Xval']
yval = data['yval']

#  Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X)

#  Training set
p = multivariateGaussian(X, mu, sigma2)

#  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set:  %f' % F1)
print('   (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of 0.615385)')
print('# Outliers found: %d\n' % sum(p < epsilon))
