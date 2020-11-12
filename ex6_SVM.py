# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week7 exercise.

SVM parameter evaluation.
Created on Tue Nov 10 14:07:38 2020
@author: yooji
"""

from scipy.io import loadmat
import os
import util
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


def loadData(folder, FN):
    """Load data and plot."""
    data = loadmat(os.path.join(folder, FN))
    print([keys for keys in data.keys() if keys[0] != '_'])
    X = data['X']
    y = data['y']
    y = y.ravel()
    util.plotData(X, y)
    return X, y, data


def plotDecisionBoundary(X, clf, show_sv=False, ax=None):
    """Plot decision boundary and show support vectors if requested."""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='b', levels=0)
    # ax.contour(XX, YY, Z, colors='b', levels=[-1, 0, 1], alpha=0.5,
    #            linestyles=['--', '-', '--'])
    if show_sv:
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none', edgecolors='m')


def gridEvalParams(X, y, Xval, yval, C_set, sigma_set):
    """Evaluate SVM params and plot decision boundaries."""
    gamma_set = 0.5/sigma_set**2
    error = np.zeros([len(C_set), len(gamma_set)])
    fig, axes = plt.subplots(len(C_set), len(gamma_set),
                             figsize=(10, 8), sharex=True, sharey=True)

    for ci, C in enumerate(C_set):
        for gi, gamma in enumerate(gamma_set):
            clf = svm.SVC(C=C, gamma=gamma)
            clf.fit(X, y)
            y_ = clf.predict(Xval)
            error[ci, gi] = np.mean(y_ != yval)
            util.plotData(X, y, ax=axes[ci, gi])
            plotDecisionBoundary(X, clf, ax=axes[ci, gi])
            if ci == 0:
                axes[ci, gi].set_title('gamma=%0.2f' % gamma)
            if gi == 0:
                axes[ci, gi].set_ylabel('C=%0.2f' % C)
    ind = np.unravel_index(np.argmin(error, axis=None), error.shape)
    C = C_set[ind[0]]
    gamma = gamma_set[ind[1]]
    print("Optimal (C, gamma): %0.2f, %0.2f" % (C, gamma))
    print("Error matrix:\n", error)
    return C, gamma


# %% Load and train dataset 1 with linear kernel
folder = '..\\machine-learning-ex6\\ex6'

X, y, _ = loadData(folder, 'ex6data1.mat')
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)
plotDecisionBoundary(X, clf)

# %% Load and train dataset 2 with RBF kernel
X, y, _ = loadData(folder, 'ex6data2.mat')
clf = svm.SVC(gamma=50, C=1)
clf.fit(X, y)
plotDecisionBoundary(X, clf, True)

# Test a range of C and gamma
Xtrain, ytrain, Xval, yval = util.divideTrainSet(X, y)
C_set = np.asarray([0.3, 1, 3, 10])
sigma_set = np.asarray([0.01, 0.03, 0.1, 0.3, 1])
gridEvalParams(Xtrain, ytrain, Xval, yval, C_set, sigma_set)

# %% Load and train dataset 3 with RBF kernel
X, y, data = loadData(folder, 'ex6data3.mat')
Xval = data['Xval']
yval = data['yval']

# Test a range of C and gamma
gridEvalParams(X, y, Xval, yval, C_set, sigma_set)
