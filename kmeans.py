# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week8 exercise.

Functions for K-means clustering
Created on Fri Nov 13 21:32:54 2020
@author: yooji
"""

import numpy as np
import matplotlib.pyplot as plt
import util
import seaborn as sns


def findClosestCentroids(X, centroids):
    """Compute the centroid memberships for every example."""
    K, d = centroids.shape
    Xw = np.tile(X, (1, K))
    Y = (Xw - centroids.ravel())**2
    Z = np.sum(Y.reshape(len(X), K, d), axis=2)
    ind = np.argmin(Z, axis=1)
    return ind


def computeCentroids(X, idx, K):
    """Return new centroids by computing the means of the data points."""
    m, n = X.shape
    idx_w = np.zeros([len(idx), K])
    idx_w[:] = idx[:, np.newaxis]
    idx_w = idx_w == range(K)
    centroids = X.T.dot(idx_w)/np.sum(idx_w, axis=0)
    return centroids.T


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    """Run the K-Means algorithm on data matrix X."""
    if plot_progress:
        plt.figure()
    m, n = X.shape
    K, d = initial_centroids.shape
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros([m, 1])

    for i in range(max_iters):
        print('K-Means iteration %d/%d...' % ((i+1), max_iters))
        idx = findClosestCentroids(X, centroids)
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
#            wait = input('Press enter to continue.')
        centroids = computeCentroids(X, idx, K)
    return centroids, idx


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    """Display the progress of k-Means as it is running."""
    plotDataPoints(X, idx)
    plt.plot(centroids[:, 0], centroids[:, 1], 'kx')
    for j in range(len(centroids)):
        # plt.plot([centroids[j, 0], previous[j, 0]],
        #          [centroids[j, 1], previous[j, 1]], 'k')
        util.drawLine(centroids[j, :], previous[j, :], 'k')
    plt.title('Iteration number %d' % (i+1))


def plotDataPoints(X, idx):
    """Display the progress of k-Means as it is running."""
    # Plot the data
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=idx, legend=False)


def kMeansInitCentroids(X, K):
    """Initialize K centroids to be used in K-Means on the dataset X."""
    centroids = np.zeros([K, X.shape[1]])
    idx = np.random.permutation(X.shape[0])
    centroids = X[idx[:K], :]
    return centroids
