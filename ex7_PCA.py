# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week8 exercise.

PCA
Created on Fri Nov 13 18:00:34 2020
@author: yooji
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import util
import numpy as np
import myLogReg
import kmeans


def pca(X):
    """Run principal component analysis on the dataset X."""
    m, n = X.shape
    Sigma = X.T.dot(X)/m
    U, S, V = np.linalg.svd(Sigma)
    return U, S


def projectData(X, U, K):
    """Compute reduced data representation projecting only on to the kth eigenvectors."""
    Z = X.dot(U[:, :K])
    return Z


def recoverData(Z, U, K):
    """Recover approximation of original data using the projected data."""
    X_rec = Z.dot(U[:, :K].T)
    return X_rec


# %% ================== Part 1: Load Example Dataset  ===================

print('Visualizing example dataset for PCA.\n')
data = loadmat('..\\machine-learning-ex7\\ex7\\ex7data1.mat')
print([keys for keys in data.keys() if keys[0] != '_'])
X = data['X']

plt.figure(figsize=([4, 4]))
plt.scatter(X[:, 0], X[:, 1])
plt.xlim([0.5, 6.5])
plt.ylim([2, 8])

wait = input('Program paused. Press enter to continue.\n')

# %% =============== Part 2: Principal Component Analysis ===============

print('\nRunning PCA on example dataset.\n')

# Before running PCA, it is important to first normalize X
X_norm, mu, sigma = util.featureNormalize(X)

# Run PCA
[U, S] = pca(X_norm)

util.drawLine(mu, mu + 1.5 * S[0] * U[:, 0], '--k')
util.drawLine(mu, mu + 1.5 * S[1] * U[:, 1], '--k')

# %% =================== Part 3: Dimension Reduction ===================

print('Dimension reduction on example dataset.\n')

# Plot the normalized dataset (returned from pca)
plt.figure(figsize=([4, 4]))
plt.scatter(X_norm[:, 0], X_norm[:, 1])
plt.xlim([-4, 3])
plt.ylim([-4, 3])

# Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: %f' % Z[0])
print('(this value should be about 1.481274)\n')

X_rec = recoverData(Z, U, K)
print('Approximation of the first example: %f %f' % (X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)\n')

# Draw lines connecting the projected points to the original points
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
for i in range(len(X_norm)):
    util.drawLine(X_norm[i, :], X_rec[i, :], '--k')

wait = input('Program paused. Press enter to continue.\n')

# %% =============== Part 4: Loading and Visualizing Face Data =============

print('Loading face dataset.\n')

#  Load Face dataset
data = loadmat('..\\machine-learning-ex7\\ex7\\ex7faces.mat')
X = data['X']

#  Display the first 100 faces in the dataset
myLogReg.displayData(X[:100, :])

wait = input('Program paused. Press enter to continue.\n')

# %% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
print('\nRunning PCA on face dataset.\n')

X_norm, mu, sigma = util.featureNormalize(X)
U, S = pca(X_norm)
myLogReg.displayData(U[:, :36].T)

wait = input('Program paused. Press enter to continue.\n')

# %% ============= Part 6: Dimension Reduction for Faces =================

print('Dimension reduction for face dataset.\n')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print('%d %d' % Z.shape)

wait = input('Program paused. Press enter to continue.\n')

# %% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====

X_rec = recoverData(Z, U, K)

# Display the original image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)
myLogReg.displayData(X[:100, :], ax=ax1)
ax1.set_title('Original')

# Display reconstructed data from only k eigenfaces
myLogReg.displayData(X_rec[:100, :], ax=ax2)
ax2.set_title('Recovered faces')

wait = input('Program paused. Press enter to continue.\n')

# %% === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===

data = loadmat("..\\machine-learning-ex7\\ex7\\bird_small.mat")
A = data["A"]

A = A / 255  # Divide by 255 so that all values are in the range 0 - 1
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3)
K = 16
max_iters = 10
initial_centroids = kmeans.kMeansInitCentroids(X, K)
centroids, idx = kmeans.runkMeans(X, initial_centroids, max_iters)

sel = np.random.rand(1000, 1) * len(X)
sel = sel.astype(int)
sel = sel.reshape(-1,)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=idx[sel])
ax.set_title('Pixel dataset plotted in 3D. Color shows centroid memberships')

wait = input('Program paused. Press enter to continue.\n')

# %% === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===

X_norm, mu, sigma = util.featureNormalize(X)
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

plt.figure(figsize=(4, 4))
kmeans.plotDataPoints(Z[sel, :], idx[sel])
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
