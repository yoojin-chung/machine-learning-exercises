# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week8 exercise.

PCA
Created on Fri Nov 13 18:00:34 2020
@author: yooji
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import util
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing


# %% ================== Part 1: Load Example Dataset  ===================

print('Visualizing example dataset for PCA.')
data = loadmat('..\\machine-learning-ex7\\ex7\\ex7data1.mat')
print([keys for keys in data.keys() if keys[0] != '_'])
X = data['X']

plt.figure(figsize=([4, 4]))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.xlim([0.5, 6.5])
plt.ylim([2, 8])
# plt.xlim([-6, 8])
# plt.ylim([-5, 9])

wait = input('Program paused. Press enter to continue.\n')

# %% =============== Part 2: Principal Component Analysis ===============

print('Running PCA on example dataset.')

# Before running PCA, it is important to first normalize X
# X_norm, mu, sigma = util.featureNormalize(X)
X_norm = preprocessing.scale(X)
mu = np.mean(X, 0)

# Run PCA
model = PCA()
model.fit(X_norm)
S = model.singular_values_
S = S/np.linalg.norm(S)
U = model.components_
util.drawLine(mu, mu + 1.5 * S[0] * U[:, 0], '--k')
util.drawLine(mu, mu + 1.5 * S[1] * U[:, 1], '--k')
plt.show()

wait = input('Program paused. Press enter to continue.\n')

# %% =================== Part 3: Dimension Reduction ===================

print('Dimension reduction on example dataset.')

# Plot the normalized dataset (returned from pca)
plt.figure(figsize=([4, 4]))
plt.scatter(X_norm[:, 0], X_norm[:, 1], alpha=0.5)
plt.xlim([-4, 3])
plt.ylim([-4, 3])

# Project the data onto K = 1 dimension
K = 1
model = PCA(n_components=K)
Z = model.fit_transform(X_norm)
print('Projection of the first example: %f' % Z[0])
print('(this value should be about 1.481274)\n')

X_rec = model.inverse_transform(Z)
print('Approximation of the first example: %f %f' % (X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)\n')

# Draw lines connecting the projected points to the original points
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro', alpha=0.5)
for i in range(len(X_norm)):
    util.drawLine(X_norm[i, :], X_rec[i, :], '--k')
plt.show()

wait = input('Program paused. Press enter to continue.\n')

# %% =============== Part 4: Loading and Visualizing Face Data =============

print('Loading face dataset.')

#  Load Face dataset
data = loadmat('..\\machine-learning-ex7\\ex7\\ex7faces.mat')
X = data['X']

#  Display the first 100 faces in the dataset
util.displayData(X[:100, :])
plt.show()

wait = input('Program paused. Press enter to continue.\n')

# %% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
print('Running PCA on face dataset.\n')

# X_norm, mu, sigma = util.featureNormalize(X)
X_norm = preprocessing.scale(X)

# Run PCA
model = PCA()
model.fit(X_norm)
U = model.components_
util.displayData(U[:36, :])
plt.show()

wait = input('Program paused. Press enter to continue.\n')

# %% ============= Part 6: Dimension Reduction for Faces =================

print('Dimension reduction for face dataset.')

K = 100
model = PCA(n_components=K, svd_solver='full')
Z = model.fit_transform(X_norm)

print('The projected data Z has a size of: ')
print('%d %d' % Z.shape)

wait = input('Program paused. Press enter to continue.\n')

# %% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====

X_rec = model.inverse_transform(Z)

# Display the original image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
util.displayData(X[:100, :], ax=ax1)
ax1.set_title('Original')

# Display reconstructed data from only k eigenfaces
util.displayData(X_rec[:100, :], ax=ax2)
ax2.set_title('Recovered faces')
plt.show()

wait = input('Program paused. Press enter to continue.\n')

# %% === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===

data = loadmat("..\\machine-learning-ex7\\ex7\\bird_small.mat")
A = data["A"]

A = A / 255  # Divide by 255 so that all values are in the range 0 - 1
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3)
K = 16
max_iters = 10
# Run K-Means
classifier = KMeans(n_clusters=K,
                    init='random',
                    n_init=1,
                    max_iter=max_iters)
classifier.fit(X)
centroids = classifier.cluster_centers_
idx = classifier.predict(X)

sel = np.random.rand(1000, 1) * len(X)
sel = sel.astype(int)
sel = sel.reshape(-1,)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=idx[sel])
ax.set_title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show()

wait = input('Program paused. Press enter to continue.\n')

# %% === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===

# X_norm, mu, sigma = util.featureNormalize(X)
X_norm = preprocessing.scale(X)
model = PCA(n_components=2)
Z = model.fit_transform(X_norm)

plt.figure(figsize=(8, 8))
util.plotDataPoints(Z[sel, :], idx[sel])
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show()
