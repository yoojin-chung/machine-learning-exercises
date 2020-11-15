# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week8 exercise.

K-means clustering using scikit-learn
Created on Fri Nov 13 10:49:55 2020
@author: yooji
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import util


# %% =================== Part 3: K-Means Clustering ======================

# Load an example dataset that we will be using
data = loadmat('..\\machine-learning-ex7\\ex7\\ex7data2.mat')
# print([keys for keys in data.keys() if keys[0] != '_'])
X = data['X']

print('Running K-Means clustering on example dataset.\n')

# Settings for running K-Means
K = 3
max_iters = 10

# Set centroids to the same specific values
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm
classifier = KMeans(n_clusters=K,
                    init=initial_centroids,
                    n_init=1,
                    max_iter=max_iters)
classifier.fit(X)
idx = classifier.predict(X)
centroids = classifier.cluster_centers_

# Plot clustering result
util.plotDataPoints(X, idx)
plt.plot(centroids[:, 0], centroids[:, 1], 'kx')
plt.show()
print('K-Means Done.')
wait = input('Program paused. Press enter to continue.\n')

# %% ============= Part 4: K-Means Clustering on Pixels ===============

print('Running K-Means clustering on pixels from an image.')

#  Load an image of a bird
data = loadmat("..\\machine-learning-ex7\\ex7\\bird_small.mat")
A = data["A"]

A = A / 255  # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(img_size[0] * img_size[1], 3)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# Run K-Means
classifier = KMeans(n_clusters=K,
                    init='random',
                    n_init=1,
                    max_iter=max_iters)
classifier.fit(X)
centroids = classifier.cluster_centers_

wait = input('Program paused. Press enter to continue.\n')

# %% ================= Part 5: Image Compression ======================

print('Applying K-Means to compress an image.')

# Find closest cluster members
idx = classifier.predict(X)

# Recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
X_recovered = centroids[idx, :]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Display the original image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 16))
ax1.imshow(A)
ax1.set_title('Original')
ax1.set_xticklabels([])
ax1.set_yticklabels([])

# Display compressed image side by side
ax2.imshow(X_recovered)
ax2.set_title('Compressed, with %d colors.' % K)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
plt.show()
