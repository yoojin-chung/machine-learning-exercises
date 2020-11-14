# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week8 exercise.

K-means clustering
Created on Fri Nov 13 10:49:55 2020
@author: yooji
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import kmeans


# %% ================= Part 1: Find Closest Centroids ====================

# Load an example dataset that we will be using
data = loadmat('..\\machine-learning-ex7\\ex7\\ex7data2.mat')
# print([keys for keys in data.keys() if keys[0] != '_'])
X = data['X']

# Select an initial set of centroids
K = 3  # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the initial_centroids
idx = kmeans.findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples:')
print('%d, %d, %d' % tuple(idx[:3]+1))
print('(the closest centroids should be 1, 3, 2 respectively)')
wait = input("Program paused. Press enter to continue.")

# %% ===================== Part 2: Compute Means =========================

print('Computing centroids means.')

#  Compute means based on the closest centroids found in the previous part.
centroids = kmeans.omputeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: \n')
print(centroids)
print('\n(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]\n')

wait = input('Program paused. Press enter to continue.')

# %% =================== Part 3: K-Means Clustering ======================

print('Running K-Means clustering on example dataset.\n')

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'True' at the end tells our function to plot
# the progress of K-Means
centroids, idx = kmeans.runkMeans(X, initial_centroids, max_iters, True)
print('K-Means Done.\n')
wait = input('Program paused. Press enter to continue.')

# %% ============= Part 4: K-Means Clustering on Pixels ===============

print('Running K-Means clustering on pixels from an image.\n')

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

# When using K-Means, it is important the initialize the centroids randomly.
# You should complete the code in kMeansInitCentroids() before proceeding
initial_centroids = kmeans.kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = kmeans.unkMeans(X, initial_centroids, max_iters)

wait = input('Program paused. Press enter to continue.\n')

# %% ================= Part 5: Image Compression ======================

print('\nApplying K-Means to compress an image.\n')

# Find closest cluster members
idx = kmeans.findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx.

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
X_recovered = centroids[idx, :]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Display the original image
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(A)
ax1.set_title('Original')

# Display compressed image side by side
ax2.imshow(X_recovered)
ax2.set_title('Compressed, with %d colors.' % K)
