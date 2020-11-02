# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:22:53 2020

@author: yooji
"""

import numpy as np
import matplotlib.pyplot as plt
import myLinReg as my
from util import *
from scipy.io import loadmat


FN1 = '..\\machine-learning-ex5\\ex5\\ex5data1.mat'
data = loadmat(FN1)

# Show keys
print('keys in ex5data1:', [key for key in data.keys() if key[0] != '_'])
X = data['X']
y = data['y']
Xtest = data['Xtest']
ytest = data['ytest']
Xval = data['Xval']
yval = data['yval']


# %%
# Number of examples
m = len(X)

# Plot training data
fig = plt.figure()
plt.scatter(X, y, marker='x')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

# Check cost and gradient
X0 = addOnes(X)
theta = np.ones([2, 1])
J = my.computeCost(theta, X0, y, 1)
print("Cost at theta [1, 1]: %f:\
      \n(this value should be about 303.993192)\n" % J)

grad = my.computeGrad(theta, X0, y, 1)
print("Gradient at theta [1, 1]: %f, %f:\
      \n(this value should be about [-15.303016; 598.250744])\n" % tuple(grad))

# %%
# Train linear regression and see the result
lam = 0
theta = my.trainLinearReg(X0, y, lam)
plt.scatter(X, X0.dot(theta), marker='o')

# Plot learning curve for linear regression
error_train, error_val = my.learningCurve(X, y, Xval, yval, lam)

fig = plt.figure()
plt.plot(error_train)
plt.plot(error_val)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')

# %%
p = 8
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = addOnes(X_poly)

X_poly_test = (polyFeatures(Xtest, p)-mu)/sigma
X_poly_test = addOnes(X_poly_test)

X_poly_val = (polyFeatures(Xval, p)-mu)/sigma
X_poly_val = addOnes(X_poly_val)

print('Normalized Training Example 1:\n')
print(X_poly[0, :])

lam = 1
theta = my.trainLinearReg(X_poly, y, lam)

fig = plt.figure()
plt.scatter(X, y, marker='x')
X_pred, y_pred = polyFit(min(X), max(X), mu, sigma, theta, p)
plt.plot(X_pred, y_pred)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = %0.2f)' % lam)

error_train, error_val =\
    my.learningCurve(X_poly, y, X_poly_val, yval, lam)

fig = plt.figure()
plt.plot(error_train)
plt.plot(error_val)
plt.title('Polynomial Regression Learning Curve (lambda = %0.2f)' % lam)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')

print('# Training examples | Training error | Validation error:')
for n in range(m):
    print('%0.0f | %0.4f | %0.4f' % (n+1, error_train[n], error_val[n]))

# %%
lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
error_train, error_val =\
    my.validationCurve(X_poly, y, X_poly_val, yval, lambda_vec)

plt.figure()    
plt.plot(lambda_vec, error_train) 
plt.plot(lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
#plt.xscale('log')

print('Lambda | Training error | Validation error:')
for n in range(len(lambda_vec)):
    print('%0.4f | %0.4f | %0.4f' % (lambda_vec[n], error_train[n], error_val[n]))
    