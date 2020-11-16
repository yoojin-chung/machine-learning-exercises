# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week6 exercise.

Run linear regression with regularization & evaluate learning curves
Created on Wed Oct 28 14:22:53 2020

@author: yooji
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import linear_model, preprocessing
from sklearn.model_selection import learning_curve, cross_val_score


def validationCurve(X, y, Xval, yval, lambda_vec):   
    m = len(lambda_vec)
    val_scores = np.zeros([m, 1])
    train_score = np.zeros([m, 1])

    for n, lam in enumerate(lambda_vec):
        reg = linear_model.Ridge(alpha=lam)
        reg = reg.fit(X, y)
        # val_scores[n] = cross_val_score(reg, Xval, yval).mean()
        val_scores[n] = reg.score(Xval, yval).mean()
        train_score[n] = reg.score(X, y)

    return train_score, val_scores


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
# Plot training data
fig = plt.figure()
plt.scatter(X, y, marker='x')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

# Train linear regression and see the result
lam = 0
# X0 = addOnes(X)
reg = linear_model.LinearRegression()
reg = reg.fit(X, y)

plt.scatter(X, reg.predict(X), marker='o')

plt.show()
wait = input('Program paused. Press enter to continue.\n')

# %% Plot learning curve for linear regression
# error_train, error_val = my.learningCurve(X, y, Xval, yval, lam)
Xall = np.vstack([X, Xval])
# Xall = addOnes(Xall)
yall = np.vstack([y, yval])
reg = linear_model.LinearRegression()
train_sizes, train_scores, valid_scores = learning_curve(
    reg, Xall, yall, train_sizes=range(1, 20))

fig = plt.figure()
plt.plot(train_sizes, 1-train_scores.mean(axis=1))
plt.plot(train_sizes, 1-valid_scores.mean(axis=1))
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')

plt.show()
wait = input('Program paused. Press enter to continue.\n')

# %%
p = 8
Xall = np.vstack([X, Xval, Xtest])
yall = np.vstack([y, yval, ytest]).reshape(-1,)

# X_poly = util.polyFeatures(Xall, p)
# X_poly, mu, sigma = util.featureNormalize(X_poly)
scaler = preprocessing.StandardScaler()
polyFeatures = preprocessing.PolynomialFeatures(p)
X_poly = polyFeatures.fit_transform(Xall)
X_poly = scaler.fit_transform(X_poly)

# X_poly_test = (polyFeatures(Xtest, p)-mu)/sigma
# X_poly_val = (polyFeatures(Xval, p)-mu)/sigma

print('Normalized Training Example 1:\n')
print(X_poly[0, :])

lam = 1
reg = linear_model.Ridge(alpha=lam)
reg = reg.fit(X_poly, yall)
# theta = np.hstack([reg.intercept_.reshape(-1,), reg.coef_])

fig = plt.figure()
plt.scatter(Xall, yall, marker='x')

# X_pred, y_pred = util.polyFit(min(X), max(X), mu, sigma, theta, p)
x_range = max(Xall) - min(Xall)
buf = x_range/20
X_pred = np.arange(min(Xall)-buf, max(Xall)+buf, x_range/500).reshape(-1, 1)
X_pred_poly = preprocessing.PolynomialFeatures(p).fit_transform(X_pred)
X_pred_poly = scaler.transform(X_pred_poly)
y_pred = reg.predict(X_pred_poly)

plt.plot(X_pred, y_pred)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = %0.2f)' % lam)
plt.show()

wait = input('Program paused. Press enter to continue.\n')
# %%

# error_train, error_val =\
#     my.learningCurve(X_poly, y, X_poly_val, yval, lam)

train_sizes, train_scores, valid_scores = learning_curve(
    reg, X_poly, yall, train_sizes=np.arange(1, 21, 2))

error_train = 1-train_scores.mean(axis=1)
error_val = 1-valid_scores.mean(axis=1)

fig = plt.figure()
plt.plot(train_sizes, error_train)
plt.plot(train_sizes, error_val)
plt.title('Polynomial Regression Learning Curve (lambda = %0.2f)' % lam)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()

print('# Training examples | Training error | Validation error:')
for n, train_size in enumerate(train_sizes):
    print('%0.0f | %0.4f | %0.4f' % (train_size, error_train[n], error_val[n]))

wait = input('Program paused. Press enter to continue.\n')

# %%
lambda_vec = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

X_poly = polyFeatures.transform(X)
X_poly = scaler.fit_transform(X_poly)
Xval_poly = polyFeatures.fit_transform(Xval)
Xval_poly = scaler.transform(Xval_poly)

train_score, val_scores =\
    validationCurve(X_poly, y, Xval_poly, yval, lambda_vec)

# %%
plt.figure()
plt.plot(lambda_vec, 1-train_score)
plt.plot(lambda_vec, 1-val_scores)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
# plt.xscale('log')
plt.show()

print('Lambda | Training error | Validation error:')
for n, lam in enumerate(lambda_vec):
    print('%0.4f | %0.4f | %0.4f' % (lam, 1-train_score[n], 1-val_scores[n]))
