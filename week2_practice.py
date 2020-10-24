# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week2 exercise.

Run linear regression with gradient descent.
Created on Sat Sep 19 15:03:12 2020
@author: yooji
"""

import numpy as np
import matplotlib.pyplot as plt
import myLinReg as my
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# %%
def runGradDesc(FN, alpha, num_iters):
    """Load data and run gradient descent."""
    data = np.loadtxt(FN, delimiter=',')

    # Print out some data points
    print('First 10 examples from the dataset:')
    print(data[:10])
    print('')

    # Reformat data
    X = data[:, :-1]
    y = data[:, -1, None]
    m, n = X.shape

    # Scale features for multivariate case
    if n > 1:
        X_norm, _, _ = my.featureNormalize(X)
    else:
        X_norm = X

    # Add intercept term to X
    X_norm = np.hstack((np.ones([m, 1]), X_norm))

    # Initialize theta
    theta = np.zeros([n+1, 1])

    # Run gradient descent
    print('Running gradient descent ...\n')
    theta, J_hist = my.gradientDescent(X_norm, y, theta, alpha, num_iters)

    # Display gradient descent's result
    print('Theta computed from gradient descent:')
    print(['%0.2f' % x for x in theta])
    print('')

    return X, y, theta, J_hist


# %% Univariate linear regression exercise
# Load data
FN1 = '..\\machine-learning-ex1\\ex1\\ex1data1.txt'

X, y, theta, J_hist = runGradDesc(FN1, 0.01, 1500)

plt.scatter(np.array(X), np.array(y))
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profits in $10,100s')

X1 = np.hstack((np.ones([len(X), 1]), X))
plt.plot(X, X1.dot(theta), 'r')

test = [3.5, 7]

for n in range(2):
    # Estimate the profit for population of 3.5k and 7k
    profit = my.estimate(np.array([[test[n]]]), theta, 0, 1)
    show = np.array([test[n], profit])*10000
    print('For population = %d, we predict a profit of $%0.2f' % tuple(show))

# Try it with sci-kit learn
print('\nUsing sci-kit learn...\n')
reg1 = LinearRegression()
reg1 = reg1.fit(X, y)
print('Intercept: ', reg1.intercept_.flatten())
print('Coefficients: ', reg1.coef_.flatten())

for n in range(2):
    profit = reg1.predict(np.array([[test[n]]]))
    show = np.array([test[n], profit])*10000
    print('For population = %d, we predict a profit of $%0.2f' % tuple(show))

print('Coefficient of determination R^2 is %0.2f' % reg1.score(X, y))
print('Mean squared error is %0.2f\n' % mean_squared_error(reg1.predict(X), y))


# %% Multivariate linear regression exercise
FN2 = '..\\machine-learning-ex1\\ex1\\ex1data2.txt'

X, y, theta, J_hist = runGradDesc(FN2, 0.01, 400)

# Estimate the price of a 1650 sq-ft, 3 br house
test = [1650, 3]
X_norm, mu, sigma = my.featureNormalize(X)
price = my.estimate(np.array([test]), theta, mu, sigma)
test.append(price)

print('Predicted price of a %d sq-ft, %d br house (using gradient descent): \
$%0.2f' % tuple(test))

# Try it with sci-kit learn
print('\nUsing sci-kit learn...\n')
reg2 = LinearRegression()
reg2 = reg2.fit(X_norm, y)
print('Intercept: ', reg2.intercept_.flatten())
print('Coefficients: ', reg2.coef_.flatten())

test = [1650, 3]
price = reg2.predict((np.array([test])-mu)/sigma)
test.append(float(price))
print('Predicted price of a %d sq-ft, %d br house (using gradient descent): \
$%0.2f' % tuple(test))

print('Coefficient of determination R^2 is %0.2f' % reg2.score(X_norm, y))
print('Mean squared error is %0.2f\n'
      % mean_squared_error(reg2.predict(X_norm), y))
