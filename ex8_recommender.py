# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week9 exercise.

Movie recommender systtem using collaborative filtering
Created on Mon Oct  5 20:56:52 2020
@author: yooji
"""

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import os
import pickle
from scipy.optimize import fmin_cg, fmin_bfgs, minimize


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lam):
    """Compute cost and gradient for collaborative filtering."""
    X = params[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

    J = np.sum((np.dot(X, Theta.T)*R - Y)**2)/2 +\
        lam/2*(np.sum(Theta**2) + np.sum(X**2))

    X_grad = (X.dot(Theta.T)*R-Y).dot(Theta) + lam*X
    Theta_grad = (X.dot(Theta.T)*R-Y).T.dot(X) + lam*Theta
    new_params = np.hstack([X_grad.flatten(), Theta_grad.flatten()])
    return J, new_params


# # A separate gradient function for fmin_cg. Not necessary for minimize.
# def cofiGradFunc(params, Y, R, num_users, num_movies, num_features, lam):
#     X = params[:num_movies*num_features].reshape(num_movies, num_features)
#     Theta = params[num_movies*num_features:].reshape(num_users, num_features)

#     X_grad = (X.dot(Theta.T)*R-Y).dot(Theta) + lam*X
#     Theta_grad = (X.dot(Theta.T)*R-Y).T.dot(X) + lam*Theta
#     new_params = np.hstack([X_grad.flatten(), Theta_grad.flatten()])
#     return new_params


def compNumericalGrad(costFunc, theta):
    """Numerically compute gradient for debugging."""
    numgrad = np.zeros_like(theta)
    perturb = np.zeros(theta.size)
    eps = 1e-4
    for p in range(theta.size):
        perturb[p] = eps
        loss1, _ = costFunc(theta - perturb.reshape(theta.shape))
        loss2, _ = costFunc(theta + perturb.reshape(theta.shape))
        # Compute Numerical Gradient
        numgrad[p] = (loss2-loss1)/(2*eps)
        perturb[p] = 0
    return numgrad


def newUserRating(DF, Ymean):
    """Get ratings from a new user."""
    cnt = 0
    my_ratings = np.zeros(len(DF))
    # ind_rate = np.random.permutation(len(DF))
    ind_rate = np.argsort(Ymean)[::-1]
    
    for ind in ind_rate:
        print(DF.Title[ind], '\nYear:', DF.Year[ind])
        rate = input('Your rating? [1-5]:\n')
        if rate:
            cnt += 1
            print('\n')
            if cnt > 10:
                break
        else:
            rate = 0
        my_ratings[ind] = rate

    my_ratings = my_ratings[:, None]
    resp = input('Save ratings? [Y/N]:\n')
    if resp == 'Y':
        filename = input('Please enter a new file name for ratings: ')
        fp = open(filename, 'wb')
        pickle.dump(my_ratings, fp)
        fp.close()
    else:
        pass
    return my_ratings


def normRatings(Y, R):
    """Normalize ratings."""
    Ymean = np.sum(Y, axis=1)/np.sum(R, axis=1)
    Ynorm = Y-Ymean[:, None]*R
    return Ynorm, Ymean


def showRatings(my_ratings, DF):
    """Show ratings."""
    titles = np.array(DF[my_ratings > 0].Title)
    ratings = my_ratings[my_ratings > 0]
    movies = np.hstack([titles[:, None], ratings[:, None]])
    print(movies)


# %% Load and take a look at the dataset
folder = 'P:\\Joint\\learnings\\Machine Learning\\machine-learning-ex8\\ex8'
FN1 = os.path.join(folder, 'ex8_movies.mat')
data1 = loadmat(FN1)
Y = data1['Y']
R = data1['R']

# Show average rating for Movie #1.
print('Average rating for movie 1 (Toy Story): %0.2f / 5\n' %
      np.mean(Y[1, R[1, :]]))

# Vvisualize ratings matrix
plt.figure()
ax = sns.heatmap(Y)
ax.set_ylabel('Movies')
ax.set_xlabel('Users')

# %% Test cost function
FN2 = os.path.join(folder, 'ex8_movieParams.mat')
data2 = loadmat(FN2)
keys = [x for x in data2.keys() if x[0] != '_']
print(keys)
X = data2['X']
Theta = data2['Theta']
num_users = data2['num_users'][0, 0]
num_movies = data2['num_movies'][0, 0]
num_features = data2['num_features'][0, 0]

#  Reduce the data set size for testing
num_users1 = 4
num_movies1 = 5
num_features1 = 3
X1 = X[:num_movies1, :num_features1]
Theta1 = Theta[:num_users1, :num_features1]
Y1 = Y[:num_movies1, :num_users1]
R1 = R[:num_movies1, :num_users1]

#  Evaluate cost function
params = np.hstack([X1.flatten(), Theta1.flatten()])
J, _ = cofiCostFunc(params, Y1, R1, num_users1, num_movies1, num_features1, 0)
print('Cost at loaded parameters: %f\n(this value should be about 22.22)\n'
      % J)

J, _ = cofiCostFunc(params, Y1, R1, num_users1, num_movies1, num_features1, 1.5)
print('Cost at loaded parameters (lambda = 1.5):\
      %f\n(this value should be about 31.34)\n' % J)

# Check gradient by comparing to numerical estimate
costFunc = partial(cofiCostFunc,
                   Y=Y1, R=R1, num_users=num_users1, num_movies=num_movies1,
                   num_features=num_features1, lam=1.5)
_, grad = cofiCostFunc(params,
                    Y1, R1, num_users1, num_movies1, num_features1, 1.5)
# grad = cofiGradFunc(params,
#                     Y1, R1, num_users1, num_movies1, num_features1, 1.5)
numgrad = compNumericalGrad(costFunc, params)
diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
print('If your cost function implementation is correct, then \n'
      'the relative difference will be small (less than 1e-9). \n'
      '\nRelative Difference: %g\n' % diff)

# %% Load movie list and save as a DataFrame
FN3 = os.path.join(folder, 'movie_ids.txt')
ind = []
title = []
year = []
with open(FN3) as fp:
    line = fp.readline()
    while line:
        sp = line.find(' ')
        ind.append(line[:sp])
        title.append(line[sp+1:-8].rstrip())
        year.append(line[-6:-2])
        line = fp.readline()

movie_data = list(zip(ind, title, year))
DF = pd.DataFrame(movie_data, columns=['ID', 'Title', 'Year'])

# %% Load saved ratings
fp = open('archive\\my_ratings.pkl', 'rb')
my_ratings = pickle.load(fp)
fp.close()
my_ratings = my_ratings.reshape(-1, 1)
showRatings(my_ratings, DF)

# %% Ratings from the tutorial
my_ratings = np.zeros([len(Y), 1])
s = 1
my_ratings[1-s] = 4
my_ratings[98-s] = 2
my_ratings[7-s] = 3
my_ratings[12-s] = 5
my_ratings[54-s] = 4
my_ratings[64-s] = 5
my_ratings[66-s] = 3
my_ratings[69-s] = 5
my_ratings[183-s] = 4
my_ratings[226-s] = 5
my_ratings[355-s] = 5

showRatings(my_ratings, DF)

# %% Set up to train the collaborative filtering model
Y = data1['Y']
R = data1['R']

Y = np.hstack([my_ratings, Y])
R = np.hstack([my_ratings != 0, R])

ind = np.sum(R, axis=1) < 10
R = R[ind, :]
Y = Y[ind, :]

Ynorm, Ymean = normRatings(Y, R)
num_movies, num_users = Y.shape
num_features = 20

X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

init_params = np.hstack([X.flatten(), Theta.flatten()])
lam = 10

# %% Train my recommender
print('Training my recommender...')
# theta = fmin_cg(cofiCostFunc,
#                 x0=init_params,
#                 fprime=cofiGradFunc,
#                 args=(Ynorm, R, num_users, num_movies, num_features, lam),
#                 maxiter=200,
#                 full_output=True)

theta = minimize(cofiCostFunc,
                 x0=init_params,
                 args=(Ynorm, R, num_users, num_movies, num_features, lam),
                 method='TNC',
                 jac=True)

# %% Show recommendations
X = theta.x[:num_movies*num_features].reshape(num_movies, num_features)
Theta = theta.x[num_movies*num_features:].reshape(num_users, num_features)
p = X.dot(Theta.T)
my_predictions = p[:, 0] + Ymean
ix = np.argsort(my_predictions)[::-1]
top_movies = DF.iloc[ix[:20]].copy()
top_movies['Rating'] = my_predictions[ix[:20]]
print(top_movies)
