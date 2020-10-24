# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week5 exercise.

Build a NN model to classify MNIST.
Created on Sun Sep 27 13:07:03 2020
@author: yooji
"""

import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_cg
from sklearn.neural_network import MLPClassifier
from myLogReg import displayData, sigmoid, sigmoidGrad


def nnCostFunc(nn_params, input_size, hidden_size, num_labels, X, y, lam):
    """Implement cost function for a 2-layer NN performs classification."""
    Theta1 = nn_params[:hidden_size * (input_size + 1)].\
        reshape(hidden_size, input_size + 1)
    Theta2 = nn_params[hidden_size * (input_size + 1):].\
        reshape(num_labels, hidden_size + 1)
    m = X.shape[0]

    yv = (np.arange(num_labels)+1) == y
    _, h = predict(Theta1, Theta2, X)

    J = (0 - yv) * np.log(h) - (1 - yv) * np.log(1 - h)
    J = np.sum(J)
    J = J/m + lam/(2*m)*(np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2))
    return J


def nnGradFunc(nn_params, input_size, hidden_size, num_labels, X, y, lam):
    """Implement gradient function for a 2-layer NN performs classification."""
    Theta1 = nn_params[:hidden_size * (input_size + 1)].\
        reshape(hidden_size, input_size + 1)
    Theta2 = nn_params[hidden_size * (input_size + 1):].\
        reshape(num_labels, hidden_size + 1)
    m = X.shape[0]

    yv = (np.arange(num_labels)+1) == y
    a_1 = np.hstack([np.ones([m, 1]), X])
    z_2 = a_1.dot(Theta1.T)
    a_2 = np.hstack([np.ones([m, 1]), sigmoid(z_2)])
    z_3 = a_2.dot(Theta2.T)
    a_3 = sigmoid(z_3)

    delta_3 = a_3-yv
    delta_3 = delta_3.T
    delta_2 = (np.dot(delta_3.T, Theta2)).T *\
        sigmoidGrad(np.vstack([np.ones([1, m]), z_2.T]))
    delta_2 = delta_2[1:]
    D2 = delta_3.dot(a_2)
    D1 = delta_2.dot(a_1)

    Theta1_grad = D1/m +\
        np.hstack([np.zeros([hidden_size, 1]), lam/m*(Theta1[:, 1:])])
    Theta2_grad = D2/m +\
        np.hstack([np.zeros([num_labels, 1]), lam/m*(Theta2[:, 1:])])

    grad = np.hstack([Theta1_grad.ravel(), Theta2_grad.ravel()])
    return grad


def randInitWe(x, y):
    """Randomly initialize weights."""
    eps = 1.e-4
    W = 2*eps*np.random.randn(x, y) - 2*eps
    return W


def predict(Theta1, Theta2, X):
    """Predict NN outcome based on the learned theta."""
    m = X.shape[0]

    a_1 = np.hstack([np.ones([m, 1]), X])
    z_2 = a_1.dot(Theta1.T)
    a_2 = np.hstack([np.ones([m, 1]), sigmoid(z_2)])
    z_3 = a_2.dot(Theta2.T)
    a_3 = sigmoid(z_3)

    p = np.argmax(a_3, axis=1)
    return p, a_3


# %%
# Setup the parameters you will use for this exercise
input_size = 400   # 20x20 Input Images of Digits
hidden_size = 25   # 25 hidden units
num_labels = 10    # 10 labels, from 1 to 10

FN1 = '..\\machine-learning-ex4\\ex4\\ex4data1.mat'
data = loadmat(FN1)
X = data['X']
y = data['y']
m = X.shape[0]
train_num = int(0.8*m)

# Show some data first
m = X.shape[0]
ind = np.random.permutation(m)
X_sel = X[ind[:100]]
X_train = X[ind[:train_num]]
X_test = X[ind[train_num:]]
y_train = y[ind[:train_num]]
y_test = y[ind[train_num:]]
displayData(X_sel, int(np.sqrt(input_size)))

FN2 = '..\\machine-learning-ex4\\ex4\\ex4weights.mat'
W = loadmat(FN2)
Theta1 = W['Theta1']
Theta2 = W['Theta2']

Theta1 = Theta1.ravel()
Theta2 = Theta2.ravel()
nn_theta = np.concatenate((Theta1, Theta2))

lam = 3
J = nnCostFunc(nn_theta, input_size, hidden_size, num_labels, X, y, lam)
grad = nnGradFunc(nn_theta, input_size, hidden_size, num_labels, X, y, lam)

theta_init1 = randInitWe(hidden_size, input_size+1)
theta_init2 = randInitWe(num_labels, hidden_size+1)
init_nntheta = np.concatenate((theta_init1.ravel(), theta_init2.ravel()))

print('Training my neural network...')
theta = fmin_cg(nnCostFunc,
                x0=init_nntheta,
                fprime=nnGradFunc,
                args=(input_size, hidden_size, num_labels, X_train, y_train, lam),
                maxiter=100,
                full_output=True)

Theta1 = theta[0][:hidden_size * (input_size + 1)].\
    reshape(hidden_size, input_size + 1)
Theta2 = theta[0][hidden_size * (input_size + 1):].\
    reshape(num_labels, hidden_size + 1)

displayData(Theta1[:, 1:], int(np.sqrt(input_size)))

pred, _ = predict(Theta1, Theta2, X_test)
print('Training Accuraty: %0.2f\n' % np.mean(pred+1 == y_test.flatten()))

# %% Now try running the same thing with sci-kit learn
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(25,), random_state=1)
print('Training the neural network using sci-kit learn...')
clf.fit(X_train, y_train.flatten())
pred = clf.predict(X_test)
print('Training Accuraty: %0.2f' % np.mean(pred == y_test.flatten()))
