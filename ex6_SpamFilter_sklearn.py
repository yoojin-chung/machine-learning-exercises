# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week7 exercise.

Spam filter using SVM
Created on Wed Nov 11 17:59:55 2020
@author: yooji
"""

import os
import re
from nltk.stem import PorterStemmer
from sklearn import svm
from scipy.io import loadmat
import pandas as pd
import numpy as np


def getVocabList():
    """Read the fixed vocabulary list."""
    vocabList = pd.read_csv(os.path.join(folder, 'vocab.txt'),
                            delimiter='\t',
                            names=['index', 'vocab'],
                            index_col='index')
    return vocabList


def processEmail(email_contents):
    """Preprocess the body of an email and return a list of word_indices."""
    # Lower case
    email_contents = email_contents.lower()
    # Strip all HTML
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    # Handle Numbers
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    # Handle URLS
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    # Handle Email Addresses
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    # Remove any non alphanumeric characters
    email_contents = re.sub('[^a-zA-Z]', ' ', email_contents)
    # Tokenize ane remove single characters
    ps = PorterStemmer()
    email_contents = [ps.stem(token) for token
                      in email_contents.split(" ") if len(token) > 1]

    vocabList = getVocabList()
    word_indices = []
    for word in email_contents:
        ind = vocabList[vocabList.vocab == word].index
        if ind.any():
            word_indices.append(ind[0])
            print(word, '\t', ind[0])

    return email_contents, word_indices


def emailFeatures(word_indices, n=1899):
    """Produce a feature vector from the word indices."""
    x = np.zeros(n)
    x[word_indices] = 1
    return x.reshape(1, -1)


# %% Part 1: Email Preprocessing & Part 2: Feature Extraction
folder = '..\\machine-learning-ex6\\ex6'
FN = 'emailSample1.txt'

fp = open(os.path.join(folder, FN))
file_contents = fp.read()
fp.close()

email_contents, word_indices = processEmail(file_contents)
features = emailFeatures(word_indices)

print('\nLength of feature vector: %d' % features.size)
print('Number of non-zero entries: %d\n' % np.sum(features > 0))
wait = input("Program paused. Press enter to continue.\n")

# %% Part 3: Train Linear SVM for Spam Classification
FN = 'spamTrain.mat'
data = loadmat(os.path.join(folder, FN))
X = data['X']
y = data['y']
y = y.reshape(-1, )
clf = svm.SVC(kernel='linear', C=0.1)
clf.fit(X, y)
wait = input("Program paused. Press enter to continue.\n")

# %% Part 4: Test Spam Classification
FN = 'spamTest.mat'
data = loadmat(os.path.join(folder, FN))
Xtest = data['Xtest']
ytest = data['ytest']
ytest = ytest.reshape(-1)

p = clf.predict(Xtest)
print('Test Accuracy: %f' % np.mean(p == ytest))
wait = input("Program paused. Press enter to continue.\n")

# %% Part 5: Top Predictors of Spam
vocab = getVocabList()
we = clf.coef_.reshape(-1)
idx = np.argsort(we)[::-1]

print('Top predictors of spam:')
print(vocab.iloc[idx[:15]])
wait = input("Program paused. Press enter to continue.\n")

# %% Part 6: Try Your Own Emails
FN = 'emailSample3.txt'

with open(os.path.join(folder, FN)) as fp:
    file_contents = fp.read()

_, word_indices = processEmail(file_contents)
features = emailFeatures(word_indices)

p = clf.predict(features)
print('\nProcessed %s\n\nSpam Classification: %d\n' % (FN, p))
