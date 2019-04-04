from __future__ import print_function
from glob import glob
import itertools
import os.path
import re
import tarfile
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB

from sklearn.externals import joblib

metadata = np.load('./wdbc_lr/0_wdbc_big_metadata.npy')
print(np.shape(metadata))

# test_metadata = np.load('../newmetadata/wdbc_metadata.npy')

# X_test = test_metadata[:, 0:396]
# y_test = test_metadata[:, 396]
# y_test[np.where(y_test>0)] = 1
# y_test[np.where(y_test<=0)] = -1

# partial_fit_classifiers = joblib.load('./partial_fit_classifiers.joblib')
# # print(partial_fit_classifiers)

# temp_X = X_test[0:5,:]
# temp_y = y_test[0:5]
# print(temp_y)
# pred = partial_fit_classifiers['SGD'].predict(temp_X)
# print(pred)
# dec = partial_fit_classifiers['SGD'].decision_function(temp_X)
# print(dec)
# print('the test accuracy is : ', partial_fit_classifiers['SGD'].score(X_test[0:5, :], y_test[0:5]))
# # for cls_name, cls in partial_fit_classifiers.items():
# #     print('current is ', cls_name)
# #     print('the test accuracy is : ', cls.score(X_test, y_test))