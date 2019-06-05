from __future__ import print_function
from glob import glob
import itertools
import os
import re
import tarfile
import time
import sys

import numpy as np

from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


all_classes = np.array([-1, 1])

dataset_name = 'australian'

# classifier
# classifier = GaussianNB(var_smoothing=)
classifier = GaussianNB()
# classifier = MultinomialNB(alpha=)
# classifier = BernoulliNB(alpha=, binarize=)

total_vect_time = 0.0

def iter_minibatches():
    metadata_dir = 'E:/australian/model1//query_time300/'

    for root, dirs, files in os.walk(metadata_dir):
        for file in files:
            print(metadata_dir+file)
            metadata = np.load(metadata_dir+file)
            X = metadata[:, 0:396]					
            y = metadata[:, 396]
            y[np.where(y>0)[0]] = 1
            y[np.where(y<=0)[0]] = -1
            yield X, y


minibatch_train_iterators = iter_minibatches()


for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
    tick = time.time()
    print('###########{}`th round training'.format(i))
    print('the number of this time train samples is {}'.format(X_train.shape[0]))
    classifier.partial_fit(X_train, y_train, all_classes)
    training_time = time.time() - tick
    print('the time of current round traing is {}'.format(training_time))
    total_vect_time +=training_time
    print('the score(r2) on the test dataset is {}'.format(classifier.score(X_train[0:1000,:], y_train[0:1000])))

print('*******training is done')

# joblib.dump(classifier, './australian_GaussianNB.joblib')
