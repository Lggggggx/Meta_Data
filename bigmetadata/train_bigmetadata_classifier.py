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


# # Here are some classifiers that support the `partial_fit` method
partial_fit_classifiers = {
    'SGD': SGDClassifier(max_iter=5),
    'Perceptron': Perceptron(tol=1e-3),
    # 'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(tol=1e-3),
}

all_classes = np.array([-1, 1])

def iter_minibatches():
    # for num in range(30):
    #     X = np.array([])
    #     y = np.array([])
    #     metadata = np.load('./wdbc_lr/'+str(num)+'_wdbc_big_metadata.npy')
    #     X = metadata[:, 0:396]
    #     y = metadata[:, 396]
    #     new_X = X[np.where(y>=0.01)[0]]
    #     new_X = np.vstack((new_X, X[np.where(y<= -0.01)[0]]))
    #     new_y = y[np.where(y>=0.01)[0]]
    #     new_y = np.append(new_y, y[np.where(y<=-0.01)[0]])
    #     new_y[np.where(new_y>0)] = 1
    #     new_y[np.where(new_y<=0)] = -1

    #     yield new_X, new_y

    metadata_dir = 'D:/generate_metadata/bigmetadata/wdbc/query_time/query_time_5interval/query_time110/'
    for root, dirs, files in os.walk(metadata_dir):
        for file in files:
            print(metadata_dir+file)
            metadata = np.load(metadata_dir+file)
            X = metadata[:, 0:396]
            y = metadata[:, 396]
            new_X = X[np.where(y>=0.01)[0]]
            new_X = np.vstack((new_X, X[np.where(y<= -0.01)[0]]))
            new_y = y[np.where(y>=0.01)[0]]
            new_y = np.append(new_y, y[np.where(y<=-0.01)[0]])
            new_y[np.where(new_y>0)] = 1
            new_y[np.where(new_y<=0)] = -1

            yield new_X, new_y
    


# sgd_clf = SGDClassifier()

minibatch_train_iterators = iter_minibatches()

test_metadata = np.load('../newmetadata/wdbc_metadata.npy')
X_test = test_metadata[:, 0:396]
y_test = test_metadata[:, 396]
y_test[np.where(y_test>0)] = 1
y_test[np.where(y_test<=0)] = -1

total_vect_time = 0.0

for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
    print('~~~~~~~~~~~the number of this time train samples is {}'.format(X_train.shape[0]))
    
    for cls_name, cls in partial_fit_classifiers.items():
        tick = time.time()
        # update estimator with examples in the current mini-batch

        cls.partial_fit(X_train, y_train, classes=all_classes)

        # accumulate test accuracy stats
        cls_stats[cls_name]['total_fit_time'] += time.time() - tick
        cls_stats[cls_name]['n_train'] += X_train.shape[0]
        cls_stats[cls_name]['n_train_pos'] += sum(y_train)
        tick = time.time()
        cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
        cls_stats[cls_name]['prediction_time'] = time.time() - tick
        acc_history = (cls_stats[cls_name]['accuracy'],
                       cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['accuracy_history'].append(acc_history)
        run_history = (cls_stats[cls_name]['accuracy'],
                       total_vect_time + cls_stats[cls_name]['total_fit_time'])
        cls_stats[cls_name]['runtime_history'].append(run_history)

        if i % 3 == 0:
            print(progress(cls_name, cls_stats[cls_name]))
    if i % 3 == 0:
        print('\n')

    # sgd_clf.partial_fit(X_train, y_train, classes=np.array([-1, 1]))
    # print("{} time".format(i))  
    # print("{} score".format(sgd_clf.score(X_test, y_test)))  

# joblib.dump(partial_fit_classifiers, './partial_fit_classifiers.joblib')

