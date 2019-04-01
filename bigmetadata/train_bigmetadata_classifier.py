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

# temp_metadata = np.load('./wdbc_lr/0_wdbc_big_metadata.npy')
# print(np.shape(temp_metadata))


def progress(cls_name, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%20s classifier : \t" % cls_name
    s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
    s += "accuracy: %(accuracy).3f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s

# # Here are some classifiers that support the `partial_fit` method
partial_fit_classifiers = {
    'SGD': SGDClassifier(max_iter=5),
    'Perceptron': Perceptron(tol=1e-3),
    # 'NB Multinomial': MultinomialNB(alpha=0.01),
    'Passive-Aggressive': PassiveAggressiveClassifier(tol=1e-3),
}

cls_stats = {}
for cls_name in partial_fit_classifiers:
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats

# test data statistics
test_stats = {'n_test': 0, 'n_test_pos': 0}

# def get_minibatch(doc_iter, size, pos_class=positive_class):
#     """Extract a minibatch of examples, return a tuple X_text, y.

#     Note: size is before excluding invalid docs with no topics assigned.

#     """
#     data = [(u'{title}\n\n{body}'.format(**doc), pos_class in doc['topics'])
#             for doc in itertools.islice(doc_iter, size)
#             if doc['topics']]
#     if not len(data):
#         return np.asarray([], dtype=int), np.asarray([], dtype=int)
#     X_text, y = zip(*data)
#     return X_text, np.asarray(y, dtype=int)


# def iter_minibatches(doc_iter, minibatch_size):
#     """Generator of minibatches."""
#     X_text, y = get_minibatch(doc_iter, minibatch_size)
#     while len(X_text):
#         yield X_text, y
#         X_text, y = get_minibatch(doc_iter, minibatch_size)
all_classes = np.array([-1, 1])

def iter_minibatches():
    for num in range(30):
        X = np.array([])
        y = np.array([])
        metadata = np.load('./wdbc_lr/'+str(num)+'_wdbc_big_metadata.npy')
        num +=1
        X = metadata[:, 0:396]
        y = metadata[:, 396]
        y[np.where(y>0)] = 1
        y[np.where(y<=0)] = -1
        yield X, y
        X = np.array([])
        y = np.array([])

# sgd_clf = SGDClassifier()

minibatch_train_iterators = iter_minibatches()

test_metadata = np.load('../newmetadata/wdbc_metadata.npy')
X_test = test_metadata[:, 0:396]
y_test = test_metadata[:, 396]
y_test[np.where(y_test>0)] = 1
y_test[np.where(y_test<=0)] = -1

total_vect_time = 0.0

for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
    
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

joblib.dump(partial_fit_classifiers, './partial_fit_classifiers.joblib')