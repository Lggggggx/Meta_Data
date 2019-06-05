from __future__ import print_function
from glob import glob
import itertools
import os
import re
import tarfile
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor

from sklearn.externals import joblib

# temp_metadata = np.load('./wdbc_lr/0_wdbc_big_metadata.npy')
# print(np.shape(temp_metadata))


# def progress(cls_name, stats):
#     """Report progress information, return a string."""
#     duration = time.time() - stats['t0']
#     s = "%20s regressor : \t" % cls_name
#     s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
#     s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
#     s += "r2: %(r2).3f " % stats
#     s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
#     return s

# # # Here are some classifiers that support the `partial_fit` method
# partial_fit_regressor = {
#     'SGD': SGDRegressor(max_iter=5),
#     # 'Perceptron': Perceptron(tol=1e-3),
#     # 'NB Multinomial': MultinomialNB(alpha=0.01),
#     'Passive-Aggressive': PassiveAggressiveRegressor(tol=1e-3),
# }

# cls_stats = {}
# for cls_name in partial_fit_regressor:
#     stats = {'n_train': 0, 'n_train_pos': 0,
#              'r2': 0.0, 'r2_history': [(0, 0)], 't0': time.time(),
#              'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
#     cls_stats[cls_name] = stats

# test data statistics
# test_stats = {'n_test': 0, 'n_test_pos': 0}

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

dataset_name = 'australian'
# sgd_regressor = SGDRegressor()
sgd_regressor = SGDRegressor(learning_rate='invscaling', alpha=0.0001, eta0=0.005, loss='huber')

# test_metadata = np.load('')

# X_test = test_metadata[:, 0:396]
# y_test = test_metadata[:, 396]

total_vect_time = 0.0

def iter_minibatches():

    # for num in range(2, 62, 2):
    #     # X = np.array([])
    #     # y = np.array([])
    #     

    #     

    #     metadata = np.load(str(num)+'australian_big_metadata.npy')
    #     X = metadata[:, 0:396]						
    #     y = metadata[:, 396]
    #     yield X, y
    #     # X = np.array([])
    #     # y = np.array([])
    # metadata_dir = 'D:/generate_metadata/bigmetadata/wdbc/query_time/query_time_5interval/query_time110/'
    # metadata_dir = 'C:/Users/31236/Desktop/meta_data/bigmetadata/australian/'
    # metadata_dir = 'F:/australian/'

    # metadata_dir = 'E:/australian/model1/split_count50/'
    metadata_dir = 'E:/australian/model1//query_time300/'

    for root, dirs, files in os.walk(metadata_dir):
        for file in files:
            print(metadata_dir+file)
            metadata = np.load(metadata_dir+file)
            X = metadata[:, 0:396]					
            y = metadata[:, 396]
            yield X, y


minibatch_train_iterators = iter_minibatches()

# for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
#     print('the number of this time train samples is {}'.format(X_train.shape[0]))
    
#     for cls_name, cls in partial_fit_regressor.items():
#         tick = time.time()
#         # update estimator with examples in the current mini-batch
#         cls.partial_fit(X_train, y_train)

#         # accumulate test r2 stats
#         cls_stats[cls_name]['total_fit_time'] += time.time() - tick
#         cls_stats[cls_name]['n_train'] += X_train.shape[0]
#         cls_stats[cls_name]['n_train_pos'] += sum(y_train)
#         tick = time.time()
#         cls_stats[cls_name]['r2'] = cls.score(X_test, y_test)
#         cls_stats[cls_name]['prediction_time'] = time.time() - tick
#         acc_history = (cls_stats[cls_name]['r2'],
#                        cls_stats[cls_name]['n_train'])
#         cls_stats[cls_name]['r2_history'].append(acc_history)
#         run_history = (cls_stats[cls_name]['r2'],
#                        total_vect_time + cls_stats[cls_name]['total_fit_time'])
#         cls_stats[cls_name]['runtime_history'].append(run_history)

#         if i % 3 == 0:
#             print(progress(cls_name, cls_stats[cls_name]))
#     if i % 3 == 0:
#         print('\n')

#     # sgd_regressor.partial_fit(X_train, y_train, classes=np.array([-1, 1]))
#     # print("{} time".format(i))  
#     # print("{} score".format(sgd_regressor.score(X_test, y_test)))  

# joblib.dump(partial_fit_regressor, './partial_fit_regressor.joblib')

for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
    tick = time.time()
    print('###########{}`th round training'.format(i))
    print('the number of this time train samples is {}'.format(X_train.shape[0]))
    sgd_regressor.partial_fit(X_train, y_train)
    training_time = time.time() - tick
    print('the time of current round traing is {}'.format(training_time))
    total_vect_time +=training_time
    print('the score(r2) on the test dataset is {}'.format(sgd_regressor.score(X_train[0:1000,:], y_train[0:1000])))

print('*******training is done')
# joblib.dump(sgd_regressor, './australian_s-c-50-inter2-regressor.joblib')
joblib.dump(sgd_regressor, './australian_q-t-300-inter2-regressor.joblib')
# joblib.dump(sgd_regressor, './australian_origin_regressor.joblib')
