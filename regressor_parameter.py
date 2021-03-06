import os
import numpy as np 
import datetime

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

lr_performance = None
sgdr_performance = None
svr_performance = None
gbr_performance = None

# testdatasetnames = np.load('datasetname.npy')
# testdatasetnames = np.load('datasetname.npy')
# testdatasetnames = ['australian' 'blood' 'breast-cancer-wisc-diag' 'breast-cancer-wisc'
#  'chess-krvkp' 'clean1' 'congressional-voting' 'credit-approval'
#  'cylinder-bands' 'diabetes' 'echocardiogram' 'ethn' 'german'
#  'heart-hungarian' 'heart-statlog' 'heart' 'hill-valley' 'horse-colic'
#  'house-votes' 'house' 'ilpd-indian-liver' 'ionosphere' 'isolet' 'krvskp'
#  'liverDisorders' 'mammographic' 'monks-1' 'monks-2' 'monks-3' 'mushroom'
#  'oocytes_merluccius_nucleus_4d' 'oocytes_trisopterus_nucleus_2f'
#  'optdigits' 'pima' 'ringnorm' 'sat' 'spambase' 'spect' 'spectf'
#  'statlog-australian-credit' 'statlog-german-credit' 'statlog-heart'
#  'texture' 'tic-tac-toe' 'titanic' 'twonorm' 'vehicle'
#  'vertebral-column-2clases' 'wdbc']
traindatasetnames = ['australian_metadata.npy', 'echocardiogram_metadata.npy', 'heart_metadata.npy']

trainmetadata = None
# doc_root = './metadata/'
# for root, dirs, files in os.walk(doc_root):
#     for file in files:
#         if file in traindatasetnames:
#             if trainmetadata is None:
#                 trainmetadata = np.load(doc_root + file)
#             else:
#                 trainmetadata = np.vstack((trainmetadata, np.load(doc_root + file)))     

metadata1 = np.load('E:/australian/model1/query_time100/2australian30_big_metadata100.npy')
metadata2 = np.load('E:/australian/model1/query_time100/20australian30_big_metadata100.npy')
metadata3 = np.load('E:/australian/model1/query_time100/60australian30_big_metadata100.npy')
metadata4 = np.load('E:/australian/model1/query_time100/90australian30_big_metadata100.npy')


trainmetadata = np.vstack((metadata1, metadata2, metadata3, metadata4))

start = datetime.datetime.now()

# SGDRegressor
# sgdr = SGDRegressor(max_iter=10)
lasso = Lasso()

# param_grid = {'learning_rate':['constant', 'optimal', 'invscaling'], 'alpha':[0.00005, 0.0001, 0.00015], 'eta0':[0.005, 0.01, 0.015]}

param_grid = {'alpha':[0.000001, 0.000005, 0.00005, 0.0001, 0.00015, 0.005, 0.01, 0.015]}
grid = GridSearchCV(estimator = lasso, param_grid = param_grid, cv=5, scoring='r2')
grid.fit(trainmetadata[:, 0:396], trainmetadata[:, 396])

print('grid.cv_results：',grid.cv_results_)
print('grid.best_score:',grid.best_score_)  
print('grid.best_param：',grid.best_params_)  
np.save('best_params.npy', grid.best_params_)

end = datetime.datetime.now()

print('The time tuning parameter used is : ',(end-start).seconds)