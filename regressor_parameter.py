import os
import copy
import numpy as np 
import datetime
from meta_data import DataSet, mate_data, model_select, cal_mate_data

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

lr_performance = None
sgdr_performance = None
svr_performance = None
gbr_performance = None

testdatasetnames = np.load('datasetname.npy')
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
traindatasetnames = ['australian_metadata.npy']

trainmetadata = None
doc_root = './metadata/'
for root, dirs, files in os.walk(doc_root):
    for file in files:
        if file in traindatasetnames:
            if trainmetadata is None:
                trainmetadata = np.load(doc_root + file)
            else:
                trainmetadata = np.vstack((trainmetadata, np.load(doc_root + file)))     

start = datetime.datetime.now()

# SGDRegressor
sgdr = SGDRegressor()

param_grid = {'learning_rate':['', '']}

grid = GridSearchCV(estimator = knn, param_grid = param_grid, cv=10, scoring='accuracy')
grid.fit(trainmetadata[:, 0:396], trainmetadata[:, 396])

print('grid.cv_results：',grid.cv_results_)
print('grid.best_score:',grid.best_score_)  
print('grid.best_param：',grid.best_params_)  
np.save('best_params.npy', grid.best_params_)

end = datetime.datetime.now()

print('The time tuning parameter used is : ',(end-start).seconds)