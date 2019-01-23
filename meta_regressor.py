import os
import copy
import numpy as np 

from meta_data import DataSet, mate_data, model_select, cal_mate_data

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib

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
testdatasetnames = ['australian']

for testdataset in testdatasetnames:
    print('****************currently testdataset is : ', testdataset)
    metadata = None
    testmetadata = None
    doc_root = './metadata/'
    for root, dirs, files in os.walk(doc_root):
        for file in files:
            if file == testdataset + '_metadata.npy':
                testmetadata = np.load(doc_root + file)
                continue
            # print(file)
            if metadata is None:
                metadata = np.load(doc_root + file)
            else:
                metadata = np.vstack((metadata, np.load(doc_root + file)))
    # metadata = metadata[0:30000]
    # print('The shape of metadata is ', np.shape(metadata))

    # compare the performace of different regressors

    # LinearRegression
    lr = LinearRegression(n_jobs=5)
    lr.fit(metadata[:, 0:396], metadata[:, 396])
    lr_pred = lr.predict(testmetadata[:, 0:396])
    lr_mse = mean_squared_error(testmetadata[:, 396], lr_pred)
    print('In the ' + testdataset + ' LinearRegression mean_squared_error is : ', lr_mse)
    lr_mae = mean_absolute_error(testmetadata[:, 396], lr_pred)
    print('In the ' + testdataset + 'LinearRegression mean_absolute_error is : ', lr_mae)
    lr_r2 = r2_score(testmetadata[:, 396], lr_pred)
    print('In the ' + testdataset + 'LinearRegression r2_score is : ', lr_r2)
    if lr_performance is None:
        lr_performance = np.array([testdataset, lr_mse, lr_mae, lr_r2])
    else:
        lr_performance = np.vstack((lr_performance, [testdataset, lr_mse, lr_mae, lr_r2]))
    joblib.dump(lr, testdataset + "meta_lr.joblib")

    # SGDRegressor
    sgdr = SGDRegressor()
    sgdr.fit(metadata[:, 0:396], metadata[:, 396])
    sgdr_pred = sgdr.predict(testmetadata[:, 0:396])
    sgdr_mse = mean_squared_error(testmetadata[:, 396], sgdr_pred)
    print('In the ' + testdataset + 'SGDRegressor mean_squared_error is : ', sgdr_mse)
    sgdr_mae = mean_absolute_error(testmetadata[:, 396], sgdr_pred)
    print('In the ' + testdataset + 'SGDRegressor mean_absolute_error is : ', sgdr_mae)
    sgdr_r2 = r2_score(testmetadata[:, 396], sgdr_pred)
    print('In the ' + testdataset + 'SGDRegressor r2_score is : ', sgdr_r2)
    if sgdr_performance is None:
        sgdr_performance = np.array([testdataset, sgdr_mse, sgdr_mae, sgdr_r2])
    else:
        sgdr_performance = np.vstack((sgdr_performance, [testdataset, sgdr_mse, sgdr_mae, sgdr_r2]))
    joblib.dump(sgdr, testdataset + "meta_sgdr.joblib")

    # # SVR
    # svr = SVR()
    # svr.fit(metadata[:, 0:396], metadata[:, 396])
    # svr_pred = svr.predict(testmetadata[:, 0:396])
    # svr_mse = mean_squared_error(testmetadata[:, 396], svr_pred)
    # print('In the ' + testdataset + 'SVR mean_squared_error is : ', svr_mse)
    # svr_mae = mean_absolute_error(testmetadata[:, 396], svr_pred)
    # print('In the ' + testdataset + 'SVR mean_absolute_error is : ', svr_mae)
    # svr_r2 = r2_score(testmetadata[:, 396], svr_pred)
    # print('In the ' + testdataset + 'SVR r2_score is : ', svr_r2)
    # if svr_performance is None:
    #     svr_performance = np.array([testdataset, svr_mse, svr_mae, svr_r2])
    # else:
    #     svr_performance = np.vstack((svr_performance, [testdataset, svr_mse, svr_mae, svr_r2]))
    # joblib.dump(svr, testdataset + "meta_svr.joblib")

    # GradientBoostingRegressor
    gbr = GradientBoostingRegressor()
    gbr.fit(metadata[:, 0:396], metadata[:, 396])
    gbr_pred = gbr.predict(testmetadata[:, 0:396])
    gbr_mse = mean_squared_error(testmetadata[:, 396], gbr_pred)
    print('In the ' + testdataset + 'GradientBoostingRegressor mean_squared_error is : ', gbr_mse)
    gbr_mae = mean_absolute_error(testmetadata[:, 396], gbr_pred)
    print('In the ' + testdataset + 'GradientBoostingRegressor mean_absolute_error is : ', gbr_mae)
    gbr_r2 = r2_score(testmetadata[:, 396], gbr_pred)
    print('In the ' + testdataset + 'GradientBoostingRegressor r2_score is : ', gbr_r2)
    if gbr_performance is None:
        gbr_performance = np.array([testdataset, gbr_mse, gbr_mae, gbr_r2])
    else:
        gbr_performance = np.vstack((gbr_performance, [testdataset, gbr_mse, gbr_mae, gbr_r2]))
    joblib.dump(gbr, testdataset + "meta_gbr.joblib")
