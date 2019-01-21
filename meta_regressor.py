import os
import copy
import numpy as np 

from meta_data import DataSet, mate_data, model_select, cal_mate_data

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib


testdataset = 'australian_metadata.npy'
metadata = None
doc_root = './metadata/'
for root, dirs, files in os.walk(doc_root):
    for file in files:
        if file == testdataset:
            continue
        print(file)
        if metadata is None:
            metadata = np.load(doc_root+file)
            print(file, np.shape(metadata))
        else:
            metadata = np.vstack((metadata, np.load(doc_root+file)))

# metadata = metadata[0:30000]
print(np.shape(metadata))

# compare the performace of different regressors

# LinearRegression
lr = LinearRegression(n_jobs=5)
lr.fit(metadata[:, 0:396], metadata[:, 396])
lr_pred = lr.predict(metadata[:, 0:396])
lr_mse = mean_squared_error(metadata[:, 396], lr_pred)
print('LinearRegression mean_squared_error is : ', lr_mse)
lr_mae = mean_absolute_error(metadata[:, 396], lr_pred)
print('LinearRegression mean_absolute_error is : ', lr_mae)
lr_r2 = r2_score(metadata[:, 396], lr_pred)
print('LinearRegression r2_score is : ', lr_r2)
joblib.dump(lr, "meta_lr.joblib")

# SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(metadata[:, 0:396], metadata[:, 396])
sgdr_pred = sgdr.predict(metadata[:, 0:396])
sgdr_mse = mean_squared_error(metadata[:, 396], sgdr_pred)
print('SGDRegressor mean_squared_error is : ', sgdr_mse)
sgdr_mae = mean_absolute_error(metadata[:, 396], sgdr_pred)
print('SGDRegressor mean_absolute_error is : ', sgdr_mae)
sgdr_r2 = r2_score(metadata[:, 396], sgdr_pred)
print('SGDRegressor r2_score is : ', sgdr_r2)
joblib.dump(sgdr, "meta_sgdr.joblib")

# SVR
svr = SVR()
svr.fit(metadata[:, 0:396], metadata[:, 396])
svr_pred = svr.predict(metadata[:, 0:396])
svr_mse = mean_squared_error(metadata[:, 396], svr_pred)
print('SVR mean_squared_error is : ', svr_mse)
svr_mae = mean_absolute_error(metadata[:, 396], svr_pred)
print('SVR mean_absolute_error is : ', svr_mae)
svr_r2 = r2_score(metadata[:, 396], svr_pred)
print('SVR r2_score is : ', svr_r2)
joblib.dump(svr, "meta_svr.joblib")

# GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(metadata[:, 0:396], metadata[:, 396])
gbr_pred = gbr.predict(metadata[:, 0:396])
gbr_mse = mean_squared_error(metadata[:, 396], gbr_pred)
print('GradientBoostingRegressor mean_squared_error is : ', gbr_mse)
gbr_mae = mean_absolute_error(metadata[:, 396], gbr_pred)
print('GradientBoostingRegressor mean_absolute_error is : ', gbr_mae)
gbr_r2 = r2_score(metadata[:, 396], gbr_pred)
print('GradientBoostingRegressor r2_score is : ', gbr_r2)
joblib.dump(gbr, "meta_gbr.joblib")