import numpy as np 
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


australian_metadata = np.load('./metadata/australian_metadata.npy')
print(np.shape(australian_metadata))

np.random.shuffle(australian_metadata)

# metadata = australian_metadata[0:10000, :]
metadata = australian_metadata


X = metadata[:, 0:396]
y = metadata[:, 396]
# print(y[0:20])
split = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
split.get_n_splits(X=X)
print(split)

turn = 0 
for train_ind, test_ind in split.split(X, y):
    turn += 1
    print('This is the ', turn, '`th**********')

    train_X = X[train_ind]
    train_y = y[train_ind]
    test_X = X[test_ind]
    test_y = y[test_ind]

    # sgdr = SGDRegressor()
    # sgdr.fit(train_X, train_y)
    # sgdr_pred = sgdr.predict(test_X)
    # sgdr_mse = mean_squared_error(test_y, sgdr_pred)
    # print('SGDRegressor mean_squared_error is : ', sgdr_mse)
    # sgdr_mae = mean_absolute_error(test_y, sgdr_pred)
    # print('SGDRegressor mean_absolute_error is : ', sgdr_mae)
    # sgdr_r2 = r2_score(test_y, sgdr_pred)
    # print('SGDRegressor r2_score is : ', sgdr_r2)

    lr = LinearRegression()
    lr.fit(train_X, train_y)
    lr_pred = lr.predict(test_X)
    lr_mse = mean_squared_error(test_y, lr_pred)
    print('LinearRegression mse is : ', lr_mse)
    lr_mae = mean_absolute_error(test_y, lr_pred)
    print('LinearRegression mae is : ', lr_mae)
    lr_r2 = r2_score(test_y, lr_pred)
    print('LinearRegression r2 is :', lr_r2)

    # svr = SVR()
    # svr.fit(train_X, train_y)
    # svr_pred = svr.predict(test_X)
    # svr_mse = mean_squared_error(test_y, svr_pred)
    # print('SVR mse is : ', svr_mse)
    # svr_mae = mean_absolute_error(test_y, svr_pred)
    # print('SVR mae is : ', svr_mae)
    # svr_r2 = r2_score(test_y, svr_pred)
    # print('SVR r2 is : ', svr_r2)


    # kernel = DotProduct() + WhiteKernel()
    # # gpr = GaussianProcessRegressor(kernel= kernel, random_state=0)
    # gpr = GaussianProcessRegressor()
    # gpr.fit(train_X, train_y)
    # gpr_pred = gpr.predict(test_X)
    # gpr_mse = mean_squared_error(test_y, gpr_pred)
    # print('GaussianProcessRegressor mse is : ', gpr_mse)
    # gpr_mae = mean_absolute_error(test_y, gpr_pred)
    # print('GaussianProcessRegressor mae is : ', gpr_mae)
    # gpr_r2 = r2_score(test_y, gpr_pred)
    # print('GaussianProcessRegressor r2 is : ', gpr_r2)

    # rfr = RandomForestRegressor()
    # rfr.fit(train_X, train_y)
    # rfr_pred = rfr.predict(test_X)
    # rfr_mse = mean_squared_error(test_y, rfr_pred)
    # print('rfr mse is : ', rfr_mse)
    # rfr_mae = mean_absolute_error(test_y, rfr_pred)
    # print('rfr mae is : ', rfr_mae)
    # rfr_r2 = r2_score(test_y, rfr_pred)
    # print('rfr r2 is : ', rfr_r2)

    # gbr = GradientBoostingRegressor()
    # gbr.fit(train_X, train_y)
    # gbr_pred = gbr.predict(test_X)
    # gbr_mse = mean_squared_error(test_y, gbr_pred)
    # print('gbr mse is : ', gbr_mse)
    # gbr_mae = mean_absolute_error(test_y, gbr_pred)
    # print('gbr mae is : ', gbr_mae)
    # gbr_r2 = r2_score(test_y, gbr_pred)
    # print('gbr r2 is : ', gbr_r2)

