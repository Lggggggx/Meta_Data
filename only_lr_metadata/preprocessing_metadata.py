import numpy as np 
from sklearn.linear_model import SGDRegressor, LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.tree import DecisionTreeClassifier

from sklearn.externals import joblib

australian_metadata_lr = np.load('./only_lr_metadata/australian_metadata_lr.npy')
print(np.shape(australian_metadata_lr))
X = australian_metadata_lr[:, 0:396]
y = australian_metadata_lr[:, 396]
print(np.shape(y))
# performance changement`s threshold is 1%
effective_positive_improvement = np.where(y >= 0.01)[0]
print(len(effective_positive_improvement))
effective_negative_improvement = np.where(y <= -0.01)[0]
print(len(effective_negative_improvement))

new_X = X[effective_negative_improvement, :]
new_X = np.vstack((new_X, X[effective_positive_improvement, :]))
new_y = y[effective_negative_improvement]
new_y = np.append(new_y, y[effective_positive_improvement])
print(np.shape(new_X))
print(np.shape(new_y))

new_y[np.where(new_y < 0)[0]] = -1
new_y[np.where(new_y > 0)[0]] = 1

# metadata = australian_metadata[0:10000, :]
# metadata = australian_metadata


# X = metadata[:, 0:396]
# y = metadata[:, 396]

# split = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
# split.get_n_splits(X=new_X)
# print(split)

# turn = 0 
# for train_ind, test_ind in split.split(new_X, new_y):
#     turn += 1
#     print('This is the ', turn, '`th**********')

#     train_X = new_X[train_ind]
#     train_y = new_y[train_ind]
#     test_X = new_X[test_ind]
#     test_y = new_y[test_ind]

#     # sgdr = SGDRegressor()
#     # sgdr.fit(train_X, train_y)
#     # sgdr_pred = sgdr.predict(test_X)
#     # sgdr_mse = mean_squared_error(test_y, sgdr_pred)
#     # print('SGDRegressor mean_squared_error is : ', sgdr_mse)
#     # sgdr_mae = mean_absolute_error(test_y, sgdr_pred)
#     # print('SGDRegressor mean_absolute_error is : ', sgdr_mae)
#     # sgdr_r2 = r2_score(test_y, sgdr_pred)
#     # print('SGDRegressor r2_score is : ', sgdr_r2)

#     # lr = LinearRegression()
#     # lr.fit(train_X, train_y)
#     # lr_pred = lr.predict(test_X)
#     # lr_mse = mean_squared_error(test_y, lr_pred)
#     # print('LinearRegression mse is : ', lr_mse)
#     # lr_mae = mean_absolute_error(test_y, lr_pred)
#     # print('LinearRegression mae is : ', lr_mae)
#     # lr_r2 = r2_score(test_y, lr_pred)
#     # print('LinearRegression r2 is :', lr_r2)

#     # svr = SVR()
#     # svr.fit(train_X, train_y)
#     # svr_pred = svr.predict(test_X)
#     # svr_mse = mean_squared_error(test_y, svr_pred)
#     # print('SVR mse is : ', svr_mse)
#     # svr_mae = mean_absolute_error(test_y, svr_pred)
#     # print('SVR mae is : ', svr_mae)
#     # svr_r2 = r2_score(test_y, svr_pred)
#     # print('SVR r2 is : ', svr_r2)


#     # kernel = DotProduct() + WhiteKernel()
#     # # gpr = GaussianProcessRegressor(kernel= kernel, random_state=0)
#     # gpr = GaussianProcessRegressor()
#     # gpr.fit(train_X, train_y)
#     # gpr_pred = gpr.predict(test_X)
#     # gpr_mse = mean_squared_error(test_y, gpr_pred)
#     # print('GaussianProcessRegressor mse is : ', gpr_mse)
#     # gpr_mae = mean_absolute_error(test_y, gpr_pred)
#     # print('GaussianProcessRegressor mae is : ', gpr_mae)
#     # gpr_r2 = r2_score(test_y, gpr_pred)
#     # print('GaussianProcessRegressor r2 is : ', gpr_r2)

#     # rfr = RandomForestRegressor()
#     # rfr.fit(train_X, train_y)
#     # rfr_pred = rfr.predict(test_X)
#     # rfr_mse = mean_squared_error(test_y, rfr_pred)
#     # print('rfr mse is : ', rfr_mse)
#     # rfr_mae = mean_absolute_error(test_y, rfr_pred)
#     # print('rfr mae is : ', rfr_mae)
#     # rfr_r2 = r2_score(test_y, rfr_pred)
#     # print('rfr r2 is : ', rfr_r2)

#     # gbr = GradientBoostingRegressor()
#     # gbr.fit(train_X, train_y)
#     # gbr_pred = gbr.predict(test_X)
#     # gbr_mse = mean_squared_error(test_y, gbr_pred)
#     # print('gbr mse is : ', gbr_mse)
#     # gbr_mae = mean_absolute_error(test_y, gbr_pred)
#     # print('gbr mae is : ', gbr_mae)
#     # gbr_r2 = r2_score(test_y, gbr_pred)
#     # print('gbr r2 is : ', gbr_r2)

#     # lr = LogisticRegression()
#     # lr.fit(train_X, train_y)
#     # lr_pred = lr.predict(test_X)
#     # lr_ac = accuracy_score(test_y, lr_pred)
#     # print('lr accuracy is : ', lr_ac)

#     # rfc = RandomForestClassifier()
#     # rfc.fit(train_X, train_y)
#     # rfc_pred = rfc.predict(test_X)
#     # rcf_ac = accuracy_score(test_y, rfc_pred)
#     # print('rfc accuracy is : ', rcf_ac)

#     # gpc = GaussianProcessClassifier(kernel = DotProduct() + WhiteKernel())
#     # gpc.fit(train_X, train_y)
#     # gpc_pred = gpc.predict(test_X)
#     # gpc_ac = accuracy_score(test_y, gpc_pred)
#     # print('gpc accuracy is : ', gpc_ac)

#     # svc = SVC(kernel='linear')
#     # svc.fit(train_X, train_y)
#     # svc_pred = svc.predict(test_X)
#     # svc_ac = accuracy_score(test_y, svc_pred)
#     # print('svc acccuracy is : ', svc_ac)

#     dtc = DecisionTreeClassifier()
#     dtc.fit(train_X, train_y)
#     dtc_pred = dtc.predict(test_X)
#     dtc_ac = accuracy_score(test_y, dtc_pred)
#     print('dtc accuracy is : ', dtc_ac)

classifier_rfc = RandomForestClassifier()
classifier_rfc.fit(new_X, new_y)
a = classifier_rfc.predict_proba(new_X[0:10,:])
joblib.dump(classifier_rfc, './only_lr_metadata/classifier_rfc.joblib')