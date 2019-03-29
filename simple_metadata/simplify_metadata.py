import copy
import numpy as np 

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.externals import joblib

metadata_name = 'australian_rfc_metadata'

save_data = False

# metadata = np.load('./orgin_metadata/' + metadata_name + '.npy')
metadata = np.load('./origin_metadata/australian_rfc_metadata.npy')
print(np.shape(metadata))
# the indes of the filtering elements which is reprensentativeness
arr1 = [1, 2, 3, 4] 
arr2 = [i for i in range(35, 59)]
arr3 = [i for i in range(109, 119)]
arr4 = [124, 130]
arr5 = [i for i in range(131, 143)]
arr6 = [i for i in range(193, 203)]
arr7 = [208, 214, 215]
arr8 = [i for i in range(366, 396)]
filtering_index = np.r_[arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, [396, 397, 398, 399]]

# the new_metadata after filtering elements
new_metadata = metadata[:, filtering_index]
print(np.shape(new_metadata))
if save_data:
    np.save('./simple_metadata/simple_'+ metadata_name +'.npy', new_metadata)

new_metadata[np.where(new_metadata[:, 95] > 0)[0], 95] = 1
new_metadata[np.where(new_metadata[:, 95] <= 0)[0], 95] = -1
print(np.shape(new_metadata))


X = copy.deepcopy(new_metadata[:, 0:95])
y_regression = copy.deepcopy(new_metadata[:, 95])

origin_X = metadata[:, 0:396]
origin_y = metadata [:, 396]
origin_y[np.where(origin_y > 0)[0]] = 1
origin_y[np.where(origin_y <= 0)[0]] = -1

print(y_regression[0:10])
print(origin_y[0:10])
split = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
split.get_n_splits(X=X)
print(split)

turn = 0 
for train_ind, test_ind in split.split(X, y_regression):
    turn += 1
    print('This is the ', turn, '`th**********')

    train_X = X[train_ind]
    train_y = y_regression[train_ind]
    test_X = X[test_ind]
    test_y = y_regression[test_ind]

    origin_train_X = origin_X[train_ind]
    origin_test_X = origin_X[test_ind]
    # sgdr = SGDRegressor()
    # sgdr.fit(train_X, train_y)
    # sgdr_pred = sgdr.predict(test_X)
    # sgdr_mse = mean_squared_error(test_y, sgdr_pred)
    # print('SGDRegressor mean_squared_error is : ', sgdr_mse)
    # sgdr_mae = mean_absolute_error(test_y, sgdr_pred)
    # print('SGDRegressor mean_absolute_error is : ', sgdr_mae)
    # sgdr_r2 = r2_score(test_y, sgdr_pred)
    # print('SGDRegressor r2_score is : ', sgdr_r2)

    # lr = LinearRegression()
    # lr.fit(train_X, train_y)
    # lr_pred = lr.predict(test_X)
    # lr_mse = mean_squared_error(test_y, lr_pred)
    # print('LinearRegression mse is : ', lr_mse)
    # lr_mae = mean_absolute_error(test_y, lr_pred)
    # print('LinearRegression mae is : ', lr_mae)
    # lr_r2 = r2_score(test_y, lr_pred)
    # print('LinearRegression r2 is :', lr_r2)

    # rfr = RandomForestRegressor()
    # rfr.fit(train_X, train_y)
    # rfr_pred = rfr.predict(test_X)
    # rfr_mse = mean_squared_error(test_y, rfr_pred)
    # print('rfr mse is : ', rfr_mse)
    # rfr_mae = mean_absolute_error(test_y, rfr_pred)
    # print('rfr mae is : ', rfr_mae)
    # rfr_r2 = r2_score(test_y, rfr_pred)
    # print('rfr r2 is : ', rfr_r2)

    # svc = SVC()
    # svc.fit(train_X, train_y)
    # svc_pred = svc.predict(test_X)
    # svc_ac = accuracy_score(test_y, svc_pred)
    # print('svc ac is : ', svc_ac)

    lr = LogisticRegression()
    lr.fit(train_X, train_y)
    lr_pred = lr.predict(test_X)
    lr_ac = accuracy_score(test_y, lr_pred)
    print('lr ac is : ', lr_ac)

    rfc = RandomForestClassifier()
    rfc.fit(train_X, train_y)
    rfc_pred = rfc.predict(test_X)
    rfc_ac = accuracy_score(test_y, rfc_pred)
    print('rfc ac is : ', rfc_ac)

    lr2 = LogisticRegression()
    lr2.fit(origin_train_X, train_y)
    lr_pred2 = lr2.predict(origin_test_X)
    lr_ac2 = accuracy_score(test_y, lr_pred2)
    print('origin lr ac is : ', lr_ac2)

    rfc2 = RandomForestClassifier()
    rfc2.fit(origin_train_X, train_y)
    rfc_pred2 = rfc2.predict(origin_test_X)
    rfc_ac2 = accuracy_score(test_y, rfc_pred2)
    print('origin rfc ac is : ', rfc_ac2)



if save_data:
    np.save('./simple_metadata/classify_simple_'+ metadata_name +'.npy', new_metadata)



# print(len(np.where(y >= 0.01)[0]))
# print(len(np.where(y <= -0.01)[0]))

# print(np.shape(np.array([new_y]).T))
# process_ethn_metadata = np.hstack((new_X, np.array([new_y]).T))
# print(np.shape(process_ethn_metadata))

# print(y[np.where(y >= 0.01)[0]][0:10])
# print(process_ethn_metadata[0:10, 396])
# print(np.shape(new_y))
# print(y[0:20])
# print(new_y[0:20])

# new_metadata = np.hstack((X, new))
# print(np.shape(new_metadata))
# print(new_metadata[0:20, 396])           
