import warnings
warnings.simplefilter('ignore')

import numpy as np 

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

positive_cdata = np.load('./processing_metadata_fitting/combination_postive_data.npy')
negative_data = np.load('./processing_metadata_fitting/combination_negative_data.npy')

# print(positive_cdata[0:3])
# print(negative_data[0:3])

c_data = np.vstack((positive_cdata, negative_data))
np.random.shuffle(c_data)
X = c_data[:, 0:792]
y = c_data[:, 792]

lr = LogisticRegression()
lr.fit(X, y)
print(lr.score(X, y))
joblib.dump(lr, './processing_metadata_fitting/lr_cdata.joblib')

# split = ShuffleSplit(n_splits=1, train_size=0.3)

# turn = 0
# for train_index, test_index in split.split(X, y):
#     turn += 1
#     print('~~~~~~this is {0}'.format(turn))

#     train_X = X[train_index]
#     train_y = y[train_index]

#     test_X = X[test_index]
#     test_y = y[test_index]

#     lr = LogisticRegression()
#     lr.fit(train_X, train_y)
#     lr_pred = lr.predict(test_X)
#     lr_ac = accuracy_score(test_y, lr_pred)
#     print('the lr_accuracy is {0}'.format(lr_ac))
#     lr_pre_prob = lr.predict_proba(test_X)
#     print(lr.classes_)
#     print(np.shape(lr_pre_prob))
#     print(lr_pre_prob[0:10, 1])
#     print(np.max(lr_pre_prob[:, 1]))
#     print(len(np.where(lr_pre_prob[:, 1]==1)[0]))


#     rfr = RandomForestClassifier()
#     rfr.fit(train_X, train_y)
#     rfr_pred = rfr.predict(test_X)
#     rfr_ac = accuracy_score(test_y, rfr_pred)
#     print('the rfr_accuracy is {0}'.format(rfr_ac))

#     rfr_pred_prob = rfr.predict_log_proba(test_X)
#     print(np.shape(rfr_pred_prob))
#     print(rfr.classes_)
#     print(rfr_pred_prob[0:10,1])
#     print(np.max(rfr_pred_prob[:,1]))
#     print(np.min(rfr_pred_prob[:,1]))
#     print(np.mean(rfr_pred_prob[:,1]))
#     print(np.std(rfr_pred_prob[:,1]))
#     print(len(np.where(rfr_pred_prob[:,1] == 1)[0]))



