import copy

import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from meta_data import DataSet, mate_data, mate_data_1, model_select

dataset_path = 'C:\\Users\\31236\\Desktop\\baseline\\data\\'
datasetnames = np.load('datasetname.npy')
dataset = DataSet('australian', dataset_path=dataset_path)
X = dataset.X
y = dataset.y
trains, tests, label_inds, unlabel_inds = dataset.split_data(test_ratio=0.3, initial_label_rate=0.9, split_count=1, saving_path='.')
print(np.shape(trains))
model = SVC(probability=True)

model.fit(X[trains], y[trains].ravel())
pre = model.predict(X)
print(pre[0:10])

print(accuracy_score(y[tests], pre[tests]))

pp = (model.predict_proba(X)[:, 1] - 0.5) * 2
prediction = np.array([1 if k>0 else -1 for k in pp])
print(prediction[0:10])

print(accuracy_score(y[tests], prediction[tests]))