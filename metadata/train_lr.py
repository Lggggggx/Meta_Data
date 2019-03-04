import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

metadata = np.load('./metadata/binary_metadata.npy')

print('train lr')
lr = LogisticRegression()
lr.fit(metadata[:, 0:396], metadata[:, 396])
print('done')

joblib.dump(lr, './metadata/classify_lr.joblib')