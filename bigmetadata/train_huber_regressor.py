import numpy as np 
from sklearn.linear_model import HuberRegressor, Ridge, Lasso
from sklearn.externals import joblib
from sklearn.metrics import r2_score

metadata1 = np.load('E:/australian/model1/query_time50/2australian30_big_metadata50.npy')
metadata2 = np.load('E:/australian/model1/query_time50/20australian30_big_metadata50.npy')
metadata3 = np.load('E:/australian/model1/query_time50/60australian30_big_metadata50.npy')
metadata4 = np.load('E:/australian/model1/query_time50/90australian30_big_metadata50.npy')
metadata5 = np.load('E:/australian/model1/query_time50/6australian30_big_metadata50.npy')
metadata6 = np.load('E:/australian/model1/query_time50/10australian30_big_metadata50.npy')
metadata7 = np.load('E:/australian/model1/query_time50/14australian30_big_metadata50.npy')




metadata = np.vstack((metadata1, metadata2, metadata3, metadata4, metadata5, metadata6, metadata7))

n_samples = metadata.shape[0]
print(n_samples)
index = np.arange(n_samples)
np.random.shuffle(index)

X = metadata[index, 0:396]
y = metadata[index, 396]

# huber_re = HuberRegressor()

# huber_re.fit(X, y)

# pred1 = huber_re.predict(X)
# print("trainning r2 score :", r2_score(y, pred1))

# print("total r2 score :", r2_score(metadata[:, 396], huber_re.predict(metadata[:, 0:396])))

ridge_alpga1 = Ridge(alpha=0.000001, copy_X=True)
ridge_alpga1.fit(X, y)

print("ridge_alpga1 trainning r2 score :", r2_score(y, ridge_alpga1.predict(X)))
joblib.dump(ridge_alpga1, './australian_qt50_7_ridge.joblib')

# ridge_alpga02 = Ridge(alpha=0.00001, copy_X=True)
# ridge_alpga02.fit(X, y)

# print("ridge_alpga1 trainning r2 score :", r2_score(y, ridge_alpga02.predict(X)))

# lasso_r = Lasso(alpha=0.0001, copy_X=True)

# lasso_r.fit(X, y)

# print("lasso_r trainning r2 score :", r2_score(y, lasso_r.predict(X)))
