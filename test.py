import numpy as np 
import h5py
# a= np.load('metadata1.npy')
# b= np.load('metadata.npy')

# print(np.shape(b))

# print(b[100:200, 396])

# from sklearn.model_selection import GroupShuffleSplit

# from sklearn.svm import SVC

# X=np.array([[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[3,1],[4,1],[5,1],

#        [5,2],[6,1],[6,2],[6,3],[6,4],[3,3],[3,4],[3,5],[4,3],[4,4],[4,5]])

# Y=np.array([1]*14+[-1]*6)

# T=np.array([[0.5,0.5],[1.5,1.5],[3.5,3.5],[4,5.5]])

# svc=SVC(kernel='poly',degree=2,gamma=1,coef0=0, probability=True)

# svc.fit(X,Y)

# pre=svc.predict(T)

# print(pre)
# print(svc.predict_proba(T))
# print((svc.predict_proba(T)[:, 1] - 0.5) * 2)

# print(svc.predict_log_proba(T))


# import numpy as np


from sklearn.model_selection import StratifiedKFold


# dt = h5py.File('.\data/australian.mat', 'r')
# X = np.transpose(dt['x'])
# y = np.transpose(dt['y'])

# # X=np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
# # y=np.array([1,1,1,2,2,2])
# skf=StratifiedKFold(n_splits=3)
# skf.get_n_splits(X,y)
# print(skf)
# for train_index,test_index in skf.split(X,y):
#     print("Train Index:",np.shape(train_index),",Test Index:",np.shape(test_index))
#     print(np.shape(np.unique(train_index)))
#     print(np.shape(np.unique(test_index)))

#     # X_train,X_test=X[train_index],X[test_index]
#     # y_train,y_test=y[train_index],y[test_index]
#     # print('y_train ',y_train)
#     # print('y_test ',y_test)

a = np.array([[1,2],[3,4]])
b= [0, 0]
for _ in range(2):
    print(np.vstack((a, b)))
    print(a[1])
    print(b-a[1])
    print(np.r_[b, b-a[1]])