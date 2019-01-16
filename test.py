import numpy as np 

# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, mean_squared_error, log_loss, hinge_loss

# from sklearn.svm import SVC

# X=np.array([[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[3,1],[4,1],[5,1],

#        [5,2],[6,1],[6,2],[6,3],[6,4],[3,3],[3,4],[3,5],[4,3],[4,4],[4,5]])

# Y=np.array([1]*14+[-1]*6)

# rp =randperm(len(Y)- 1)



# svc=SVC(kernel='poly',degree=2,gamma=1,coef0=0, probability=True)
# svc.fit(X[rp[0:10]],Y[rp[0:10]])
# pre = svc.predict(X[rp[10:]])
# preprob = svc.predict_proba(X[rp[10:]])
# output = (preprob[:, 1] - 0.5) * 2
# print(accuracy_score(Y[rp[10:]], pre))
# print(roc_auc_score(Y[rp[10:]], output))
# print(roc_auc_score(Y[rp[10:]], preprob[:, 1]))

# print(log_loss(Y[rp[10:]], pre))
# print(log_loss(Y[rp[10:]], output))

# print(mean_squared_error(Y[rp[10:]], pre))
# print(mean_squared_error(Y[rp[10:]], output))

import numpy as np

from meta_data import DataSet, mate_data, model_select, cal_mate_data

dataset_path = './newdata/'
datasetnames = ['echocardiogram']
# Different types of models, each type has many models with different parameters
modelnames = ['KNN']

# in the same dataset and the same ratio of initial_label_rate,the number of split.
split_count = 1
# The number of unlabel data to select to generate the meta data.
num_xjselect = 2

# first choose a dataset
for datasetname in datasetnames:
    dataset = DataSet(datasetname, dataset_path)
    X = dataset.X
    y = dataset.y
    distacne = dataset.get_distance()
    _, cluster_center_index = dataset.get_cluster_center()
    print(datasetname + ' DataSet currently being processed........')
    metadata = None
    # run multiple split on the same dataset
    # every time change the value of initial_label_rate
    for i_l_r in np.arange(0.03, 0.04, 0.01, dtype=float):
        if datasetname in ['echocardiogram', 'heart', 'heart-statlog', 'house', 'spect', 'statlog-heart']:
            if i_l_r <= 0.07:
                i_l_r = 0.07
        trains, tests, label_inds, unlabel_inds = dataset.split_data_labelbalance(test_ratio=0.3, 
             initial_label_rate=i_l_r, split_count=split_count, saving_path='./split')
        meta_data = cal_mate_data(X, y, distacne, cluster_center_index, modelnames,  
             trains, tests, label_inds, unlabel_inds, split_count, num_xjselect)
        if metadata is None:
            metadata = meta_data
        else:
            metadata = np.vstack((metadata, meta_data))
    print(datasetname + ' is complete and saved successfully.')
    np.save(datasetname + "_metadata.npy", metadata)

print("All done!")

