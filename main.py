import numpy as np

from meta_data import DataSet, mate_data, model_select, cal_mate_data

dataset_path = './data/'
datasetnames = np.load('datasetname.npy')
# Different types of models, each type has many models with different parameters
modelnames = ['KNN', 'RFC', 'RFR',
    'DTC', 'DTR', 'SVM', 'GBDT', 'ABC', 'ABR']

# in the same dataset and the same ratio of initial_label_rate,the number of split.
split_count = 10
# The number of unlabel data to select to generate the meta data.
num_xjselect = 20

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
        trains, tests, label_inds, unlabel_inds = dataset.split_data(test_ratio=0.3, 
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