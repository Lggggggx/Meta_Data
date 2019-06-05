import os
import numpy as np

from new_meta_data import DataSet, model_select, newmeta_data, new_cal_meta_data_sequence

dataset_path = './newdata/'

datasetnames = ['australian', 'wdbc']

for datasetname in datasetnames:

    dataset = DataSet(datasetname, dataset_path)
    static_mf = dataset.get_static_meta_features()

    metadata_dir = 'E:/metadata数据集/new_bigmetadata/'+datasetname+'/'
    for _, _, files in os.walk(metadata_dir):
        for file in files:
            temp_metadata = np.load(metadata_dir + file)
            n_sample = np.shape(temp_metadata)[0]
            sameshape_static_mf = np.repeat([static_mf], n_sample, axis=0)
            metadata = np.hstack((sameshape_static_mf, temp_metadata))
            np.save(metadata_dir + file, metadata)

# dataset = DataSet('australian', dataset_path)
# static_mf = dataset.get_static_meta_features()
# print(static_mf.shape)
# d = np.load('E:/metadata数据集/new_bigmetadata/australian/10australian30_big_metadata30.npy')
# d2 = np.load('E:/metadata数据集/new_bigmetadata/australian/2australian30_big_metadata30.npy')
# print(d2.shape)
# print(d.shape)
# d = d[:,19:]
# print(d.shape)
# np.save('E:/metadata数据集/new_bigmetadata/australian/10australian30_big_metadata30.npy', d)