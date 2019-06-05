import numpy as np

import warnings
warnings.filterwarnings("ignore")

from new_meta_data import DataSet, model_select, newmeta_data, new_cal_meta_data_sequence

if __name__ == "__main__":
    
    dataset_path = './newdata/'
    # datasetnames = np.load('datasetname.npy')
    # datasetnames = ['echocardiogram', 'heart', 'heart-hungarian', 'heart-statlog', 'house',
    #                     'house-votes', 'spect', 'statlog-heart', 'vertebral-column-2clases']
    # 'australian', 'wdbc', 'clean1', 'ethn', , 'blood', 'breast-cancer-wisc'
    # , 'statlog-heart', 'statlog-german-credit', 'pima', 'ringnorm', 'oocytes_merluccius_nucleus_4d'
    datasetnames = ['australian', 'wdbc']
    # Different types of models, e, ach type has many models with different parameters
    # modelnames = ['KNN', 'LR', 'RFC', 'RFR', 'DTC', 'DTR', 'SVM', 'GBC', 'ABC', 'ABR']
    modelnames = ['LR']

    # in the same dataset and the same ratio of initial_label_rate,the number of split.
    split_count = 2
    # The number of unlabel data to select to generate the meta data.

    query_time = [2]
    print('query_time: ', query_time)
    n_labelleds = np.arange(2, 4, 2)
    print('n_labelleds: ', n_labelleds)
    diff_five_round = 20
    # first choose a dataset
    for datasetname in datasetnames:
    
        dataset = DataSet(datasetname, dataset_path)
        X = dataset.X
        y = dataset.y 
        _, cluster_center_index = dataset.get_cluster_center()
        distance, min_distance, max_distance = dataset.get_distance()
        node_potential = dataset.get_node_potential()
        gra_matrix, p_matrix = dataset.get_graph_density()
        static_mf = dataset.get_static_meta_features()

        print(datasetname + ' DataSet currently being processed********************************************')
        
        # run multiple split on the same dataset
        # every time change the value of initial_label_rate
        for num_xjselect in query_time:
            for n_labelled in n_labelleds:
                metadata = None
                # trains, tests, label_inds, unlabel_inds = dataset.split_data_by_nlabelled(n_labelled, test_ratio=0.6, split_count=split_count, saving_path='./n_labelled_split_info')
                trains, tests, label_inds, unlabel_inds = dataset.split_data_by_nlabelled_fulldataset(n_labelled, test_ratio=0.5, split_count=split_count)
                for t in range(split_count):
                    meta_data = new_cal_meta_data_sequence(X, y, distance, min_distance, max_distance, node_potential, gra_matrix, p_matrix, cluster_center_index, modelnames,  
                        tests[t], label_inds[t], unlabel_inds[t], t, num_xjselect, diff_five_round)
                    print(np.shape(meta_data))
                    
                    if metadata is None:
                        metadata = meta_data
                    else:
                        metadata = np.vstack((metadata, meta_data))       
                    # print(np.shape(metadata))
                    # print(metadata)
                np.save('./new_bigmetadata/'+str(n_labelled)+datasetname +str(split_count)+ '_big_metadata'+str(num_xjselect)+'.npy', metadata)           

            # print(datasetname + ' is complete and saved successfully.')
            # np.save('./bigmetadata/'+datasetname + '_big_metadata.npy', metadata)

    print("All done!")
