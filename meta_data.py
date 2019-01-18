"""
Meta features designing for binary classification tasks 
 in the pool based active learning scenario.
"""
import os
import copy
import h5py
import numpy as np 
import scipy.io as sio

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, mean_squared_error, log_loss, hinge_loss
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import StratifiedKFold

def randperm(n, k=None):
    """Generate a random array which contains k elements range from (n[0]:n[1])

    Parameters
    ----------
    n: int or tuple
        range from [n[0]:n[1]], include n[0] and n[1].
        if an int is given, then n[0] = 0

    k: int, optional (default=end - start + 1)
        how many numbers will be generated. should not larger than n[1]-n[0]+1,
        default=n[1] - n[0] + 1.

    Returns
    -------
    perm: list
        the generated array.
    """
    if isinstance(n, np.generic):
        n = np.asscalar(n)
    if isinstance(n, tuple):
        if n[0] is not None:
            start = n[0]
        else:
            start = 0
        end = n[1]
    elif isinstance(n, int):
        start = 0
        end = n
    else:
        raise TypeError("n must be tuple or int.")

    if k is None:
        k = end - start + 1
    if not isinstance(k, int):
        raise TypeError("k must be an int.")
    if k > end - start + 1:
        raise ValueError("k should not larger than n[1]-n[0]+1")

    randarr = np.arange(start, end + 1)
    np.random.shuffle(randarr)
    return randarr[0:k]


class DataSet():
    """

    Parameters
    ----------
    X: 2D array, optional (default=None) [n_samples, n_features]
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None) [n_samples]
        Label matrix of the whole dataset. It is a reference which will not use additional memory.
        
    """
    def __init__(self, dataset_name, dataset_path=None, X=None, y=None):   
        self.dataset_name = dataset_name
        if dataset_path:
            self.get_dataset(dataset_path)
        elif (X is not None) and (y is not None) :
            self.X = X
            self.y = y
        else:
            raise ValueError("Please input dataset_path or X, y")
        self.n_samples, self.n_features = np.shape(self.X)
        self.distance = None
    
    def get_dataset(self, dataset_path):
        """
        Get the dataset by name.
        The dataset format is *.mat.
        """
        filename = dataset_path + self.dataset_name +'.mat'
        # dt = h5py.File(filename, 'r')
        # self.X = np.transpose(dt['x'])
        # self.y = np.transpose(dt['y'])
        dt = sio.loadmat(filename)
        self.X = dt['x']
        self.y = dt['y']
    
    def get_cluster_center(self, n_clusters=10, method='Euclidean'):
        """Use the Kmeans in sklearn to get the cluster centers.

        Parameters
        ----------
        n_clusters: int 
            The number of cluster centers.
        Returns
        -------
        data_cluster_centers: np.ndarray
            The samples in origin dataset X is the closest to the cluster_centers.

        index_cluster_centers: np.ndarray
            The cluster centers index corresponding to the samples in origin data set.     
        """
        # if self.distance is None:
        #     self.get_distance()
        data_cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(self.X)
        data_origin_cluster_centers = data_cluster.cluster_centers_
        closest_distance_data_cluster_centers = np.zeros(n_clusters) + np.infty
        index_cluster_centers = np.zeros(n_clusters, dtype=int) - 1
 
        # obtain the cluster centers index
        for i in range(self.n_samples):
            for j in range(n_clusters):
                if method == 'Euclidean':
                    distance = np.linalg.norm(self.X[i] - data_origin_cluster_centers[j])
                    if distance < closest_distance_data_cluster_centers[j]:
                        closest_distance_data_cluster_centers[j] = distance
                        index_cluster_centers[j] = i

        if(np.any(index_cluster_centers == -1)):
            raise IndexError("data_cluster_centers_index is wrong")

        return self.X[index_cluster_centers], index_cluster_centers

    def get_distance(self, method='Euclidean'):
        """

        Parameters
        ----------
        method: str
            The method calculate the distance.
        Returns
        -------
        distance_martix: 2D
            D[i][j] reprensts the distance between X[i] and X[j].
        """
        if self.n_samples == 1:
            raise ValueError("There is only one sample.")
        
        distance = np.zeros((self.n_samples, self.n_samples))
        for i in range(1, self.n_samples):
            for j in range(i+1, self.n_samples):
                if method == 'Euclidean':
                    distance[i][j] = np.linalg.norm(self.X[i] - self.X[j])
        
        self.distance = distance + distance.T
        return self.distance
    
    def split_data_labelbalance(self, test_ratio=0.3, initial_label_rate=0.05, split_count=10, saving_path='.'):
        """Split given data considered the problem of label balance.
        The train test label unlabel sets` proportion of positive and negative categories should be consistent with the original data set.

        Parameters
        ----------
        test_ratio: float, optional (default=0.3)
            Ratio of test set

        initial_label_rate: float, optional (default=0.05)
            Ratio of initial label set
            e.g. Initial_labelset*(1-test_ratio)*n_samples

        split_count: int, optional (default=10)
            Random split data _split_count times

        saving_path: str, optional (default='.')
            Giving None to disable saving.

        Returns
        -------
        train_idx: list
            index of training set, shape like [n_split_count, n_training_indexes]

        test_idx: list
            index of testing set, shape like [n_split_count, n_testing_indexes]

        label_idx: list
            index of labeling set, shape like [n_split_count, n_labeling_indexes]

        unlabel_idx: list
            index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]
        """
        # check parameters
        len_of_parameters = [len(self.X) if self.X is not None else None, len(self.y) if self.y is not None else None]
        number_of_instance = np.unique([i for i in len_of_parameters if i is not None])
        if len(number_of_instance) > 1:
            raise ValueError("Different length of instances and _labels found.")
        else:
            number_of_instance = number_of_instance[0]
        
        positive_ind = np.where(self.y == 1)[0]
        negative_ind = np.where(self.y == -1)[0]
        # split
        train_idx = []
        test_idx = []
        label_idx = []
        unlabel_idx = []
        for i in range(split_count):
            prp = randperm(len(positive_ind) - 1)
            nrp = randperm(len(negative_ind) - 1)
            p_cutpoint = int((1 - test_ratio) * len(prp))
            n_cutpoint = int((1 - test_ratio) * len(nrp))
            if p_cutpoint <= 1:
                p_cutpoint = 1
            if n_cutpoint <= 1:
                n_cutpoint = 1
            tp_train = np.r_[positive_ind[prp[1:p_cutpoint]], negative_ind[nrp[1:n_cutpoint]]]
            tp_train = tp_train[randperm(len(tp_train) - 1)]
            # guarantee there is at least one positive and negative instance in label_index
            tp_train = np.r_[positive_ind[prp[0]], negative_ind[nrp[0]], tp_train]
            train_idx.append(tp_train)
            tp_test = np.r_[positive_ind[prp[p_cutpoint:]], negative_ind[nrp[n_cutpoint:]]]
            test_idx.append(tp_test[randperm(len(tp_test) - 1)])

            cutpoint = int(initial_label_rate * len(tp_train))
            if cutpoint <= 2:
                cutpoint = 2
            label_idx.append(tp_train[0:cutpoint])
            unlabel_idx.append(tp_train[cutpoint:])

        self.split_save(train_idx=train_idx, test_idx=test_idx, label_idx=label_idx,
                unlabel_idx=unlabel_idx, initial_label_rate=initial_label_rate, path=saving_path)
        return train_idx, test_idx, label_idx, unlabel_idx

    def split_data(self, test_ratio=0.3, initial_label_rate=0.05, split_count=10, saving_path='.'):
        """Split given data.

        Parameters
        ----------
        test_ratio: float, optional (default=0.3)
            Ratio of test set

        initial_label_rate: float, optional (default=0.05)
            Ratio of initial label set
            e.g. Initial_labelset*(1-test_ratio)*n_samples

        split_count: int, optional (default=10)
            Random split data _split_count times

        saving_path: str, optional (default='.')
            Giving None to disable saving.

        Returns
        -------
        train_idx: list
            index of training set, shape like [n_split_count, n_training_indexes]

        test_idx: list
            index of testing set, shape like [n_split_count, n_testing_indexes]

        label_idx: list
            index of labeling set, shape like [n_split_count, n_labeling_indexes]

        unlabel_idx: list
            index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]
        """
        # check parameters
        len_of_parameters = [len(self.X) if self.X is not None else None, len(self.y) if self.y is not None else None]
        number_of_instance = np.unique([i for i in len_of_parameters if i is not None])
        if len(number_of_instance) > 1:
            raise ValueError("Different length of instances and _labels found.")
        else:
            number_of_instance = number_of_instance[0]

        instance_indexes = np.arange(number_of_instance)

        # split
        train_idx = []
        test_idx = []
        label_idx = []
        unlabel_idx = []
        for i in range(split_count):
            rp = randperm(number_of_instance - 1)
            cutpoint = int((1 - test_ratio) * len(rp))
            tp_train = instance_indexes[rp[0:cutpoint]]
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            cutpoint = int(initial_label_rate * len(tp_train))
            if cutpoint <= 1:
                cutpoint = 1
            label_idx.append(tp_train[0:cutpoint])
            unlabel_idx.append(tp_train[cutpoint:])

        # self.split_save(train_idx=train_idx, test_idx=test_idx, label_idx=label_idx,
        #         unlabel_idx=unlabel_idx, initial_label_rate=initial_label_rate, path=saving_path)
        return train_idx, test_idx, label_idx, unlabel_idx

    def split_load(self, path, datasetname, initial_label_rate):
        """Load split from path.

        Parameters
        ----------
        path: str
            Path to a dir which contains train_idx.txt, test_idx.txt, label_idx.txt, unlabel_idx.txt.

        Returns
        -------
        train_idx: list
            index of training set, shape like [n_split_count, n_training_samples]

        test_idx: list
            index of testing set, shape like [n_split_count, n_testing_samples]

        label_idx: list
            index of labeling set, shape like [n_split_count, n_labeling_samples]

        unlabel_idx: list
            index of unlabeling set, shape like [n_split_count, n_unlabeling_samples]
        """
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))
        saving_path = os.path.abspath(path)
        if not os.path.isdir(saving_path):
            raise Exception("A path to a directory is expected.")

        ret_arr = []

        for fname in ['_train_idx.npy', '_test_idx.npy', '_label_idx.npy', '_unlabel_idx.npy']:
            if os.path.exists(os.path.join(saving_path, datasetname + initial_label_rate.__str__() + fname)):
                ret_arr.append(np.load(os.path.join(saving_path, datasetname + initial_label_rate.__str__() + fname)))
            else:
                raise ValueError("the {0} split information does not exit.".format(datasetname + initial_label_rate.__str__()))        

        # for fname in ['train_idx.npy', 'test_idx.npy', 'label_idx.npy', 'unlabel_idx.npy']:
        #     if not os.path.exists(os.path.join(saving_path, fname)):
        #         if os.path.exists(os.path.join(saving_path, fname.split()[0] + '.npy')):
        #             ret_arr.append(np.load(os.path.join(saving_path, fname.split()[0] + '.npy')))
        #         else:
        #             ret_arr.append(None)
        #     else:
        #         ret_arr.append(np.loadtxt(os.path.join(saving_path, fname)))
        
        return ret_arr[0], ret_arr[1], ret_arr[2], ret_arr[3]

    def split_save(self, train_idx, test_idx, label_idx, unlabel_idx, initial_label_rate, path):
        """Save the split to file for auditting or loading for other methods.

        Parameters
        ----------
        saving_path: str
            path to save the settings. If a dir is not provided, it will generate a folder called
            'alipy_split' for saving.

        """
        if path is None:
            return
        else:
            if not isinstance(path, str):
                raise TypeError("A string is expected, but received: %s" % str(type(path)))

        saving_path = os.path.abspath(path)
        if os.path.isdir(saving_path):
            np.save(os.path.join(saving_path, self.dataset_name + initial_label_rate.__str__() + '_train_idx.npy'), train_idx)
            np.save(os.path.join(saving_path, self.dataset_name + initial_label_rate.__str__() + '_test_idx.npy'), test_idx)
            np.save(os.path.join(saving_path, self.dataset_name + initial_label_rate.__str__() + '_label_idx.npy'), label_idx)
            np.save(os.path.join(saving_path, self.dataset_name + initial_label_rate.__str__() + '_unlabel_idx.npy'), unlabel_idx)
        else:
            raise Exception("A path to a directory is expected.")


def mate_data(X, y, distance, cluster_center_index, label_indexs, unlabel_indexs, modelOutput, query_index):
    """Calculate the meta data according to the current model,dataset and five rounds before information.


    Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y:  {list, np.ndarray}
        The true label of the each round of iteration,corresponding to label_indexs.
    
    distance: 2D
        distance[i][j] reprensts the distance between X[i] and X[j].

    cluster_center_index: np.ndarray
        The index corresponding to the samples which is the result of cluster in origin data set.  

    label_indexs: {list, np.ndarray} shape=(number_iteration, corresponding_label_index)
        The label indexs of each round of iteration,

    unlabel_indexs: {list, np.ndarray} shape=(number_iteration, corresponding_unlabel_index)
        The unlabel indexs of each round of iteration,

    modelOutput: {list, np.ndarray} shape=(number_iteration, corresponding_perdiction)

    query_index: int
        The unlabel sample will be queride,and calculate the performance improvement after add to the labelset.
        
    Returns
    -------
    metadata: 1d-array
        The meta data about the current model and dataset.
    """
    if(np.any(cluster_center_index == -1)):
        raise IndexError("cluster_center_index is wrong")
    if len(label_indexs) != len(modelOutput) or len(unlabel_indexs) != len(modelOutput) or len(unlabel_indexs) != len(label_indexs):
        raise ValueError("the shape of {label_indexs, unlabel_indexs, modelOutput} is inconsonant")
    for i in range(5):
        assert(np.shape(X)[0] == np.shape(modelOutput[i])[0]) 
        if(not isinstance(label_indexs[i], np.ndarray)):
            label_indexs[i] = np.array(label_indexs[i])
        if(not isinstance(unlabel_indexs[i], np.ndarray)):
            unlabel_indexs[i] = np.array(unlabel_indexs[i])
    
    n_samples, n_feature = np.shape(X)

    current_label_size = len(label_indexs[5])
    current_label_y = y[label_indexs[5]]
    current_unlabel_size = len(unlabel_indexs[5])
    current_prediction = modelOutput[5]

    ratio_label_positive = (sum(current_label_y > 0)) / current_label_size
    ratio_label_negative = (sum(current_label_y < 0)) / current_label_size

    ratio_unlabel_positive = (sum(current_prediction[unlabel_indexs[5]] > 0)) / current_unlabel_size
    ratio_unlabel_negative = (sum(current_prediction[unlabel_indexs[5]] < 0)) / current_unlabel_size

    sorted_labelperdiction_index = np.argsort(current_prediction[label_indexs[5]])
    sorted_current_label_data = X[label_indexs[5][sorted_labelperdiction_index]]
    
    label_10_equal_index = [label_indexs[5][sorted_labelperdiction_index][int(i * current_label_size)] for i in np.arange(0, 1, 0.1)]

    sorted_unlabelperdiction_index = np.argsort(current_prediction[unlabel_indexs[5]])
    sorted_current_unlabel_data = X[unlabel_indexs[5][sorted_unlabelperdiction_index]]
    unlabel_10_equal_index = [unlabel_indexs[5][sorted_unlabelperdiction_index][int(i * current_unlabel_size)] for i in np.arange(0, 1, 0.1)]
     
    cc = []
    l10e = []
    u10e = []
    for j in range(10):
        cc.append(distance[query_index][cluster_center_index[j]])
        l10e.append(distance[query_index][label_10_equal_index[j]])
        u10e.append(distance[query_index][unlabel_10_equal_index[j]])

    cc = minmax_scale(cc)
    cc_sort_index = np.argsort(cc)
    l10e = minmax_scale(l10e)
    u10e = minmax_scale(u10e)
    distance_query_data = np.hstack((cc[cc_sort_index], l10e, u10e))

    ratio_tn = []
    ratio_fp = []
    ratio_fn = []
    ratio_tp = []
    label_pre_10_equal = []
    labelmean = []
    labelstd = []
    unlabel_pre_10_equal = []
    round5_ratio_unlabel_positive = []
    round5_ratio_unlabel_negative = []
    unlabelmean = []
    unlabelstd = []   
    for i in range(6):
        label_size = len(label_indexs[i])
        unlabel_size = len(unlabel_indexs[i])
        # cur_prediction = modelOutput[i]
        cur_prediction = np.array([1 if k>0 else -1 for k in modelOutput[i]])
        label_ind = label_indexs[i]
        unlabel_ind = unlabel_indexs[i]

        tn, fp, fn, tp = confusion_matrix(y[label_ind], cur_prediction[label_ind], labels=[-1, 1]).ravel()
        ratio_tn.append(tn / label_size)
        ratio_fp.append(fp / label_size)
        ratio_fn.append(fn / label_size)
        ratio_tp.append(tp / label_size)

        sort_label_pred = np.sort(minmax_scale(modelOutput[i][label_ind]))
        i_label_10_equal = [sort_label_pred[int(i * label_size)] for i in np.arange(0, 1, 0.1)]
        label_pre_10_equal = np.r_[label_pre_10_equal, i_label_10_equal]
        labelmean.append(np.mean(i_label_10_equal))
        labelstd.append(np.std(i_label_10_equal))

        round5_ratio_unlabel_positive.append((sum(current_prediction[unlabel_ind] > 0)) / unlabel_size)
        round5_ratio_unlabel_negative.append((sum(current_prediction[unlabel_ind] < 0)) / unlabel_size)
        sort_unlabel_pred = np.sort(minmax_scale(modelOutput[i][unlabel_ind]))
        i_unlabel_10_equal = [sort_unlabel_pred[int(i * unlabel_size)] for i in np.arange(0, 1, 0.1)]
        unlabel_pre_10_equal = np.r_[unlabel_pre_10_equal, i_unlabel_10_equal]
        unlabelmean.append(np.mean(i_unlabel_10_equal))
        unlabelstd.append(np.std(i_unlabel_10_equal))
    model_infor = np.hstack((ratio_tp, ratio_fp, ratio_tn, ratio_fn, label_pre_10_equal, labelmean, labelstd, \
         round5_ratio_unlabel_positive, round5_ratio_unlabel_negative, unlabel_pre_10_equal, unlabelmean, unlabelstd))

    f_x_a = []
    f_x_c = []
    f_x_d = []
    for round in range(6):
        model_output = minmax_scale(modelOutput[round])
        for j in range(10):
            f_x_a.append(model_output[query_index] - model_output[cluster_center_index[cc_sort_index[j]]])
        for j in range(10):
            f_x_c.append(model_output[query_index] - model_output[label_10_equal_index[j]])
        for j in range(10):
            f_x_d.append(model_output[query_index] - model_output[unlabel_10_equal_index[j]])
    fdata = np.hstack((current_prediction[query_index], f_x_a, f_x_c, f_x_d))

    metadata = np.hstack((n_feature, ratio_label_positive, ratio_label_negative, \
         ratio_unlabel_positive, ratio_unlabel_negative, distance_query_data, model_infor, fdata))
    metadata = np.array([metadata])
    return metadata


def model_select(modelname):
    """
    Parameters
    ----------
    modelname: str
        The name of model.
        'KNN', 'LR', 'RFC', 'RFR', 'DTC', 'DTR', 'SVM', 'GBC', 'ABC', 'ABR'

    Returns
    -------
    models: list
        The models in sklearn with corresponding parameters.
    """

    if modelname not in ['KNN', 'LR', 'RFC', 'RFR', 'DTC', 'DTR', 'SVM', 'GBC', 'ABC', 'ABR']:
        raise ValueError("There is no " + modelname)

    if modelname == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier 
        models = []
        n_neighbors_parameter = [3, 4, 5, 6]
        algorithm_parameter = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size_parameter = [25, 30, 35]
        p_parameter = [1, 2]
        for n in n_neighbors_parameter:
            for a in algorithm_parameter:
                for l in leaf_size_parameter:
                    for p in p_parameter:
                        models.append(KNeighborsClassifier(n_neighbors=n, algorithm=a, leaf_size=l, p=p))
        return models 

    if modelname == 'LR':
        from sklearn.linear_model import LogisticRegression
        models = []
        # penalty_parameter = ['l1', 'l2']
        C_parameter = [1e-1, 0.5, 1]
        tol_parameter = [1e-5, 1e-4, 1e-3]
        solver_parameter = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        for c in C_parameter:
            for t in tol_parameter:
                for s in solver_parameter:
                    models.append(LogisticRegression(C=c, tol=t, solver=s))
        return models

    if modelname == 'RFC':
        from sklearn.ensemble import RandomForestClassifier
        models = []
        n_estimators_parameter = [10, 40, 70, 110]
        max_features_parameter = ['auto', 'sqrt', 'log2', None]
        for n in n_estimators_parameter:
            for m in max_features_parameter:
                models.append(RandomForestClassifier(n_estimators=n, max_features=m))
        return models
    
    if modelname == 'RFR':
        from sklearn.ensemble import RandomForestRegressor
        models = []
        n_estimators_parameter = [10, 40, 70, 110]
        max_features_parameter = ['auto', 'sqrt', 'log2', None]
        for n in n_estimators_parameter:
            for m in max_features_parameter:
                models.append(RandomForestRegressor(n_estimators=n, max_features=m))
        return models
    
    if modelname == 'DTC':
        from sklearn.tree import DecisionTreeClassifier
        models = []
        splitter_parameter = ['best', 'random']
        max_features_parameter = ['auto', 'sqrt', 'log2', None]
        for s in splitter_parameter:
            for m in max_features_parameter:
                models.append(DecisionTreeClassifier(splitter=s, max_features=m))
        return models

    if modelname == 'DTR':
        from sklearn.tree import DecisionTreeRegressor
        models = []
        splitter_parameter = ['best', 'random']
        max_features_parameter = ['auto', 'sqrt', 'log2', None]
        for s in splitter_parameter:
            for m in max_features_parameter:
                models.append(DecisionTreeRegressor(splitter=s, max_features=m))
        return models   

    if modelname == 'SVM':
        from sklearn.svm import SVC
        models = []
        C_parameter = [1e-1, 0.5, 1]
        kernel_parameter = ['linear', 'poly', 'sigmoid']
        # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        degree_parameter = [2, 3, 4]
        tol_parameter = [1e-5, 1e-4, 1e-3]
        for c in C_parameter:
            for k in kernel_parameter:
                for t in tol_parameter:
                    if k == 'poly':             
                        for d in degree_parameter:   
                            models.append(SVC(C=c ,kernel=k, degree=d, tol=t, probability=True))
                    else:
                        models.append(SVC(C=c ,kernel=k, tol=t, probability=True))
        return models

    if modelname == 'GBC':
        from sklearn.ensemble import GradientBoostingClassifier
        models = []
        loss_parameter = ['deviance', 'exponential']
        learning_rate_parameter = [0.02, 0.05, 0.1]
        n_estimators_parameter = [40, 70, 110]
        max_features_parameter = ['auto', 'sqrt', 'log2', None]
        for l in loss_parameter:
            for le in learning_rate_parameter:
                for n in n_estimators_parameter:
                    for mf in max_features_parameter:
                        models.append(GradientBoostingClassifier(loss=l, learning_rate=le, n_estimators=n, max_features=mf))
        return models    

    if modelname == 'ABC':
        from sklearn.ensemble import AdaBoostClassifier
        models = []
        learning_rate_parameter = [0.02, 0.05, 0.1]
        n_estimators_parameter = [40, 70, 110]
        for le in learning_rate_parameter:
            for n in n_estimators_parameter:
                models.append(AdaBoostClassifier(learning_rate=le, n_estimators=n))
        return models    

    if modelname == 'ABR':
        from sklearn.ensemble import AdaBoostRegressor
        models = []
        learning_rate_parameter = [0.02, 0.05, 0.1]
        n_estimators_parameter = [40, 70, 110]
        loss_parameter = ['linear', 'square', 'exponential']
        for le in learning_rate_parameter:
            for n in n_estimators_parameter:
                for l in loss_parameter:
                    models.append(AdaBoostRegressor(learning_rate=le, n_estimators=n, loss=l))
        return models  
    

def cal_mate_data(X, y, distacne, cluster_center_index, modelnames, trains, tests, label_inds, unlabel_inds, split_count, num_xjselect):
    """calculate the designed mate data. 
    Parameters
    ----------
    X: 2D array, optional (default=None) [n_samples, n_features]
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None) [n_samples]
        Label matrix of the whole dataset. It is a reference which will not use additional memory.
    
    distance_martix: 2D
        D[i][j] reprensts the distance between X[i] and X[j].

    cluster_center_index: np.ndarray
        The cluster centers index corresponding to the samples in origin data set. 

    modelname: str
    The name of model.
    'KNN', 'LR', 'RFC', 'RFR', 'DTC', 'DTR', 'SVM', 'GBDT', 'ABC', 'ABR'

    train_idx: list
        index of training set, shape like [n_split_count, n_training_indexes]

    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_indexes]

    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_indexes]

    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]

    split_count: int, optional
        Random split data _split_count times
    
    num_xjselect: int 
        The number of unlabel data to select to generate the meta data.

    Returns
    -------
    metadata: 2D
        The meta data about the current model and dataset.[num_xjselect, 396 + 1(features + label)]

    """
    metadata = None
    for t in range(split_count):  
        label_inds_t = label_inds[t]
        unlabel_inds_t = unlabel_inds[t]
        test = tests[t]
        for modelname in modelnames:
            # choose one type models
            models = model_select(modelname)

            # the same type model with different parameters
            num_models = len(models)
            for k in range(num_models):
                model = models[k]
                # Repeated many(20) times in the same model and split
                for _ in range(20):
                    # genearte five rounds before
                    l_ind = copy.deepcopy(label_inds_t)
                    u_ind = copy.deepcopy(unlabel_inds_t)
                    modelOutput = []
                    modelPerformance = None
                    labelindex = []
                    unlabelindex = []
                    for i in range(5):
                        i_sampelindex = np.random.choice(u_ind)
                        u_ind = np.delete(u_ind, np.where(u_ind == i_sampelindex)[0])
                        l_ind = np.r_[l_ind, i_sampelindex]
                        labelindex.append(l_ind)
                        unlabelindex.append(u_ind)

                        model_i = copy.deepcopy(model)
                        model_i.fit(X[l_ind], y[l_ind].ravel())
                        if modelname in ['RFR', 'DTR', 'ABR']:
                            i_output = model_i.predict(X)
                        else:
                            i_output = (model_i.predict_proba(X)[:, 1] - 0.5) * 2
                        i_prediction = np.array([1 if k>0 else -1 for k in i_output])
                        modelOutput.append(i_output)
                        i_acc = accuracy_score(y[test], i_prediction[test])
                        i_roc = roc_auc_score(y[test], i_output[test])
                        i_mse = mean_squared_error(y[test], i_prediction[test])
                        i_ll = log_loss(y[test], i_prediction[test])
                        
                        if modelPerformance is None:
                            modelPerformance = np.array([i_acc, i_roc, i_mse, i_ll])
                        else:
                            modelPerformance = np.vstack((modelPerformance, [i_acc, i_roc, i_mse, i_ll]))
                    
                    # calualate the meta data z(designed features) and r(performance improvement) 
                    for j in range(num_xjselect):
                        j_l_ind = copy.deepcopy(l_ind)
                        j_u_ind = copy.deepcopy(u_ind)
                        j_labelindex = copy.deepcopy(labelindex)
                        j_unlabelindex = copy.deepcopy(unlabelindex)
                        jmodelOutput = copy.deepcopy(modelOutput)

                        j_sampelindex = np.random.choice(u_ind)
                        j_u_ind = np.delete(j_u_ind, np.where(j_u_ind == j_sampelindex)[0])
                        j_l_ind = np.r_[j_l_ind, j_sampelindex]
                        j_labelindex.append(j_l_ind)
                        j_unlabelindex.append(j_u_ind)

                        model_j = copy.deepcopy(model)
                        model_j.fit(X[j_l_ind], y[j_l_ind].ravel())
                        # model`s predicted values continuous [-1, 1]
                        if modelname in ['RFR', 'DTR', 'ABR']:
                            j_output = model_j.predict(X)
                        else:
                            j_output = (model_j.predict_proba(X)[:, 1] - 0.5) * 2
                        jmodelOutput.append(j_output)
                        # model`s prediction for label -1 or 1
                        j_prediction = np.array([1 if k>0 else -1 for k in j_output])
                        # calulate the designed mate_data Z
                        j_meta_data = mate_data(X, y, distacne, cluster_center_index, j_labelindex, j_unlabelindex, jmodelOutput, j_sampelindex)
                        # calulate the performace improvement
                        j_acc = accuracy_score(y[test], j_prediction[test])
                        j_roc = roc_auc_score(y[test], j_output[test])
                        j_mse = mean_squared_error(y[test], j_prediction[test])
                        j_ll = log_loss(y[test], j_prediction[test])
                        # model improvement on accuracy,auc
                        j_perf_impr = [j_acc - modelPerformance[4][0]]
                        j_perf_impr.append(j_roc - modelPerformance[4][1])
                        # model ratio improvement on mean-squared-error,log-loss
                        j_perf_impr.append((modelPerformance[4][2] - j_mse) / modelPerformance[4][2])
                        j_perf_impr.append((modelPerformance[4][3] - j_ll) / modelPerformance[4][3])
                        j_meta_data = np.c_[j_meta_data, np.array([j_perf_impr])]
                        if metadata is None:
                            metadata = j_meta_data
                        else:
                            metadata = np.vstack((metadata, j_meta_data))
                    
    return metadata
