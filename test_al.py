import os
import copy
import numpy as np 
import scipy.io as sio

import pickle

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib

from alipy import ToolBox
from alipy.query_strategy.query_labels import QueryInstanceGraphDensity, QueryInstanceQBC, \
    QueryInstanceQUIRE, QueryRandom, QueryInstanceUncertainty, QureyExpectedErrorReduction, QueryInstanceLAL

from meta_data import DataSet, mate_data, model_select, cal_mate_data
from QueryMetaData import QueryMetaData

dataset_path = './newdata/'
# testdatasetnames = ['pima', 'heart', 'echocardiogram', 'vertebral-column-2clases', 'spect', 'australian', 'oocytes_merluccius_nucleus_4d', 'titanic', 'german', 'chess-krvkp', 'mammographic’,‘ringnorm', 'tic-tac-toe', 'twonorm', 'blood', 'statlog-australian-credit', 'diabetes', 'ionosphere', 'heart-statlog', 'oocytes_trisopterus_nucleus_2f', 'wdbc', 'cylinder-bands’, ‘breast-cancer-wisc-diag', 'clean1', 'isolet', 'credit-approval', 'optdigits', 'mushroom', 'statlog-heart', 'heart-hungarian', 'spambase', 'ethn', 'monks-1’,‘sat', 'monks-3', 'vehicle', 'texture', 'monks-2', 'liverDisorders', 'statlog-german-credit', 'congressional-voting', 'ilpd-indian-liver', 'breast-cancer-wisc', 'spectf', 'krvskp']
testdatasetnames = ['australian', 'blood', 'breast-cancer-wisc-diag', 'breast-cancer-wisc',
    'chess-krvkp', 'clean1', 'congressional-voting', 'credit-approval','heart-statlog']


for testdataset in testdatasetnames:
    print('currently testdataset is : ', testdataset)

    # active learning 
    dt = DataSet(testdataset, dataset_path)
    X = dt.X
    y = dt.y.ravel()
    y = np.asarray(y, dtype=int)

    alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='./experiment_result/'+testdataset+'/')

    # Split data
    alibox.split_AL(test_ratio=0.3, initial_label_rate=0.05, split_count=5)

    # Use the default Logistic Regression classifier
    model = alibox.get_default_model()

    # The cost budget is 50 times querying
    stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 30)

    # random query 
    random = QueryRandom(X, y)
    random_result = []

    for round in range(5):
        # Get the data split of one fold experiment
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        # Get intermediate results saver for one fold experiment
        saver = alibox.get_stateio(round)
        # calc the initial point
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)
        saver.set_initial_point(accuracy)

        while not stopping_criterion.is_stop():
            # Select a subset of Uind according to the query strategy
            # Passing model=None to use the default model for evaluating the committees' disagreement
            select_ind = random.select(unlab_ind)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)

            # Update model and calc performance according to the model you are using
            model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
            pred = model.predict(X[test_idx, :])
            accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                    y_pred=pred,
                                                    performance_metric='accuracy_score')

            # Save intermediate results to file
            st = alibox.State(select_index=select_ind, performance=accuracy)
            saver.add_state(st)
            saver.save()

            # Passing the current progress to stopping criterion object
            stopping_criterion.update_information(saver)
        # Reset the progress in stopping criterion object
        stopping_criterion.reset()
        random_result.append(copy.deepcopy(saver))

    def main_loop(alibox, strategy, round):
        # Get the data split of one fold experiment
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        # Get intermediate results saver for one fold experiment
        saver = alibox.get_stateio(round)
        while not stopping_criterion.is_stop():
            # Select a subset of Uind according to the query strategy
            # Passing model=None to use the default model for evaluating the committees' disagreement
            select_ind = strategy.select(label_ind, unlab_ind, batch_size=1)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)

            # Update model and calc performance according to the model you are using
            model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
            pred = model.predict(X[test_idx, :])
            accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                    y_pred=pred,
                                                    performance_metric='accuracy_score')

            # Save intermediate results to file
            st = alibox.State(select_index=select_ind, performance=accuracy)
            saver.add_state(st)

            # Passing the current progress to stopping criterion object
            stopping_criterion.update_information(saver)
        # Reset the progress in stopping criterion object
        stopping_criterion.reset()
        return saver

    unc_result = []
    qbc_result = []
    eer_result = []

    for round in range(5):
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)

        # Use pre-defined strategy
        unc = QueryInstanceUncertainty(X, y)
        qbc = QueryInstanceQBC(X, y)

        unc_result.append(copy.deepcopy(main_loop(alibox, unc, round)))
        qbc_result.append(copy.deepcopy(main_loop(alibox, qbc, round)))


    analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
    analyser.add_method(method_name='QBC', method_results=qbc_result)
    analyser.add_method(method_name='Unc', method_results=unc_result)
    analyser.add_method(method_name='random', method_results=random_result)

    np.save('./experiment_result/'+testdataset+'/'+'qbc_result.npy', qbc_result)
    np.save('./experiment_result/'+testdataset+'/'+'unc_result.npy', unc_result)
    np.save('./experiment_result/'+testdataset+'/'+'random_result.npy', random_result)

    pickle.dump(qbc_result, open('./experiment_result/'+testdataset+'/'+'qbc_result.pkl', 'wb'), -1)
    pickle.dump(unc_result, open('./experiment_result/'+testdataset+'/'+'unc_result.pkl', 'wb'), -1)
    pickle.dump(random_result, open('./experiment_result/'+testdataset+'/'+'random_result.pkl', 'wb'), -1)


    analyser.plot_learning_curves(title=testdataset, std_area=False, saving_path='./experiment_result/'+testdataset+'/')
