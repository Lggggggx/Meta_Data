import os
import copy
import numpy as np 
import scipy.io as sio

import warnings

from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from alipy import ToolBox
from alipy.query_strategy.query_labels import QueryInstanceGraphDensity, QueryInstanceQBC, \
    QueryInstanceQUIRE, QueryRandom, QueryInstanceUncertainty, QureyExpectedErrorReduction, QueryInstanceLAL

from meta_data import DataSet, mate_data, model_select, cal_mate_data
from QueryMetaData import QueryMetaData, QueryMetaData_classify

dataset_path = './newdata/'

# testdatasetnames=['australian', 'blood', 'breast-cancer-wisc-diag', 'breast-cancer-wisc',
#     'chess-krvkp', 'clean1', 'congressional-voting', 'credit-approval']

# testdatasetnames = ['cylinder-bands', 'diabetes', 'ethn', 'german',
#  'hill-valley', 'horse-colic',
#  'ilpd-indian-liver', 'ionosphere', 'isolet', 'krvskp',
#  'liverDisorders', 'mammographic', 'monks-1', 'monks-2', 'monks-3', 'mushroom',
#  'oocytes_merluccius_nucleus_4d', 'oocytes_trisopterus_nucleus_2f',
#  'optdigits', 'pima', 'ringnorm', 'sat', 'spambase', 'spectf',
#   'statlog-german-credit', 'statlog-heart',
#  'texture', 'tic-tac-toe', 'titanic', 'twonorm', 'vehicle',
#   'wdbc']

testdatasetnames = [ 'australian', 'clean1', 'ethn', 'blood', 'breast-cancer-wisc', 'wdbc']

# testdatasetnames = ['wdbc']
 
splitcount = 5

query_num = 50

test_ratio = 0.3

initial_label_ratio = 0.005

savefloder_path = './n_labelleds_new_australian_regression_exp-0.005/'
# metadata regressior
rfr = joblib.load('./newmetadata/rfr_p_regression_australian.joblib')

# Use the default Logistic Regression classifier
model = LogisticRegression(solver='lbfgs')
# model = RandomForestClassifier()
# model = SVC(gamma='auto')

for testdataset in testdatasetnames:
    print('***********currently dataset is : ', testdataset)
    # metadata = np.load('./metadata/binary_metadata.npy')

    # # compare the performace of different regressors

    # # # LinearRegression
    # print('train rfc')
    # rfc = LogisticRegression()
    # rfc.fit(metadata[:, 0:396], metadata[:, 396])
    # print('done')

    # active learning 
    dt = DataSet(testdataset, dataset_path)
    X = dt.X
    y = dt.y.ravel()
    y = np.asarray(y, dtype=int)

    alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=savefloder_path + testdataset +'/')

    # Split data
    alibox.split_AL(test_ratio=test_ratio, initial_label_rate=initial_label_ratio, split_count=splitcount)



    # The cost budget is 50 times querying
    stopping_criterion = alibox.get_stopping_criterion('num_of_queries', query_num)

    # experiment
    # meta_regressor = joblib.load('meta_lr.joblib')
    # meta_regressor = sgdr
    # meta_result = []

    random = QueryRandom(X, y)
    random_result = []

    for round in range(splitcount):
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
            select_ind = strategy.select(label_ind, unlab_ind, model=model, batch_size=1)
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

    for round in range(splitcount):
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)

        # Use pre-defined strategy
        unc = QueryInstanceUncertainty(X, y)
        qbc = QueryInstanceQBC(X, y)
        eer = QureyExpectedErrorReduction(X, y)
        # random = QueryRandom(X, y)

        unc_result.append(copy.deepcopy(main_loop(alibox, unc, round)))
        qbc_result.append(copy.deepcopy(main_loop(alibox, qbc, round)))
        # eer_result.append(copy.deepcopy(main_loop(alibox, eer, round)))
        # random_result.append(copy.deepcopy(main_loop(alibox, random, round)))


    rfr_regression_result = []
    for round in range(splitcount):

        meta_query = QueryMetaData(X, y, rfr)
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
            select_ind = meta_query.select(label_ind, unlab_ind, model=model)
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
        rfr_regression_result.append(copy.deepcopy(saver))

    analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
    analyser.add_method(method_name='QBC', method_results=qbc_result)
    analyser.add_method(method_name='Unc', method_results=unc_result)
    # analyser.add_method(method_name='EER', method_results=eer_result)
    analyser.add_method(method_name='random', method_results=random_result)
    analyser.add_method(method_name='rfr_regression', method_results=rfr_regression_result)


    analyser.plot_learning_curves(title=testdataset, std_area=False, saving_path=savefloder_path + testdataset +'/')
