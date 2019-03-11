import os
import copy
import numpy as np 
import scipy.io as sio

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib

from alipy import ToolBox
from alipy.query_strategy.query_labels import QueryInstanceGraphDensity, QueryInstanceQBC, \
    QueryInstanceQUIRE, QueryRandom, QueryInstanceUncertainty, QureyExpectedErrorReduction, QueryInstanceLAL

from meta_data import DataSet, mate_data, model_select, cal_mate_data
from QueryMetaData import QueryMetaData

dataset_path = './newdata/'
# testdatasetnames = np.load('datasetname.npy')
# testdatasetnames = ['australian' 'blood' 'breast-cancer-wisc-diag' 'breast-cancer-wisc'
#  'chess-krvkp' 'clean1' 'congressional-voting' 'credit-approval'
#  'cylinder-bands' 'diabetes' 'echocardiogram' 'ethn' 'german'
#  'heart-hungarian' 'heart-statlog' 'heart' 'hill-valley' 'horse-colic'
#  'house-votes' 'house' 'ilpd-indian-liver' 'ionosphere' 'isolet' 'krvskp'
#  'liverDisorders' 'mammographic' 'monks-1' 'monks-2' 'monks-3' 'mushroom'
#  'oocytes_merluccius_nucleus_4d' 'oocytes_trisopterus_nucleus_2f'
#  'optdigits' 'pima' 'ringnorm' 'sat' 'spambase' 'spect' 'spectf'
#  'statlog-australian-credit' 'statlog-german-credit' 'statlog-heart'
#  'texture' 'tic-tac-toe' 'titanic' 'twonorm' 'vehicle'
#  'vertebral-column-2clases' 'wdbc']
testdatasetnames = ['australian']

# lr_performance = None
sgdr_performance = None
# svr_performance = None
# gbr_performance = None

for testdataset in testdatasetnames:
    print('currently testdataset is : ', testdataset)
    metadata = None
    testmetadata = None
    doc_root = './metadata/'
    for root, dirs, files in os.walk(doc_root):
        for file in files:
            if file == testdataset + '_metadata.npy':
                testmetadata = np.load(doc_root + file)
                continue
            print(file)
            if metadata is None:
                metadata = np.load(doc_root + file)
            else:
                metadata = np.vstack((metadata, np.load(doc_root + file)))
    # metadata = metadata[0:30000]
    # print('The shape of metadata is ', np.shape(metadata))

    # compare the performace of different regressors

    # # LinearRegression
    # lr = LinearRegression(n_jobs=5)
    # lr.fit(metadata[:, 0:396], metadata[:, 396])
    # lr_pred = lr.predict(testmetadata[:, 0:396])
    # lr_mse = mean_squared_error(testmetadata[:, 396], lr_pred)
    # print('In the ' + testdataset + ' LinearRegression mean_squared_error is : ', lr_mse)
    # lr_mae = mean_absolute_error(testmetadata[:, 396], lr_pred)
    # print('In the ' + testdataset + 'LinearRegression mean_absolute_error is : ', lr_mae)
    # lr_r2 = r2_score(testmetadata[:, 396], lr_pred)
    # print('In the ' + testdataset + 'LinearRegression r2_score is : ', lr_r2)
    # if lr_performance is None:
    #     lr_performance = np.array([testdataset, lr_mse, lr_mae, lr_r2])
    # else:
    #     lr_performance = np.vstack((lr_performance, [testdataset, lr_mse, lr_mae, lr_r2]))
    # joblib.dump(lr, testdataset + "meta_lr.joblib")

    # SGDRegressor
    sgdr = SGDRegressor()
    sgdr.fit(metadata[:, 0:396], metadata[:, 396])
    sgdr_pred = sgdr.predict(testmetadata[:, 0:396])
    sgdr_mse = mean_squared_error(testmetadata[:, 396], sgdr_pred)
    print('In the ' + testdataset + 'SGDRegressor mean_squared_error is : ', sgdr_mse)
    sgdr_mae = mean_absolute_error(testmetadata[:, 396], sgdr_pred)
    print('In the ' + testdataset + 'SGDRegressor mean_absolute_error is : ', sgdr_mae)
    sgdr_r2 = r2_score(testmetadata[:, 396], sgdr_pred)
    print('In the ' + testdataset + 'SGDRegressor r2_score is : ', sgdr_r2)
    if sgdr_performance is None:
        sgdr_performance = np.array([testdataset, sgdr_mse, sgdr_mae, sgdr_r2])
    else:
        sgdr_performance = np.vstack((sgdr_performance, [testdataset, sgdr_mse, sgdr_mae, sgdr_r2]))
    joblib.dump(sgdr, testdataset + "meta_sgdr.joblib")

    # # SVR
    # svr = SVR()
    # svr.fit(metadata[:, 0:396], metadata[:, 396])
    # svr_pred = svr.predict(testmetadata[:, 0:396])
    # svr_mse = mean_squared_error(testmetadata[:, 396], svr_pred)
    # print('In the ' + testdataset + 'SVR mean_squared_error is : ', svr_mse)
    # svr_mae = mean_absolute_error(testmetadata[:, 396], svr_pred)
    # print('In the ' + testdataset + 'SVR mean_absolute_error is : ', svr_mae)
    # svr_r2 = r2_score(testmetadata[:, 396], svr_pred)
    # print('In the ' + testdataset + 'SVR r2_score is : ', svr_r2)
    # if svr_performance is None:
    #     svr_performance = np.array([testdataset, svr_mse, svr_mae, svr_r2])
    # else:
    #     svr_performance = np.vstack((svr_performance, [testdataset, svr_mse, svr_mae, svr_r2]))
    # joblib.dump(svr, testdataset + "meta_svr.joblib")

    # # GradientBoostingRegressor
    # gbr = GradientBoostingRegressor()
    # gbr.fit(metadata[:, 0:396], metadata[:, 396])
    # gbr_pred = gbr.predict(testmetadata[:, 0:396])
    # gbr_mse = mean_squared_error(testmetadata[:, 396], gbr_pred)
    # print('In the ' + testdataset + 'GradientBoostingRegressor mean_squared_error is : ', gbr_mse)
    # gbr_mae = mean_absolute_error(testmetadata[:, 396], gbr_pred)
    # print('In the ' + testdataset + 'GradientBoostingRegressor mean_absolute_error is : ', gbr_mae)
    # gbr_r2 = r2_score(testmetadata[:, 396], gbr_pred)
    # print('In the ' + testdataset + 'GradientBoostingRegressor r2_score is : ', gbr_r2)
    # if gbr_performance is None:
    #     gbr_performance = np.array([testdataset, gbr_mse, gbr_mae, gbr_r2])
    # else:
    #     gbr_performance = np.vstack((gbr_performance, [testdataset, gbr_mse, gbr_mae, gbr_r2]))
    # joblib.dump(gbr, testdataset + "meta_gbr.joblib")

    # active learning 
    dt = DataSet(testdataset, dataset_path)
    X = dt.X
    y = dt.y.ravel()
    y = np.asarray(y, dtype=int)

    alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='./experiment_result/')

    # Split data
    alibox.split_AL(test_ratio=0.3, initial_label_rate=0.05, split_count=5)

    # Use the default Logistic Regression classifier
    model = alibox.get_default_model()

    # The cost budget is 50 times querying
    stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 30)

    # experiment
    # meta_regressor = joblib.load('meta_lr.joblib')
    meta_regressor = sgdr
    meta_result = []

    for round in range(5):

        meta_query = QueryMetaData(X, y, meta_regressor)
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
            select_ind = meta_query.select(label_ind, unlab_ind, model=None)
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
        meta_result.append(copy.deepcopy(saver))


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
        eer = QureyExpectedErrorReduction(X, y)
        # random = QueryRandom(X, y)

        unc_result.append(copy.deepcopy(main_loop(alibox, unc, round)))
        qbc_result.append(copy.deepcopy(main_loop(alibox, qbc, round)))
        eer_result.append(copy.deepcopy(main_loop(alibox, eer, round)))
        # random_result.append(copy.deepcopy(main_loop(alibox, random, round)))


    analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
    analyser.add_method(method_name='QBC', method_results=qbc_result)
    analyser.add_method(method_name='Unc', method_results=unc_result)
    analyser.add_method(method_name='EER', method_results=eer_result)
    analyser.add_method(method_name='random', method_results=random_result)
    analyser.add_method(method_name='Meta', method_results=meta_result)

    analyser.plot_learning_curves(title=testdataset, std_area=False, saving_path='./experiment_result/')