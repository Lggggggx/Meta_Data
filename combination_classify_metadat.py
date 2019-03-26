import copy
import numpy as np 

import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from alipy import ToolBox
from alipy.query_strategy.query_labels import QueryInstanceQBC, QueryRandom, QueryInstanceUncertainty

from meta_data import DataSet
from QueryMetaData import QueryMetaData_combination, QueryMetaData, QueryMetaData_classify

dataset_path = './newdata/'

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
 
splitcount = 5

query_num = 40

test_ratio = 0.3

initial_label_ratio = 0.005

savefloder_path = './experiment_result/combination_classify/australian_lrmetadata_0.01/'
# metadata regressior
cd_lr = joblib.load('./processing_metadata_fitting/lr_cdata.joblib')
rfr_meta = joblib.load('./newmetadata/rfr_p_regression_australian.joblib')
rfc_meta = joblib.load('./newmetadata/rfc_p_classify_australian.joblib')
lr_meta = joblib.load('./newmetadata/lr_p_classify_australian.joblib')
# Use the default Logistic Regression classifier
model = LogisticRegression(solver='lbfgs')
# model = RandomForestClassifier()
# model = SVC(gamma='auto')

for testdataset in testdatasetnames:
    print('***********currently dataset is : ', testdataset)

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

    def main_loop(alibox, strategy, round):
        # Get the data split of one fold experiment
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        # Get intermediate results saver for one fold experiment
        saver = alibox.get_stateio(round)

        # To balance such effects that QueryMeta need to select the first five rounds selection
        temp_rand = QueryRandom(X, y)
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        for i in range(5):
            rand_select_ind = temp_rand.select(label_ind, unlab_ind, model=model) 
            label_ind.update(rand_select_ind)
            unlab_ind.difference_update(rand_select_ind)
            model.fit(X=X[label_ind.index, :], y=y[label_ind.index])

        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)
        saver.set_initial_point(accuracy)

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
    
    random_result = []
    unc_result = []
    qbc_result = []

    for round in range(splitcount):
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)

        # Use pre-defined strategy
        random = QueryRandom(X, y)
        unc = QueryInstanceUncertainty(X, y)
        # qbc = QueryInstanceQBC(X, y)

        random_result.append(copy.deepcopy(main_loop(alibox, random, round)))
        unc_result.append(copy.deepcopy(main_loop(alibox, unc, round)))
        # qbc_result.append(copy.deepcopy(main_loop(alibox, qbc, round)))

    # the QueryMetadata we designed
    # turn = 0

    # the combination way
    lr_cdata_result = []
    for round in range(splitcount):

        meta_query = QueryMetaData_combination(X, y, cd_lr)
        # Get the data split of one fold experiment
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        # Get intermediate results saver for one fold experiment
        saver = alibox.get_stateio(round)
        # calc the initial point
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)
        saver.set_initial_point(accuracy)
        # print('the initial point accuracy is {0}'.format(accuracy))

        while not stopping_criterion.is_stop():
            # Select a subset of Uind according to the query strategy
            # Passing model=None to use the default model for evaluating the committees' disagreement
            # select_ind = meta_query.select(label_ind, unlab_ind, model=model)
            # label_ind.update(select_ind)
            # unlab_ind.difference_update(select_ind)

            select_ind, after_select_label_ind, after_select_unlabel_ind = meta_query.select(label_ind, unlab_ind, model)

            # print('the len of after_select_label_ind is {0}'.format(len(after_select_label_ind)))
            # Update model and calc performance according to the model you are using
            model.fit(X=X[after_select_label_ind.index, :], y=y[after_select_label_ind.index])
            pred = model.predict(X[test_idx, :])
            accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                    y_pred=pred,
                                                    performance_metric='accuracy_score')
            # turn +=1
            # print('this is the  {0}`th turn, the accuracy is {1}'.format(turn, accuracy))
            # Save intermediate results to file
            st = alibox.State(select_index=select_ind, performance=accuracy)
            saver.add_state(st)
            saver.save()

            # Passing the current progress to stopping criterion object
            stopping_criterion.update_information(saver)
        # Reset the progress in stopping criterion object
        stopping_criterion.reset()
        lr_cdata_result.append(copy.deepcopy(saver))

    # the regression way
    rfr_regression_result = []
    for round in range(splitcount):

        meta_query = QueryMetaData(X, y, rfr_meta)
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
            select_ind, after_select_label_ind, after_select_unlabel_ind = meta_query.select(label_ind, unlab_ind, model=model)
            # label_ind.update(select_ind)
            # unlab_ind.difference_update(select_ind)

            # Update model and calc performance according to the model you are using
            model.fit(X=X[after_select_label_ind.index, :], y=y[after_select_label_ind.index])
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
    
    # the classify way 
    rfc_classify_result = []
    for round in range(splitcount):

        meta_query = QueryMetaData_classify(X, y, rfc_meta)
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
            select_ind, after_select_label_ind, after_select_unlabel_ind = meta_query.select(label_ind, unlab_ind, model=model)
            # label_ind.update(select_ind)
            # unlab_ind.difference_update(select_ind)

            # Update model and calc performance according to the model you are using
            model.fit(X=X[after_select_label_ind.index, :], y=y[after_select_label_ind.index])
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
        rfc_classify_result.append(copy.deepcopy(saver))

    # the classify way 
    lr_classify_result = []
    for round in range(splitcount):

        meta_query = QueryMetaData_classify(X, y, lr_meta)
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
            select_ind, after_select_label_ind, after_select_unlabel_ind = meta_query.select(label_ind, unlab_ind, model=model)
            # label_ind.update(select_ind)
            # unlab_ind.difference_update(select_ind)

            # Update model and calc performance according to the model you are using
            model.fit(X=X[after_select_label_ind.index, :], y=y[after_select_label_ind.index])
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
        lr_classify_result.append(copy.deepcopy(saver))

    analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
    # analyser.add_method(method_name='QBC', method_results=qbc_result)
    analyser.add_method(method_name='Unc', method_results=unc_result)
    analyser.add_method(method_name='random', method_results=random_result)
    analyser.add_method(method_name='lr_cdata', method_results=lr_cdata_result)
    analyser.add_method(method_name='rfr_regression', method_results=rfr_regression_result)
    analyser.add_method(method_name='rfc_classify', method_results=rfc_classify_result)
    analyser.add_method(method_name='lr_classify', method_results=lr_classify_result)

    analyser.plot_learning_curves(title=testdataset, std_area=False, saving_path=savefloder_path + testdataset +'/', show=False)
