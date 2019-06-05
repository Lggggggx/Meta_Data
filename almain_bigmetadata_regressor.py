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
#  'ethn', 'australian', 'wdbc', 'clean1', 'blood', 'breast-cancer-wisc'
testdatasetnames = ['australian', 'wdbc']
 
splitcount = 5

query_num = 50

test_ratio = 0.3    

initial_label_ratio = 0.005

# savefloder_path = './experiment_result/bigmetadata/wdbc_lrmetadata_0.05/split_count_50-300/'

savefloder_path = './experiment_result/auto-sklearn/regressor/australian_lrmetadata_0.005/'
# metadata regressior
# partial_fit_regressor = joblib.load('./bigmetadata/regressor_model/wdbc_origin_regressor.joblib')
# split_count70_sgdr = joblib.load('./wdbc_q-t-50_regressor.joblib')
# meta_regressor = {
#     'origin_sgd': partial_fit_regressor['SGD'],
#     'q_t_50_1model': joblib.load('./bigmetadata/regressor_model/big59wdbc_q-t-50-interval5_regressor.joblib'),
#     'q_t_70_1model': joblib.load('./bigmetadata/regressor_model/wdbc_q-t-110-interval5_regressor.joblib'),
#     'q_t_90_1model': joblib.load('./bigmetadata/regressor_model/wdbc_q-t-200-inter2-regressor.joblib'),
#     'q_t_110_1model': joblib.load('./bigmetadata/regressor_model/wdbc_q-t-300-inter2-regressor.joblib')
# }

# meta_regressor = {
#     'origin_sgd': partial_fit_regressor['SGD'],
#     's_c_50_1model': joblib.load('./bigmetadata/regressor_model/wdbc_s-c-50-model1_regressor.joblib'),
#     's_c_110_1model': joblib.load('./bigmetadata/regressor_model/wdbc_s-c-110-model1_regressor.joblib'),
#     's_c_200_1model': joblib.load('./bigmetadata/regressor_model/wdbc_q-t-200-inter2-regressor.joblib'),
#     's_c_300_1model': joblib.load('./bigmetadata/regressor_model/wdbc_q-t-300-inter2-regressor.joblib')
# }

# meta_regressor = {
#     'auto_regressor': joblib.load('./bigmetadata/automl.joblib')
# }

# cd_lr = joblib.load('./processing_metadata_fitting/lr_cdata.joblib')
# rfr_meta = joblib.load('./newmetadata/rfr_p_regression_australian.joblib')
# rfc_meta = joblib.load('./newmetadata/rfc_p_classify_australian.joblib')
# lr_meta = joblib.load('./newmetadata/lr_p_classify_australian.joblib')
# Use the default Logistic Regression classifier
model = LogisticRegression(solver='lbfgs')
# model = RandomForestClassifier()
# model = SVC(gamma='auto')

for testdataset in testdatasetnames:
    print('***********currently dataset is : ', testdataset)
    # prepare dataset
    dt = DataSet(testdataset, dataset_path)
    X = dt.X
    y = dt.y.ravel()
    y = np.asarray(y, dtype=int)

    alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=savefloder_path + testdataset +'/')
    # Split data
    alibox.split_AL(test_ratio=test_ratio, initial_label_rate=initial_label_ratio, split_count=splitcount)

    # The cost budget is 50 times querying
    stopping_criterion = alibox.get_stopping_criterion('num_of_queries', query_num)

    # generate the first five rounds data(label_index unlabel_index model_output)
    label_index_round = []
    unlabel_index_round = []
    model_output_round = []

    for round in range(splitcount):
        label_inds_5 = []
        unlabel_inds_5 = []
        model_output_5 = []

        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        temp_rand = QueryRandom(X, y)
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        for i in range(5):
            rand_select_ind = temp_rand.select(label_ind, unlab_ind) 
            label_ind.update(rand_select_ind)
            unlab_ind.difference_update(rand_select_ind)
            label_inds_5.append(copy.deepcopy(label_ind))
            unlabel_inds_5.append(copy.deepcopy(unlab_ind))
            model.fit(X=X[label_ind.index, :], y=y[label_ind.index])  
            if hasattr(model, 'predict_proba'):
                output = (model.predict_proba(X)[:, 1] - 0.5) * 2
            else:
                output = model.predict(X)    
            model_output_5.append(output)
        
        label_index_round.append(label_inds_5)
        unlabel_index_round.append(unlabel_inds_5)
        model_output_round.append(model_output_5)

    def main_loop(alibox, strategy, round):
        # Get the data split of one fold experiment
        train_idx, test_idx, label_ind_, unlab_ind_ = alibox.get_split(round)
        # Get intermediate results saver for one fold experiment
        saver = alibox.get_stateio(round)

        label_ind = copy.deepcopy(label_index_round[round][4])
        unlab_ind = copy.deepcopy(unlabel_index_round[round][4])

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

    origin_sgd_result = []
    s_c_50_1model_result = []
    s_c_70_1model_result = []
    s_c_90_1model_result = []
    s_c_110_1model_result = []
    s_c_200_1model_result = []
    s_c_300_1model_result = []


    q_t_50_1model_result = []
    q_t_70_1model_result = []
    q_t_90_1model_result = []
    q_t_110_1model_result = []
    q_t_200_1model_result = []
    q_t_300_1model_result = []

    auto_regressor_result = []


    for name, regressor in meta_regressor.items():
        for round in range(splitcount):
            meta_query = QueryMetaData(X, y, regressor, copy.deepcopy(label_index_round[round]), copy.deepcopy(unlabel_index_round[round]), copy.deepcopy(model_output_round[round]))
            # Get the data split of one fold experiment
            train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
            # Get intermediate results saver for one fold experiment
            saver = alibox.get_stateio(round)
            # calc the initial point
            label_ind = copy.deepcopy(label_index_round[round][4])
            unlab_ind = copy.deepcopy(unlabel_index_round[round][4])
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
            eval(name+'_result').append(copy.deepcopy(saver))

    analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
    # analyser.add_method(method_name='QBC', method_results=qbc_result)
    analyser.add_method(method_name='Unc', method_results=unc_result)
    analyser.add_method(method_name='random', method_results=random_result)
    # analyser.add_method(method_name='lr_cdata_uncertainty', method_results=lr_cdata_unc_result)
    # analyser.add_method(method_name='lr_cdata_random', method_results=lr_cdata_random_result)
    # analyser.add_method(method_name='rfr_regression', method_results=rfr_regression_result)
    # analyser.add_method(method_name='rfc_classify', method_results=rfc_classify_result)
    # analyser.add_method(method_name='lr_classify', method_results=lr_classify_result)
    # analyser.add_method(method_name='bigsgd_regressor', method_results=bigsgd_regressor_result)
    # analyser.add_method(method_name='query_time-50_sgd_p', method_results=split_count70_regressor_result)
    # analyser.add_method(method_name='bigPassive_Aggressive_regressor', method_results=bigPassive_Aggressive_regressor_result)
    for name, re in meta_regressor.items():
        analyser.add_method(method_name=name, method_results=eval(name+'_result'))

    plt = analyser.plot_learning_curves(title=testdataset, std_area=False, saving_path=savefloder_path + testdataset +'/', show=False)
    plt.close()
