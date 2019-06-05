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
testdatasetnames = ['ethn', 'australian', 'wdbc', 'clean1', 'blood', 'breast-cancer-wisc']
 
splitcount = 10

query_num = 100

test_ratio = 0.3

initial_label_ratio = 0.005

savefloder_path = './experiment_result/classical_active_learning/'

model = LogisticRegression(solver='lbfgs')

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


    def main_loop(alibox, strategy, round):
        # Get the data split of one fold experiment
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        # Get intermediate results saver for one fold experiment
        saver = alibox.get_stateio(round)

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
        qbc = QueryInstanceQBC(X, y)

        random_result.append(copy.deepcopy(main_loop(alibox, random, round)))
        unc_result.append(copy.deepcopy(main_loop(alibox, unc, round)))
        qbc_result.append(copy.deepcopy(main_loop(alibox, qbc, round)))



    analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
    analyser.add_method(method_name='QBC', method_results=qbc_result)
    analyser.add_method(method_name='Unc', method_results=unc_result)
    analyser.add_method(method_name='random', method_results=random_result)


    plt = analyser.plot_learning_curves(title=testdataset, std_area=False, saving_path=savefloder_path + testdataset +'/', show=False)
    plt.close()
