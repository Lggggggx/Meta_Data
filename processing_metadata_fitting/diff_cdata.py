import numpy as np 

from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

datanamelist = [ 'australian', 'clean1', 'ethn', 'diabetes', 'breast-cancer-wisc', 'wdbc']


for dataname in datanamelist:

    metadata = np.load('./origin_metadata/australian_rfc_metadata.npy')
    print(np.shape(metadata))

    X = metadata[:, 0:396]
    y = metadata[:, 396]

    # num_positive = len(np.where(y >= 0)[0])
    # num_negative = len(np.where(y <= 0)[0])
    # print('The num of total postive improvement : ', num_positive)
    # print('The num of total negative improvement : ', num_negative)

    y_sort_index = np.argsort(y)

    # percent10_negative_index = y_sort_index[0:int(0.01*num_negative)]
    # percent10_positive_index = y_sort_index[-int(0.01*num_positive):-1]

    # num_10p_neg = len(percent10_negative_index)
    # num_10p_pos = len(percent10_positive_index)
    # print('the num of 10% positive : ', num_10p_pos)
    # print('the num of 10% negative : ', num_10p_neg)

    # print("positive ", y[percent10_positive_index])
    # print("negative ", y[percent10_negative_index])

    # all big than 100
    num_p = 353
    num_n = 108

    positive_1440index = y_sort_index[-num_p:-1]
    negative_600index = y_sort_index[0:num_n]

    num_positive = len(positive_1440index)
    num_negative = len(negative_600index)

    # print(y[positive_1440index[-10:-1]])
    # print(y[negative_600index[0:10]])
    # generate the designed classification data : [x+, x-] is +1, else is -1 [x+, x+], [x-, x+], [x-, x-] are negative
    positive_data = None
    negative_data = None

    for i in np.random.randint(0, num_p-1, 100):
        for j in np.random.randint(0, num_n-1, 100):
            new_1positive_data = np.concatenate((X[positive_1440index[i], 0:396], X[negative_600index[j], 0:396], [+1]))
            if positive_data is None:
                positive_data = new_1positive_data
            else:
                positive_data = np.vstack((positive_data, new_1positive_data))
            
            new_3negative_data = np.concatenate((X[negative_600index[j], 0:396], X[positive_1440index[i], 0:396], [-1]))
            if negative_data is None:
                negative_data = new_3negative_data
            else:
                negative_data = np.vstack((negative_data, new_3negative_data))

        for j in np.random.randint(0, num_p-1, 50):
            new_2negative_data = np.concatenate((X[positive_1440index[i], 0:396], X[positive_1440index[j], 0:396], [-1]))
            negative_data = np.vstack((negative_data, new_2negative_data))

    for i in np.random.randint(0, num_n-1, 50):    
        for j in np.random.randint(0, num_n-1, 50):
            new_4negative_data = np.concatenate((X[negative_600index[i], 0:396], X[negative_600index[j], 0:396], [-1]))
            negative_data = np.vstack((negative_data, new_4negative_data))


    # for i in range(num_positive):
        
    #     for j in range(num_negative):
    #         new_1positive_data = np.concatenate((X[positive_1440index[i], 0:396], X[negative_600index[j], 0:396], [+1]))
    #         if positive_data is None:
    #             positive_data = new_1positive_data
    #         else:
    #             positive_data = np.r_[positive_data, new_1positive_data]
            
    #         new_3negative_data = np.concatenate((X[negative_600index[j], 0:396], X[positive_1440index[i], 0:396], [-1]))
    #         if negative_data is None:
    #             negative_data = new_3negative_data
    #         else:
    #             negative_data = np.r_[negative_data, new_3negative_data]
        
    #     for j in range(num_positive):
    #         new_2negative_data = np.concatenate((X[positive_1440index[i], 0:396], X[positive_1440index[j], 0:396], [-1]))
    #         negative_data = np.r_[negative_data, new_2negative_data]

    # for i in range(num_negative):
    #     for j in range(num_negative):
    #         new_4negative_data = np.concatenate((X[negative_600index[i], 0:396], X[negative_600index[j], 0:396], [-1]))
    #         negative_data = np.r_[negative_data, new_4negative_data]


    print(np.shape(positive_data))
    print(np.shape(negative_data))

    np.save('./processing_metadata_fitting/australian_rfccombination_postive_data.npy', positive_data)
    np.save('./processing_metadata_fitting/australian_rfccombination_negative_data.npy', negative_data)
