import warnings
# warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import numpy as np 

from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score



metadata = np.load('./simple_metadata/simple_australian_rfc_metadata.npy')
print(np.shape(metadata))

X = metadata[:, 0:95]
y = metadata[:, 95]

print('y max: ', np.max(y))
print('y min: ', np.min(y))
print('y mean: ', np.mean(y))
print('y std: ', np.std(y))

print('The num of postive improvement : ', len(np.where(y >= 0)[0]))
print('The num of negative improvement : ', len(np.where(y <= 0)[0]))

# threshold = np.arange(0.01, 0.2, 0.02)

# for t in threshold:
#     print('The num of lt '+ str(t) +' improvement : ', len(np.where(y <= -t )[0]))
#     print('The num of bt '+ str(t) +' improvement : ', len(np.where(y >= t )[0]))


# regressor
# svr = SVR()
lr = LinearRegression()
sgdr = SGDRegressor()
rfr = RandomForestRegressor()
regressor_list = [lr, sgdr, rfr]

# classifier
lor = LogisticRegression()
# svc = SVC()
rfc = RandomForestClassifier()
classifier_list = [lor, rfc]


threshold = np.arange(0.01, 0.21, 0.01)
# svr_score_r2 = []
lr_score_r2 = []
sgdr_score_r2 = []
rfr_score_r2 = []
# svr_score_mae = []
lr_score_mae = []
sgdr_score_mae = []
rfr_score_mae = []


lor_score_ac = []
# svc_score_ac = []
rfc_score_ac = []

lor_score_pre = []
# svc_score_pre = []
rfc_score_pre = []

for t in threshold:
    print('***********currently threshold is : ', t)
    new_X = X[np.where(y >= t)[0], :]
    new_X = np.vstack((new_X, X[np.where(y <= -t)[0], :]))

    new_y = y[np.where(y >= t)[0]]
    new_y = np.append(new_y, y[np.where(y <= -t)[0]])

    print('The num of ne '+ str(t) +' improvement : ', len(np.where(y <= -t )[0]))
    print('The num of bt '+ str(t) +' improvement : ', len(np.where(y >= t )[0]))
    process_metadata = np.hstack((new_X, np.array([new_y]).T))

    for regressor in regressor_list:
        print('############currently regressor model is : ', regressor)
        cross_score_r2 = cross_val_score(regressor, new_X, new_y, cv=5, scoring='r2', n_jobs=3) 
        # cross_score = cross_val_score(regressor, new_X, new_y, cv=5, scoring='r2') 
        cross_score_mae = cross_val_score(regressor, new_X, new_y, cv=5, scoring='neg_mean_absolute_error', n_jobs=3) 
        
        mean_score_r2 = np.mean(cross_score_r2) 
        mean_score_mae = np.mean(cross_score_mae) 

        # if isinstance(regressor, SVR):
        #     svr_score.append(mean_score)
        if isinstance(regressor, LinearRegression):
            lr_score_r2.append(mean_score_r2)
            lr_score_mae.append(mean_score_mae)
        if isinstance(regressor, SGDRegressor):
            sgdr_score_r2.append(mean_score_r2)
            sgdr_score_mae.append(mean_score_mae)
        if isinstance(regressor, RandomForestRegressor):
            rfr_score_r2.append(mean_score_r2)   
            rfr_score_mae.append(mean_score_mae)     
        print('the r2_score is : ', mean_score_r2)
        print('the mae_score is : ', mean_score_mae)


    process_metadata[0:len(np.where(y>=t)[0]), 95] = 1
    process_metadata[len(np.where(y>=t)[0]):, 95] = -1
    np.random.shuffle(process_metadata)
    for classifier in classifier_list:
        print('############currently classifier model is : ', classifier)
        # cross_score = cross_val_score(classifier, process_metadata[:, 0:396], process_metadata[:, 396], cv=5, scoring='accuracy', n_jobs=4)
        cross_score_ac = cross_val_score(classifier, process_metadata[:, 0:95], process_metadata[:, 95], cv=5, scoring='accuracy', n_jobs=3) 
        cross_score_pre = cross_val_score(classifier, process_metadata[:, 0:95], process_metadata[:, 95], cv=5, scoring='precision', n_jobs=3) 


        mean_accuracy_ac = np.mean(cross_score_ac)
        mean_cross_score_pre = np.mean(cross_score_pre)
        if isinstance(classifier, LogisticRegression):
            lor_score_ac.append(mean_accuracy_ac)
            lor_score_pre.append(mean_cross_score_pre)
        # if isinstance(classifier, SVC):
        #     svc_score.append(mean_accuracy)
        if isinstance(classifier, RandomForestClassifier):
            rfc_score_ac.append(mean_accuracy_ac)  
            rfc_score_pre.append(mean_cross_score_pre)        
        print('the accuracy is : ', mean_accuracy_ac)     
        print('the accuracy is : ', mean_cross_score_pre)       


regressor_scorelist = np.vstack((threshold, lr_score_r2, sgdr_score_r2, rfr_score_r2, np.zeros_like(threshold), lr_score_mae, sgdr_score_mae, rfr_score_mae))
classifier_scorelist = np.vstack((threshold, lor_score_ac, rfc_score_ac, np.zeros_like(threshold), lor_score_pre, rfc_score_pre))

np.savetxt('./processing_metadata_fitting/simple_australian_rfc_regressor_scorelist_mae', regressor_scorelist, delimiter='    ')
np.savetxt('./processing_metadata_fitting/simple_australian_rfc_classifier_scorelist_precision', classifier_scorelist, delimiter='    ')
