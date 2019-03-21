import numpy as np 

from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

metadata = np.load('./origin_metadata/australian_lr_metadata.npy')
print(np.shape(metadata))

X = metadata[:, 0:396]
y = metadata[:, 396]

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
# svr_score = []
lr_score = []
sgdr_score = []
rfr_score = []

lor_score = []
# svc_score = []
rfc_score = []

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
        # cross_score = cross_val_score(regressor, new_X, new_y, cv=5, scoring='r2', n_jobs=4) 
        # cross_score = cross_val_score(regressor, new_X, new_y, cv=5, scoring='r2') 
        cross_score = cross_val_score(regressor, new_X, new_y, cv=5, scoring='neg_mean_absolute_error') 
        
        mean_score = np.mean(cross_score) 
        # if isinstance(regressor, SVR):
        #     svr_score.append(mean_score)
        if isinstance(regressor, LinearRegression):
            lr_score.append(mean_score)
        if isinstance(regressor, SGDRegressor):
            sgdr_score.append(mean_score)
        if isinstance(regressor, RandomForestRegressor):
            rfr_score.append(mean_score)        
        print('the r2_score is : ', mean_score)

    process_metadata[0:len(np.where(y>=t)[0]), 396] = 1
    process_metadata[len(np.where(y>=t)[0]):, 396] = -1
    np.random.shuffle(process_metadata)
    for classifier in classifier_list:
        print('############currently classifier model is : ', classifier)
        # cross_score = cross_val_score(classifier, process_metadata[:, 0:396], process_metadata[:, 396], cv=5, scoring='accuracy', n_jobs=4)
        # cross_score = cross_val_score(classifier, process_metadata[:, 0:396], process_metadata[:, 396], cv=5, scoring='accuracy') 
        cross_score = cross_val_score(classifier, process_metadata[:, 0:396], process_metadata[:, 396], cv=5, scoring='precision') 


        mean_accuracy = np.mean(cross_score)
        if isinstance(classifier, LogisticRegression):
            lor_score.append(mean_accuracy)
        # if isinstance(classifier, SVC):
        #     svc_score.append(mean_accuracy)
        if isinstance(classifier, RandomForestClassifier):
            rfc_score.append(mean_accuracy)          
        print('the accuracy is : ', mean_accuracy)       

regressor_scorelist = np.vstack((threshold, lr_score, sgdr_score, rfr_score))
classifier_scorelist = np.vstack((threshold, lor_score, rfc_score))

np.savetxt('./processing_metadata_fitting/regressor_scorelist_mae', regressor_scorelist, delimiter='    ')
np.savetxt('./processing_metadata_fitting/classifier_scorelist_precision', classifier_scorelist, delimiter='    ')
