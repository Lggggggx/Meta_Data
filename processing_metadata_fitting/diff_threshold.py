import numpy as np 

from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
metadata = np.load('./origin_metadata/australian_lr_metadata.npy')
print(np.shape(metadata))

X = metadata[:, 0:396]
y = metadata[:, 396]

print('The num of postive improvement : ', len(np.where(y >= 0)[0]))
print('The num of negative improvement : ', len(np.where(y <= 0)[0]))

# regressor
svr = SVR()
lr = LinearRegression()
sgdr = SGDRegressor()
rfr = RandomForestRegressor()
regressor_list = [svr, lr, sgdr, rfr]

# classifier
lor = LogisticRegression()
svc = SVC()
rfc = RandomForestClassifier()
classifier_list = [lor, svc, rfc]


threshold = np.arange(0.05, 0.3, 0.02)
test_reressor_score = []
test_classifier_score = []

for t in threshold:
    print('***********currently threshold is : ', t)
    new_X = X[np.where(y >= t)[0], :]
    new_X = np.vstack((new_X, X[np.where(y <= -t)[0], :]))

    new_y = y[np.where(y >= t)[0]]
    new_y = np.append(new_y, y[np.where(y <= -t)[0]])

    process_metadata = np.hstack((new_X, np.array([new_y]).T))

    for regressor in regressor_list:
        print('############currently regressor model is : ', regressor)
        cross_sore = cross_val_score(regressor, new_X, new_y, cv=5, scoring=r2_score, n_jobs=6) 
        print('the r2_score is : ', np.mean(cross_sore))

    process_metadata[0:len(np.where(y>=t)[0]), 396] = 1
    process_metadata[len(np.where(y>=t)[0]):, 396] = -1
    for classifier in classifier_list:
        print('############currently classifier model is : ', classifier)
        cross_sore = cross_val_score(classifier, new_X, process_metadata[:, 396], cv=5, scoring=accuracy_score, n_jobs=6) 
        print('the r2_score is : ', np.mean(cross_sore))       
