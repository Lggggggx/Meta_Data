import scipy.io as sio
import numpy as np 

dt = sio.loadmat('./australian.mat')
print(sio.whosmat('./australian.mat'))
X = dt['x']
y = dt['y']

print(X[0:2, :])
print(y[0:2])
