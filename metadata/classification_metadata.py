import numpy as np 


australian_metadata = np.load('./metadata/australian_metadata.npy')
print(np.shape(australian_metadata))

X = australian_metadata[:, 0:396]
y = australian_metadata[:, 396]

new_y = np.zeros_like(y)
new_y[np.where(y>0)[0]] = 1
new_y[np.where(y<0)[0]] = -1

new = new_y.reshape((-1,1))
print(np.shape(new))

print(np.shape(new_y))
print(y[0:20])
print(new_y[0:20])

new_metadata = np.hstack((X, new))
print(np.shape(new_metadata))
print(new_metadata[0:20, 396])           
np.save('binary_metadata', new_metadata)             