import numpy as np 

ethn_metadata = np.load('./newmetadata/australian_metadata.npy')
print(np.shape(ethn_metadata))

X = ethn_metadata[:, 0:396]
y = ethn_metadata[:, 396]

print(len(np.where(y >= 0.01)[0]))
print(len(np.where(y <= -0.01)[0]))

new_X = X[np.where(y >= 0.01)[0], :]
new_X = np.vstack((new_X, X[np.where(y <= -0.01)[0], :]))

new_y = y[np.where(y >= 0.01)[0]]
new_y = np.append(new_y, y[np.where(y <= -0.01)[0]])

print(np.shape(new_X))
print(np.shape(new_y))

print(np.shape(np.array([new_y]).T))
process_ethn_metadata = np.hstack((new_X, np.array([new_y]).T))
print(np.shape(process_ethn_metadata))

print(y[np.where(y >= 0.01)[0]][0:10])
print(process_ethn_metadata[0:10, 396])
# print(np.shape(new_y))
# print(y[0:20])
# print(new_y[0:20])

# new_metadata = np.hstack((X, new))
# print(np.shape(new_metadata))
# print(new_metadata[0:20, 396])           
np.save('./newmetadata/process_australian_metadata.npy', process_ethn_metadata)     

process_ethn_metadata[0:len(np.where(y>=0.01)[0]), 396] = 1
process_ethn_metadata[len(np.where(y>=0.01)[0]):, 396] = -1
print(np.unique(process_ethn_metadata[:, 396]))

np.save('./newmetadata/process_classify_australian_metadata.npy', process_ethn_metadata)