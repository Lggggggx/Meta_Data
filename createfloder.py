import os
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path)            
        
testdatasetnames =['australian', 'heart-statlog', 'heart', 'house']
for name in testdatasetnames:
    mkdir('./experiment_result/'+name+'/')