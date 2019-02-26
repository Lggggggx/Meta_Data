import os
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path)            
        
testdatasetnames=['australian', 'blood', 'breast-cancer-wisc-diag', 'breast-cancer-wisc',
    'chess-krvkp', 'clean1', 'congressional-voting', 'credit-approval']
for name in testdatasetnames:
    mkdir('./classify_experiment_result/'+name+'/')