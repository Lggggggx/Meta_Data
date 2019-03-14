import os
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path)            
        
testdatasetnames = ['australian', 'blood', 'breast-cancer-wisc-diag', 'breast-cancer-wisc',
    'chess-krvkp', 'clean1', 'congressional-voting', 'credit-approval','cylinder-bands', 
	'diabetes', 'echocardiogram', 'ethn', 'german',
 'heart-hungarian', 'heart-statlog', 'heart', 'hill-valley', 'horse-colic',
 'house-votes', 'house', 'ilpd-indian-liver', 'ionosphere', 'isolet', 'krvskp',
 'liverDisorders', 'mammographic', 'monks-1', 'monks-2', 'monks-3', 'mushroom',
 'oocytes_merluccius_nucleus_4d', 'oocytes_trisopterus_nucleus_2f',
 'optdigits', 'pima', 'ringnorm', 'sat', 'spambase', 'spect', 'spectf',
 'statlog-australian-credit', 'statlog-german-credit', 'statlog-heart',
 'texture', 'tic-tac-toe', 'titanic', 'twonorm', 'vehicle',
 'vertebral-column-2clases', 'wdbc']

for name in testdatasetnames:
    mkdir('./preprocessing_classify_experiment_result-0.03/'+name+'/')