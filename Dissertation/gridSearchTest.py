import sys
import os
import setup
from methods import grid_selection_classifier
from sklearn.grid_search import GridSearchCV



def main():
    is_windows = sys.platform.startswith('win')
    sep = '\\'
    
    if is_windows == False:
        sep = '/'

    path = os.getcwd()+sep+'data'+sep
    
    #loading a dataset
    dataValues, dataLabels, description = setup.load2CDT(path, sep)
 
    #testing grid search
    sizes=[200]
    batches=[80]
    tuned_params = {"excludingPercentage" : [0.95, 0.85, 0.75, 0.7, 0.65, 0.6, 0.55], "clfName":['rf'], "sizeOfBatch":[sizes[0]], "batches":[batches[0]]}
    gs = GridSearchCV(grid_selection_classifier.proposed_gmm_core_extraction(tuned_params), tuned_params)
    gs.fit(dataValues, dataLabels)
    print(gs.best_score_)
    print(gs.best_params_)
    
if __name__ == "__main__":
    main()
