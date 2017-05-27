import sys
import os
from source import plotFunctions
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from experiments import setup
from source import classifiers
from source import metrics
from source import util
from experiments.methods import kmeans_svm
from experiments.methods import proposed_gmm_core_extraction
from experiments.methods import improved_intersection
from experiments.methods import compose
from experiments.methods import raul_classifier
from sklearn.grid_search import GridSearchCV
#from hpsklearn import HyperoptEstimator, any_classifier
#from hyperopt import tpe
'''
from experiments.methods import compose2
from experiments.methods import compose3
from experiments.methods import intersection
'''

def main():
    is_windows = sys.platform.startswith('win')
    sep = '\\'
    
    if is_windows == False:
        sep = '/'

    path = os.getcwd()+sep+'experiments'+sep+'data'+sep
    
    #loading a dataset
    #dataValues, dataLabels = setup.loadNOAADataset(path)
    #dataValues = pd.read_csv(path+'keystroke'+sep+'keystroke.txt',sep = ",")
    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CDT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
 
    '''
    #choose the best classifier with best parameters
    X_train, y_train = util.loadLabeledData(dataValues, dataLabels, 0, 1200, False)
    X_test, y_test = util.loadLabeledData(dataValues, dataLabels, 1200, 1600, False)
    estim = HyperoptEstimator( classifier=any_classifier('clf'),  
                            algo=tpe.suggest, trial_timeout=300)
    
    estim.fit( X_train, y_train )
    
    print( estim.score( X_test, y_test ) )
    # <<show score here>>
    print( estim.best_model() )
    '''
    #testing grid search
    tuned_params = {"excludingPercentage" : [0.9, 0.75, 0.70]}
    gs = GridSearchCV(raul_classifier.raulClassifier(tuned_params), tuned_params)
    gs.fit(dataValues, dataLabels)
    print(gs.best_score_)
    print(gs.best_params_)
    
if __name__ == "__main__":
    main()
