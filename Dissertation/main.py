import sys
import os
from timeit import default_timer as timer
import numpy as np
from experiments import setup
from source import metrics
from experiments.methods import kmeans_svm
from experiments.methods import proposed_gmm_core_extraction
'''
from experiments.methods import compose2
from experiments.methods import compose3
from experiments.methods import intersection
'''
from experiments.methods import compose
'''
from experiments.methods import improved_intersection
'''


class Experiment():
    def __init__(self, method, X, y, d):
        #commom for all experiments
        self.method = method
        self.dataValues = X
        self.dataLabels = y
        self.description = d
        self.batches = 40
        self.sizeOfBatch = 365
        self.initialLabeledDataPerc=0.05
        self.classes=[0, 1]
        self.usePCA=False
        #used only by gmm and cluster-label process
        self.densityFunction='gmm'
        self.excludingPercentage = 0.3
        self.K = 5
        self.classifier='cluster_and_label'
        #used in alpha-shape version only
        self.CP=0.65
        self.alpha=0.5
        #used in kmeans_svm and composeonly
        self.useSVM=False
        self.isImbalanced=False


def doExperiments(experiments, numberOfTimes, sizeOfBatch):
    
    for name, e in experiments.items():
        elapsedTime = []
        accTotal = []
        accuracies=[]
        e.batches = sizeOfBatch
        
        print(e.description)
        
        for i in range(numberOfTimes):
            start = timer()
            #accuracy per step
            accuracies = e.method.start(dataValues=e.dataValues, dataLabels=e.dataLabels, usePCA=e.usePCA, classes=e.classes, classifier=e.classifier, densityFunction=e.densityFunction, batches=e.batches, sizeOfBatch = e.sizeOfBatch, initialLabeledDataPerc=e.initialLabeledDataPerc, excludingPercentage=e.excludingPercentage, K=e.K, CP=e.CP, alpha=e.alpha, useSVM=e.useSVM, isImbalanced=e.isImbalanced)
            end = timer()
            averageAccuracy = np.mean(accuracies)
            
            #elapsed time per step
            elapsedTime.append(end - start)
            
            accTotal.append(averageAccuracy)
        print("Total of ", numberOfTimes, " experiment iterations with an average accuracy of ", np.mean(accTotal))
        print("Average execution time: ", np.mean(elapsedTime))
        metrics.finalEvaluation(accuracies)
        print("\n\n")
        
        

def main():
    experiments = {}
    is_windows = sys.platform.startswith('win')
    sep = '\\'
    
    if is_windows == False:
        sep = '/'

    path = os.getcwd()+sep+'experiments/data'+sep
    
    #loading a dataset
    dataValues, dataLabels = setup.load2CDT(path)
    
    
    '''
    Paper: Core  Support  Extraction  for  Learning  from  Initially  Labeled Nonstationary  Environments  using  COMPOSE
    link: http://s3.amazonaws.com/academia.edu.documents/45784667/2014_-_Core_Support_Extraction_for_Learning_from_Initially_Labeled_NSE_using_COMPOSE_-_IJCNN.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1489296600&Signature=9Z5DQZeDxcCtHUw7445uELSkgBg%3D&response-content-disposition=inline%3B%20filename%3DCore_support_extraction_for_learning_fro.pdf
    '''
    #experiments[0] = Experiment(compose2, dataValues, dataLabels, "STARTING TEST with Cluster and label as classifier and GMM with BIC and Mahalanobis as cutting data")
    
    '''
    Original compose (alpha-shape version)
    '''
    experiments[1] = Experiment(compose, dataValues, dataLabels, "STARTING TEST with Cluster and label as classifier and alpha-shape as cutting data")
    
    '''
    K-Means / SVM
    '''
    #experiments[2] = Experiment(kmeans_svm, dataValues, dataLabels, "STARTING TEST K-Means / SVM alone as classifier")
    
    ''' Proposed Method 1 (GMM core extraction) '''
    #experiments[3] = Experiment(proposed_gmm_core_extraction, dataValues, dataLabels, "STARTING TEST with Cluster and label as classifier and GMM / KDE as cutting data")
    
    ''' Proposed Method 2 (Alvim) '''
    ##experiments[4] = Experiment(compose3, dataValues, dataLabels, "STARTING TEST with Cluster and label as classifier and GMM / KDE as cutting data")

    '''
    Proposed method 3 (Intersection between two distributions)
    '''
    ##experiments[5] = Experiment(intersection, dataValues, dataLabels, "STARTING TEST Cluster and label as classifier and Intersection between two distributions")
    
    '''
    Proposed method 4 (Intersection between two distributions + GMM)
    '''
    #experiments[6] = Experiment(improved_intersection, dataValues, dataLabels, "Improved Intersection")
                                
    doExperiments(experiments, 1, 40)
    

    
if __name__ == "__main__":
    main()