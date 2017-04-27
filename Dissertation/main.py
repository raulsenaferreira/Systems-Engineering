import os
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from source import metrics
from experiments import kmeans_svm
from experiments import checkerboard
from experiments.composeGMM import compose
from experiments.composeGMM import compose2
from experiments.composeGMM import compose3
from experiments.composeGMM import intersection
from experiments.composeGMM import original_compose
from experiments.composeGMM import improved_intersection


class Experiment():
    def __init__(self, method, X, y, d):
        #commom for all experiments
        self.method = method
        self.dataValues = X
        self.dataLabels = y
        self.description = d
        self.batches = 40
        self.sizeOfBatch = 365
        self.initialLabeledDataPerc=0.1
        self.classes=[0, 1]
        self.usePCA=True
        #used only by gmm and cluster-label process
        self.densityFunction='gmm'
        self.excludingPercentage = 0.3
        self.K = 5
        self.classifier='cluster_and_label'
        #used in alpha-shape version only
        self.CP=0.65
        self.alpha=0.5


def doExperiments(experiments, numberOfTimes):
    
    for name, e in experiments.items():
        elapsedTime = []
        accTotal = []
        accuracies=[]
        
        print(e.description)
        
        for i in range(numberOfTimes):
            start = timer()
            #accuracy per step
            accuracies = e.method.start(dataValues=e.dataValues, dataLabels=e.dataLabels, usePCA=e.usePCA, classes=e.classes, classifier=e.classifier, densityFunction=e.densityFunction, batches=e.batches, sizeOfBatch = e.sizeOfBatch, initialLabeledDataPerc=e.initialLabeledDataPerc, excludingPercentage=e.excludingPercentage, K=e.K, CP=e.CP, alpha=e.alpha)
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
    path = os.getcwd()+'\\data\\'      
    #Test sets: Predicting 365 instances by step. 50 steps. Two classes.
    '''
    NOAA dataset:
    Eight  features  (average temperature, minimum temperature, maximum temperature, dew
    point,  sea  level  pressure,  visibility,  average wind speed, maximum  wind  speed)
    are  used  to  determine  whether  each  day  experienced  rain  or no rain.
    '''
    dataValues = pd.read_csv(path+'noaa_data.csv',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = pd.read_csv(path+'noaa_label.csv',sep = ",")
    dataLabels = pd.DataFrame.as_matrix(dataLabels)

    

    #Test sets: Predicting N instances by step. T steps. Two classes.
    '''
    Rotated checkerboard dataset. Rotating 2*PI
    '''
    T = 50 #time steps
    N = 365 #instances
    a = np.linspace(0,2*np.pi,T)
    side = 0.25
    
    auxV, auxL = checkerboard.generateData(side, a, N, T)
    #print(dV[2][0])#tempo 2 da classe 0
    dataLabels = auxL[0]
    dataValues = auxV[0]
    
    for i in range(1, T):
        dataLabels = np.hstack([dataLabels, auxL[i]])
        dataValues = np.vstack([dataValues, auxV[i]])
    
    
    
    #Test set: 
    '''
    Artificial One Class Diagonal Translation. 2 Dimensional data
    '''
    dataValues = pd.read_csv(path+'1CDT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    
    
    experiments = {}
    
    '''
    Paper: Core  Support  Extraction  for  Learning  from  Initially  Labeled Nonstationary  Environments  using  COMPOSE
    link: http://s3.amazonaws.com/academia.edu.documents/45784667/2014_-_Core_Support_Extraction_for_Learning_from_Initially_Labeled_NSE_using_COMPOSE_-_IJCNN.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1489296600&Signature=9Z5DQZeDxcCtHUw7445uELSkgBg%3D&response-content-disposition=inline%3B%20filename%3DCore_support_extraction_for_learning_fro.pdf
    '''
    #experiments[0] = Experiment(compose2, dataValues, dataLabels, "STARTING TEST with Cluster and label as classifier and GMM with BIC and Mahalanobis as cutting data")
    
    '''
    Original compose (alpha-shape version)
    '''
    #experiments[1] = Experiment(original_compose, dataValues, dataLabels, "STARTING TEST with Cluster and label as classifier and alpha-shape as cutting data")
    
    '''
    K-Means / SVM
    '''
    experiments[2] = Experiment(kmeans_svm, dataValues, dataLabels, "STARTING TEST K-Means / SVM alone as classifier")
    
    ''' Proposed Method 1 (GMM) '''
    #experiments[3] = Experiment(compose, dataValues, dataLabels, "STARTING TEST with Cluster and label as classifier and GMM / KDE as cutting data")
    
    ''' Proposed Method 2 (Alvim) '''
    ##experiments[4] = Experiment(compose3, dataValues, dataLabels, "STARTING TEST with Cluster and label as classifier and GMM / KDE as cutting data")

    '''
    Proposed method 3 (Intersection between two distributions)
    '''
    ##experiments[5] = Experiment(intersection, dataValues, dataLabels, "STARTING TEST Cluster and label as classifier and Intersection between two distributions")
    
    '''
    Proposed method 4 (Intersection between two distributions + GMM)
    '''
    experiments[6] = Experiment(improved_intersection, dataValues, dataLabels, "Improved Intersection")
                                
    doExperiments(experiments, 2)

    
if __name__ == "__main__":
    main()