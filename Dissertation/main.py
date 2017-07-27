import sys
import os
import matplotlib.pyplot as plt
from source import plotFunctions
from timeit import default_timer as timer
import numpy as np
import setup
from source import metrics
from methods import sliding_svm
from methods import static_svm
from methods import sliding_random_forest
from methods import static_rf
from methods import proposed_gmm_core_extraction
from methods import improved_intersection
from methods import compose
from methods import compose_gmm_version
'''
from experiments.methods import compose3
from experiments.methods import intersection
'''



class Experiment():
    def __init__(self, method):
        #commom for all experiments
        self.method = method
        self.initialLabeledDataPerc=0.05
        #self.classes=[0, 1]
        self.usePCA=False
        #used only by gmm and cluster-label process
        self.densityFunction='gmm'
        self.excludingPercentage = 0.95
        self.K_variation = 5
        self.classifier='cluster_and_label'
        #used in alpha-shape version only
        self.CP=0.65
        self.alpha=0.5
        #used in kmeans_svm and compose only
        self.useSVM=False
        self.isImbalanced=False


def plotBoxplot(data, titles):
    print("Boxplots from all methods")
    fig = plt.figure()
    fig.add_subplot(122)
    plt.boxplot(data, labels=titles)
    plt.xticks(rotation=60)
    plt.show()


def doExperiments(dataValues, dataLabels, datasetDescription, experiments, numberOfTimes, batches):
    listOfAccuracies = []
    listOfMethods = []
    sizeOfBatch = int(len(dataLabels)/batches)
    
    print(datasetDescription)
    print("{} batches of {} instances".format(batches, sizeOfBatch))
    print("\n\n")
    
    for name, e in experiments.items():
        CoreX = []
        CoreY = []
        elapsedTime = []
        accTotal = []
        accuracies=[]
        classes = list(set(dataLabels))#getting all possible classes existent in data
        e.sizeOfBatch = sizeOfBatch
        e.batches = batches
        e.dataLabels = dataLabels
        e.dataValues = dataValues
        e.clfName = 'rf'

        for i in range(numberOfTimes):
            start = timer()
            #accuracy per step
            algorithmName, accuracies, CoreX, CoreY = e.method.start(dataValues=e.dataValues, dataLabels=e.dataLabels, usePCA=e.usePCA, classes=classes, classifier=e.classifier, densityFunction=e.densityFunction, batches=e.batches, sizeOfBatch = e.sizeOfBatch, initialLabeledDataPerc=e.initialLabeledDataPerc, excludingPercentage=e.excludingPercentage, K_variation=e.K_variation, CP=e.CP, alpha=e.alpha, clfName=e.clfName , useSVM=e.useSVM, isImbalanced=e.isImbalanced)
            end = timer()
            averageAccuracy = np.mean(accuracies)

            #elapsed time per step
            elapsedTime.append(end - start)
            
            accTotal.append(averageAccuracy)
        
        listOfAccuracies.append(accuracies)
        listOfMethods.append(algorithmName)
        #print("Total of ", numberOfTimes, " experiment iterations with an average accuracy of ", np.mean(accTotal))
        print("Average execution time: ", np.mean(elapsedTime))
        metrics.finalEvaluation(accuracies, batches)
        #print data distribution in step t
        initial = (batches*sizeOfBatch)-sizeOfBatch
        final = initial + sizeOfBatch
        plotFunctions.plot(dataValues[initial:final], dataLabels[initial:final], CoreX, CoreY, batches)
        print("\n\n")
    #plotFunctions.plotBoxplot(listOfAccuracies)
    plotBoxplot(listOfAccuracies, listOfMethods)


def main():
    experiments = {}
    is_windows = sys.platform.startswith('win')
    sep = '\\'

    if is_windows == False:
        sep = '/'

    path = os.getcwd()+sep+'data'+sep
    #sinthetic
    dataValues, dataLabels, description = setup.loadCDT(path, sep)
    '''dataValues, dataLabels, description = setup.loadCHT(path, sep)
    dataValues, dataLabels, description = setup.load2CDT(path, sep)
    dataValues, dataLabels, description = setup.load2CHT(path, sep)
    dataValues, dataLabels, description = setup.loadUG_2C_2D(path, sep)
    dataValues, dataLabels, description = setup.loadUG_2C_3D(path, sep)
    dataValues, dataLabels, description = setup.loadUG_2C_5D(path, sep)
    dataValues, dataLabels, description = setup.loadMG_2C_2D(path, sep)
    dataValues, dataLabels, description = setup.loadFG_2C_2D(path, sep)
    dataValues, dataLabels, description = setup.loadGEARS_2C_2D(path, sep)
    dataValues, dataLabels, description = setup.loadCSurr(path, sep)
    dataValues, dataLabels, description = setup.load5CVT(path, sep)
    dataValues, dataLabels, description = setup.load4CR(path, sep)
    dataValues, dataLabels, description = setup.load4CRE_V1(path, sep)
    dataValues, dataLabels, description = setup.load4CRE_V2(path, sep)
    dataValues, dataLabels, description = setup.load4CE1CF(path, sep)
    #real
    dataValues, dataLabels, description = setup.loadCheckerBoard(path, sep)
    dataValues, dataLabels, description = setup.loadNOAADataset(path, sep)
    dataValues, dataLabels, description = setup.loadKeystroke(path, sep)
    dataValues, dataLabels, description = setup.loadElecData(path, sep)'''

    '''
    Paper: Core  Support  Extraction  for  Learning  from  Initially  Labeled Nonstationary  Environments  using  COMPOSE
    link: http://s3.amazonaws.com/academia.edu.documents/45784667/2014_-_Core_Support_Extraction_for_Learning_from_Initially_Labeled_NSE_using_COMPOSE_-_IJCNN.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1489296600&Signature=9Z5DQZeDxcCtHUw7445uELSkgBg%3D&response-content-disposition=inline%3B%20filename%3DCore_support_extraction_for_learning_fro.pdf
    '''
    #experiments[0] = Experiment(compose_gmm_version)

    '''
    Original compose (alpha-shape version)
    '''
    #experiments[1] = Experiment(compose)

    '''
    SVM / Random Forest
    '''
    #experiments[2] = Experiment(static_svm)
    #experiments[3] = Experiment(static_rf)

    ''' Proposed Method 1 (GMM core extraction) '''
    #experiments[4] = Experiment(proposed_gmm_core_extraction)

    '''
    Proposed method 2 (Intersection between two distributions + GMM)
    '''
    experiments[5] = Experiment(improved_intersection)
    
    ''' Proposed Method 3 (Alvim) '''
    ##experiments[4] = Experiment(compose3, dataValues, dataLabels, "STARTING TEST with Cluster and label as classifier and GMM / KDE as cutting data")

    '''
    Proposed method 4 (Intersection between two distributions)
    '''
    ##experiments[5] = Experiment(intersection")

    #params: X, y, method, num of experiment repetitions, num of batches
    doExperiments(dataValues, dataLabels, description, experiments, 1, 80)



if __name__ == "__main__":
    main()