import numpy as np
from source import classifiers
from source import metrics
from source import util


def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    classes = kwargs["classes"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    excludingPercentage = kwargs["excludingPercentage"]
    K = kwargs["K_variation"]
    clfName = kwargs["clfName"]
    classifier = kwargs["classifier"]
    densityFunction='gmmBIC'
    distanceMetric = 'mahalanobis'

    print("METHOD: Cluster and label as classifier and GMM with BIC and Mahalanobis as core support extraction")
    
    arrAcc = []
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    
    #Starting the process
    for t in range(batches):
        #print("Step: ", t)
        # ***** Box 2 *****
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        predicted = classifiers.classify(X, y, Ut, K, classes, clfName) 
        instances = np.vstack([X, Ut])
        labelsInstances = np.hstack([y, predicted])
        
        # ***** Box 4 *****
        indexesByClass = util.slicingClusteredData(y, classes)
        bestModelSelectedByClass = util.loadBestModelByClass(X, indexesByClass, classifiers.gmmWithBIC)
        
        # ***** Box 5 *****
        predictedByClass = util.slicingClusteredData(predicted, [0,1])
        selectedIndexes = util.mahalanobisCoreSupportExtraction(Ut, predictedByClass, bestModelSelectedByClass, excludingPercentage)
        selectedIndexes = np.hstack([selectedIndexes[0],selectedIndexes[1]])
        
        # ***** Box 6 *****
        X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)

        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))   
    #metrics.finalEvaluation(arrAcc)
    
    return "GMM + BIC + Mahalanobis", arrAcc, X, y