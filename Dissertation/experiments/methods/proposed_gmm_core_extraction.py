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
    K = kwargs["K"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    excludingPercentage = kwargs["excludingPercentage"]
    useSVM = kwargs["useSVM"]
    isImbalanced=kwargs["isImbalanced"]
    
    clf='Cluster and label'
    if useSVM:
        clf='SVM'
    print("STARTING TEST with {} as classifier and GMM as cutting data".format(clf))    
    
    arrAcc = []
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    #Initial labeled data
    #X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    X = np.copy(dataValues[initialDataLength:finalDataLength])
    y = np.copy(dataLabels[initialDataLength:finalDataLength])
    
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    
    for t in range(batches):
        # ***** Box 2 *****            
        #Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        Ut = np.copy(dataValues[initialDataLength:finalDataLength])
        yt = np.copy(dataLabels[initialDataLength:finalDataLength])
    
        # ***** Box 3 *****
        if useSVM:
            clf = classifiers.svmClassifier(X, y, isImbalanced)
            predicted = util.baseClassifier(Ut, clf)
        else:
            predicted = classifiers.clusterAndLabel(X, y, Ut, K)
            
        allInstances = np.vstack([X, Ut])
        #allLabelsInstances = np.hstack([y, predicted])
        
        # ***** Box 4 *****
        #pdfs from each class applied on entire data
        #pdfsByClass = util.pdfByClass2(allInstances, allLabelsInstances, classes)
        #pdfs from each new points from each class applied on new arrived points
        pdfsByClass = util.pdfByClass2(Ut, predicted, classes)
        
        # ***** Box 5 *****
        #selectedIndexes = util.compactingDataDensityBased(allInstances, pdfsByClass, excludingPercentage)
        selectedIndexes = util.compactingDataDensityBased(Ut, pdfsByClass, excludingPercentage)
        
        # ***** Box 6 *****
        #X, y = util.selectedSlicedData(allInstances, allLabelsInstances, selectedIndexes)
        X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)
        
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted)*100)
 
    # returns accuracy array and last selected points
    return arrAcc, X, y