import numpy as np
from experiments.methods import box3, box4, box5, box6
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
    classifier = kwargs["classifier"]
    K = kwargs["K"]
    densityFunction=kwargs["densityFunction"] 
    useSVM = kwargs["useSVM"]
    isImbalanced=kwargs["isImbalanced"]
    clf=''
    arrAcc = []
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    
    XGMM = X.copy()
    yGMM = y.copy()
    
    for t in range(batches):
        # ***** Box 2 *****            
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        if useSVM:
            clf = classifiers.svmClassifier(X, y, isImbalanced)
            predicted = util.baseClassifier(Ut, clf)
        else:
            predicted = classifiers.clusterAndLabel(XGMM, yGMM, Ut, K)
 
        allInstances, allLabelsInstances = box3.stack(X, Ut, y, predicted)
        
        # ***** Box 4 *****
        previousPdfByClass, currentPdfByClass = box4.pdfByClass(X, y, Ut, predicted, allInstances, classes, densityFunction)
        
        # ***** Box 5 *****
        selectedIndexesOld = box5.cuttingDataByPercentage(X, previousPdfByClass, excludingPercentage)
        selectedIndexesNew = box5.cuttingDataByPercentage(Ut, currentPdfByClass, excludingPercentage)
        selectedIndexes = np.hstack([selectedIndexesOld, selectedIndexesNew])
        
        # ***** Box 6 *****
        XGMM, yGMM = box6.selectedSlicedData(allInstances, allLabelsInstances, selectedIndexes)
        X, y = Ut, predicted
           
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
    
    return arrAcc