import numpy as np
from experiments.methods import box3, box4, box5, box6
from source import metrics


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
    classifier = "svm"
    
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = 0
    finalDataLength = sizeOfLabeledData
    arrAcc = []
    isStep1 = True
    
    # ***** Box 1 *****
    X, y = box1.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    #Starting the process
    for t in range(batches):
        #print("Step: ", t)
        # ***** Box 2 *****
        initialDataLength=finalDataLength
        if isStep1:
            finalDataLength=sizeOfBatch
            XGMM = X.copy()
            yGMM = y.copy()
            isStep1 = False
        else:
            finalDataLength+=sizeOfBatch
            
        Ut, yt = box2.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        predicted = box3.classify(XGMM, yGMM, Ut, K, classifier)#isImbalanced=True when using SVM
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
        #print("Acc: ", metrics.evaluate(yt, predicted))
        arrAcc.append(metrics.evaluate(yt, predicted))
    
    return arrAcc