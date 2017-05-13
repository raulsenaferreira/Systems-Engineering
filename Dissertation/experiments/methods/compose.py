import numpy as np
from experiments.composeGMM import box1, box2, box3, box4, box5, box6
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
    CP=kwargs["CP"]
    alpha=kwargs["alpha"]
    
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = 0
    finalDataLength = sizeOfLabeledData
    arrAcc = []
    isStep1=True
    
    # ***** Box 1 *****
    X, y = box1.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    #Starting the process
    for t in range(batches):
        #print("Step: ", t)
        # ***** Box 2 *****
        initialDataLength=finalDataLength
        if isStep1:
            finalDataLength=sizeOfBatch
            isStep1 = False
        else:
            finalDataLength+=sizeOfBatch
        Ut, yt = box2.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        predicted = box3.classify(X, y, Ut, K, classifier)
        instances, labelsInstances = box3.stack(X, Ut, y, predicted)
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # ***** Box 4 & Box 5 *****
        threshold = int( len(instances)*(1-CP) )
        selectedPointsByClass, selectedIndexesByClass = box4.geometricCoreExtraction(instances, labelsInstances, classes, alpha, threshold)
        
        # ***** Box 6 *****
        X, y = box6.gettingSelectedData(selectedPointsByClass, selectedIndexesByClass, labelsInstances)
           
    #metrics.finalEvaluation(arrAcc)
    
    return arrAcc