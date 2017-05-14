import numpy as np
from source import util
from experiments.methods import box4, box6
from source import metrics
from source import classifiers



def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    classes = kwargs["classes"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    CP=kwargs["CP"]
    alpha=kwargs["alpha"]
    
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    arrAcc = []
    
    # ***** Box 1 *****
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    #Starting the process
    for t in range(batches):
        #print("Step: ", t)
        # ***** Box 2 *****
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        predicted = classifiers.clusterAndLabel(X, y, Ut)
        instances = np.vstack([X, Ut])
        labelsInstances = np.hstack([y, predicted])
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # ***** Box 4 & Box 5 *****
        threshold = int( len(instances)*(1-CP) )
        selectedPointsByClass, selectedIndexesByClass = box4.geometricCoreExtraction(instances, labelsInstances, classes, alpha, threshold)
        
        # ***** Box 6 *****
        X = np.vstack([selectedPointsByClass[0], selectedPointsByClass[1]])
        y = np.hstack([labelsInstances[selectedIndexesByClass[0]], labelsInstances[selectedIndexesByClass[1]]])
        #X, y = util.selectedSlicedData(selectedPointsByClass, selectedIndexesByClass, labelsInstances)
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch   
    #metrics.finalEvaluation(arrAcc)
    
    return arrAcc, X, y