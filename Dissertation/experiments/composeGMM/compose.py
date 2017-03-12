import numpy as np
from experiments.composeGMM import box1, box2, box3, box4, box5, box6
from source import metrics


def start(dataValues, dataLabels, densityFunction='gmm', excludingPercentage = 0.2, batches = 50, sizeOfBatch = 365, initialLabeledDataPerc=0.05, classes = [0,1], K = 5):
    
    print(">>>>> STARTING TEST with K-Means, Cluster and label as classifier and ", densityFunction, " as cutting data <<<<<")
    
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = sizeOfLabeledData
    finalDataLength = sizeOfBatch
    arrAcc = []
    
    # ***** Box 1 *****
    X, y = box1.process(dataValues, dataLabels, initialDataLength)
    
    #Starting the process
    for t in range(batches):
        # ***** Box 2 *****
        Ut = box2.process(dataValues, initialDataLength, finalDataLength)

        # ***** Box 3 *****
        predicted = box3.classify(X, y, Ut, K)
        instances, labelsInstances = box3.stack(X, Ut, y, predicted)
        # Evaluating classification
        yt = dataLabels.loc[initialDataLength:finalDataLength].copy()
        yt = yt.values
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # ***** Box 4 *****
        pdfByClass = box4.pdfByClass(instances, labelsInstances, classes, densityFunction)
        
        # ***** Box 5 *****
        selectedIndexes = box5.cuttingDataByPercentage(instances, pdfByClass, excludingPercentage)
        
        
        # ***** Box 6 *****
        X, y = box6.selectedData(instances, labelsInstances, selectedIndexes)
        initialDataLength=finalDataLength+1
        finalDataLength+=sizeOfBatch
           
    metrics.finalEvaluation(arrAcc)
    
    print(">>>>> END OF TEST <<<<<")