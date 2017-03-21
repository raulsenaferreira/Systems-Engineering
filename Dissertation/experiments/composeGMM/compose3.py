import numpy as np
from experiments.composeGMM import box1, box2, box3, box4, box5, box6
from source import metrics


def start(dataValues, dataLabels, usePCA=True, densityFunction='gmm', classifier='cluster_and_label', excludingPercentage = 0.2, batches = 50, sizeOfBatch = 365, initialLabeledDataPerc=0.05, classes = [0,1], K = 5):
    
    print(">>>>> STARTING TEST with ",classifier," as classifier and ", densityFunction, " as cutting data <<<<<")
    
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = 0
    finalDataLength = sizeOfLabeledData
    arrAcc = []
    
    # ***** Box 1 *****
    X, y = box1.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    #Starting the process
    for t in range(batches):
        # ***** Box 2 *****
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        Ut, yt = box2.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        predicted = box3.classify(X, y, Ut, K, classifier, usePCA)
        instances, labelsInstances = box3.stack(X, Ut, y, predicted)
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # ***** Box 4 *****
        box4.intersection(X, y, Ut, predicted, classes, densityFunction)
        pdfByClass = box4.pdfByClass(instances, labelsInstances, classes, densityFunction)
        
        # ***** Box 5 *****
        selectedIndexes = box5.cuttingDataByPercentage(instances, pdfByClass, excludingPercentage)
        
        # ***** Box 6 *****
        X, y = box6.selectedData(instances, labelsInstances, selectedIndexes)
           
    metrics.finalEvaluation(arrAcc)
    
    print(">>>>> END OF TEST <<<<<")