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
        #print("Step: ", t)
        # ***** Box 2 *****
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        Ut, yt = box2.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        predicted = box3.classify(X, y, Ut, K, classifier)
        instances, labelsInstances = box3.stack(X, Ut, y, predicted)
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # ***** Box 4 *****
        previousPdfByClass, currentPdfByClass = box4.pdfByClass(X, y, Ut, predicted, instances, classes, densityFunction)
        
        # ***** Box 5 *****
        selectedIndexesOld = box5.cuttingDataByPercentage(instances, previousPdfByClass, excludingPercentage)
        selectedIndexesNew = box5.cuttingDataByPercentage(instances, currentPdfByClass, excludingPercentage)
        selectedIndexes = np.hstack([selectedIndexesOld, selectedIndexesNew])
        
        # ***** Box 6 *****
        X, y = box6.selectedSlicedData(instances, labelsInstances, selectedIndexes)
           
    metrics.finalEvaluation(arrAcc)
    
    print(">>>>> END OF TEST <<<<<")