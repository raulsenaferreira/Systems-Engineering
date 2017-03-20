import numpy as np
from experiments.composeGMM import box1, box2, box3, box4, box5, box6
from source import metrics


def start(dataValues, dataLabels, usePCA=True, densityFunction='gmm', classifier='cluster_and_label', excludingPercentage = 0.2, batches = 50, sizeOfBatch = 365, initialLabeledDataPerc=0.05, classes = [0,1], K = 5):
    
    print(">>>>> STARTING TEST with ",classifier," as classifier and ", densityFunction, " as cutting data <<<<<")
    
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
        predicted = box3.classify(X, y, Ut, K, classifier, usePCA)
        instances, labelsInstances = box3.stack(X, Ut, y, predicted)
        # Evaluating classification
        yt = dataLabels.loc[initialDataLength:finalDataLength].copy()
        yt = yt.values
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        if len(X) <= sizeOfLabeledData:
            #Just unifies the two distributions and goes to next time t
            print("step ", t, ": unifying distributions...")
            X, y = instances, labelsInstances
        else:
            #Make the extraction of intersection process
            print("step ", t, ": making intersection process...")
            # ***** Box 4 *****
            #pdfByClass = box4.pdfByClass(instances, labelsInstances, classes, densityFunction)
            #previousPdfByClass = box4.pdfByClass2(X, y, classes, instances, densityFunction)
            #currentPdfByClass = box4.pdfByClass2(Ut, predicted, instances, classes, densityFunction)

            # ***** Box 5 *****
            #selectedIndexes = box5.cuttingDataByPercentage(instances, pdfByClass, excludingPercentage)
            selectedX, selectedY = box5.cuttingDataByIntersection(X, Ut, y, predicted, classes)
            
            # ***** Box 6 *****
            #X, y = box6.selectedSlicedData(instances, labelsInstances, selectedIndexes)
            X = np.vstack([selectedX[0], selectedX[1]])
            y = np.hstack([selectedY[0], selectedY[1]])
            
        initialDataLength=finalDataLength+1
        finalDataLength+=sizeOfBatch
           
    metrics.finalEvaluation(arrAcc)
    
    print(">>>>> END OF TEST <<<<<")