import numpy as np
from experiments.composeGMM import box1, box2, box3, box4, box5, box6
from source import metrics


def start(**kwargs):
    
    "Algorithm flow: 1)Verifies if X% of labeled data is above the minimum. If yes, does the steps 2 up to 4. If not, does step 5 or 6; 2)Train with X% of labeled data and classifies the other part of the database (Dt); 3)Takes the intersection between two distributions D(t) x D(t+1) = D(t)'; 4)D(t)' will be used to train and classify D(t+1) on next phase; 5)Takes data from D(t-1)' + X% of greatest pdfs values from GMM(D(t)); 6)Takes the X% greatest from GMM(D(t-1) and GMM(D(t)))"
    
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
        
        if len(y) <= sizeOfLabeledData+1:
            #Just unifies the two distributions and goes to next time t
            #print("step ", t, ": unifying distributions...")
            X, y = instances, labelsInstances
        else:
            #Make the extraction of intersection process
            #print("step ", t, ": making intersection process...")
            # ***** Box 4 & Box 5 *****
            selectedX, selectedY = box5.cuttingDataByIntersection(X, Ut, y, np.array(predicted), classes)
            
            # ***** Box 6 *****
            X = np.vstack([selectedX[0], selectedX[1]])
            y = np.hstack([selectedY[0], selectedY[1]])
           
    #metrics.finalEvaluation(arrAcc)
    
    return arrAcc