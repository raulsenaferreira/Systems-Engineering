import numpy as np
from experiments.composeGMM import box1, box2, box3, box4, box5, box6
from source import metrics


def start(dataValues, dataLabels, usePCA=True, densityFunction='gmm', classifier='cluster_and_label', excludingPercentage = 0.2, batches = 50, sizeOfBatch = 365, initialLabeledDataPerc=0.05, classes = [0,1], K = 5):
    
    print(">>>>> STARTING TEST with ",classifier," as classifier and intersection between two distributions as cutting data <<<<<")
    
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
    print(">>>>> END OF TEST <<<<<")
    
    return np.mean(arrAcc)