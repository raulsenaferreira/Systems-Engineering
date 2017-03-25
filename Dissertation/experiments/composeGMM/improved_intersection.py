import numpy as np
from experiments.composeGMM import box1, box2, box3, box4, box5, box6
from source import metrics
from source import classifiers
from source import util


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
    #classifier = "svm"
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = 0
    finalDataLength = sizeOfLabeledData
    arrAcc = []
    experimentType = 2
    
    # ***** Box 1 *****
    X, y = box1.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    isStep0=True
    
    predicted=[]
    
    #Starting the process
    for t in range(batches):
        
            #print("Step: ", t)
            # ***** Box 2 *****
            initialDataLength=finalDataLength
            if isStep0:
                finalDataLength=sizeOfBatch
            else:
                finalDataLength+=sizeOfBatch
            Ut, yt = box2.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

            # ***** Box 3 *****
            if isStep0:
                isStep0=False
                XIntersec = X.copy()
                yIntersec = y.copy()
                predicted = box3.classify(XIntersec, yIntersec, Ut, K, classifier)
                X, y = box3.stack(X, Ut, y, predicted)
            else:
                XIntersec, yIntersec = box5.cuttingDataByIntersection3(X, Ut, y)

                if len(yIntersec[yIntersec==0]) <= sizeOfLabeledData or len(yIntersec[yIntersec==1]) <= sizeOfLabeledData:
                    #print("ELSE IF")
                    allInstances = np.vstack([X, Ut])
                    
                    if len(yIntersec) < 1:
                        #print(y)
                        y=np.array(y)
                        predicted = box3.classify(X, y, Ut, K, classifier)
                    else:
                        predicted = box3.classify(XIntersec, yIntersec, Ut, K, classifier)
                    previousPdfByClass, currentPdfByClass = box4.pdfByClass(X, y, Ut, predicted, allInstances, classes, densityFunction)
                    
                    #excludingPercentage = 1 - initialLabeledDataPerc
                    
                    selectedIndexesOld = box5.cuttingDataByPercentage(X, previousPdfByClass, excludingPercentage)
                    selectedIndexesNew = box5.cuttingDataByPercentage(Ut, currentPdfByClass, excludingPercentage)
                    selectedIndexes = np.hstack([selectedIndexesOld, selectedIndexesNew])
                    instances, labelsInstances = box3.stack(X, Ut, y, predicted)
                    
                    newXIntersec, newyIntersec = box6.selectedSlicedData(instances, labelsInstances, selectedIndexes)
                    #Discard previous intersection
                    #XIntersec, yIntersec = newXIntersec, newyIntersec

                    #preserve previous intersection
                    XIntersec, yIntersec = np.vstack([newXIntersec, XIntersec]), np.hstack([newyIntersec, yIntersec])

                    X, y = Ut, predicted
                else:
                    predicted = box3.classify(XIntersec, yIntersec, Ut, K, classifier)
                    X, y = Ut, predicted

            # Evaluating classification
            #print("Acc: ", metrics.evaluate(yt, predicted))
            arrAcc.append(metrics.evaluate(yt, predicted))
                
    return arrAcc