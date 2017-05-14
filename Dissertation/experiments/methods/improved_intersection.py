import numpy as np
from source import metrics
from source import classifiers
from source import util


def start(**kwargs):
    print("Intersection between two distributions + GMM")
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
    useSVM = kwargs["useSVM"]
    isImbalanced=kwargs["isImbalanced"]
    
    def classify(X, y, Ut, K, classifier, isImbalanced):
        if useSVM:
            clf = classifiers.svmClassifier(X, y, isImbalanced)
            return util.baseClassifier(Ut, clf)
        else:
            return classifiers.clusterAndLabel(X, y, Ut, K)
    
    
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = 0
    finalDataLength = sizeOfLabeledData
    arrAcc = []
    
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
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
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        if isStep0:
            isStep0=False
            XIntersec = X.copy()
            yIntersec = y.copy()
            predicted = classify(XIntersec, yIntersec, Ut, K, classifier, isImbalanced)
            
            X = np.vstack([X, Ut])
            y = np.hstack([y, predicted])
        else:
            XIntersec, yIntersec = util.cuttingDataByIntersection3(X, Ut, y)

            if len(yIntersec[yIntersec==0]) <= sizeOfLabeledData or len(yIntersec[yIntersec==1]) <= sizeOfLabeledData:
                allInstances = np.vstack([X, Ut])

                if len(yIntersec) < 1:
                    y=np.array(y)
                    predicted = classify(X, y, Ut, K, classifier, isImbalanced)
                else:
                    predicted = classify(XIntersec, yIntersec, Ut, K, classifier, isImbalanced)

                previousPdfByClass, currentPdfByClass = util.pdfByClass(X, y, Ut, predicted, allInstances, classes, densityFunction)
                #excludingPercentage = 1 - initialLabeledDataPerc

                selectedIndexesOld = util.compactingDataDensityBased(X, previousPdfByClass, excludingPercentage)
                selectedIndexesNew = util.compactingDataDensityBased(Ut, currentPdfByClass, excludingPercentage)
                selectedIndexes = np.hstack([selectedIndexesOld, selectedIndexesNew])
                
                instances = np.vstack([X, Ut])
                labelsInstances = np.hstack([y, predicted])
                newXIntersec, newyIntersec = util.selectedSlicedData(instances, labelsInstances, selectedIndexes)
                #Discard previous intersection
                #XIntersec, yIntersec = newXIntersec, newyIntersec

                #preserve previous intersection
                XIntersec, yIntersec = np.vstack([newXIntersec, XIntersec]), np.hstack([newyIntersec, yIntersec])

                X, y = Ut, predicted
            else:
                predicted = classify(XIntersec, yIntersec, Ut, K, classifier, isImbalanced)
                X, y = Ut, predicted

        # Evaluating classification
        #print("Acc: ", metrics.evaluate(yt, predicted))
        arrAcc.append(metrics.evaluate(yt, predicted))
                
    return arrAcc, XIntersec, yIntersec