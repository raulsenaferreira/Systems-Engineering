from source import classifiers
from source import metrics
from source import util
import numpy as np



def start(**kwargs):
    '''Algorithm flow: X% data are labeled at beginning
    1)Verifies if labeled data is above the minimum (labeldData >= X%). 
        If yes, does the steps 2 up to 4. If not, does step 5 or 6; 
    2)Train with X% of labeled data and classifies the other part of the database (Dt); 
    3)Takes the intersection between two distributions D(t-1) x D(t) = D(t)'; 
    4)D(t)' will be used to train and classify D(t+1) on next phase; 
    5)Takes data from D(t-1)' + X% of greatest pdfs values from GMM(D(t));
    6)Takes the X% greatest from GMM(D(t-1) and GMM(D(t)))'''
    
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    classes = kwargs["classes"]
    K = kwargs["K_variation"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    excludingPercentage = kwargs["excludingPercentage"]
    #densityFunction = kwargs["densityFunction"]
    
    print("STARTING TEST with cluster and label as classifier and GMM + intersection as cutting data")
    
    arrAcc = []
    initialDataLength = 0
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    finalDataLength = sizeOfLabeledData
    # ***** Box 1 *****
    #Initial labeled data
    XIntersec, yIntersec = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    
    for t in range(batches):
        # ***** Box 2 *****
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        predicted = classifiers.clusterAndLabel(XIntersec, yIntersec, Ut, K, classes)
        X = np.vstack([XIntersec, Ut])
        y = np.hstack([yIntersec, predicted])
    
        XIntersec, yIntersec = util.cuttingDataByIntersection3(X, Ut, y)

        if len(yIntersec[yIntersec==0]) <= sizeOfLabeledData or len(yIntersec[yIntersec==1]) <= sizeOfLabeledData:
            if len(yIntersec) < 1:
                y=np.array(y)
                predicted = classifiers.clusterAndLabel(X, y, Ut, K, classes)
            #else:
                #predicted = classifiers.clusterAndLabel(XIntersec, yIntersec, Ut, K, classes)

            pdfsByClass = util.pdfByClass2(Ut, predicted, classes)
            selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)
            newXIntersec, newyIntersec = util.selectedSlicedData(Ut, predicted, selectedIndexes)
            #preserve previous intersection
            XIntersec, yIntersec = np.vstack([newXIntersec, XIntersec]), np.hstack([newyIntersec, yIntersec])
        #else:
            #predicted = classifiers.clusterAndLabel(XIntersec, yIntersec, Ut, K, classes)
        
        X, y = Ut, predicted
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
                
    return "GMM + Intersection", arrAcc, XIntersec, yIntersec