import numpy as np
from source import classifiers
from source import metrics
from source import util


def start(**kwargs):
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
    clfName = kwargs["clfName"]
    
    print("METHOD: Cluster and label / Random Forest as classifier and GMM as core support extraction with cutting data method")

    arrAcc = []
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch

    for t in range(batches):
        # ***** Box 2 *****
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        #if len(Ut) > sizeOfBatch:
        #print(len(Ut))
        # ***** Box 3 *****
        #Adding intersection points, if exists, to train set
        #XIntersec, yIntersec = util.cuttingDataByIntersection3(X, Ut, y)
        #if len(XIntersec)>0:
        #    X, y = np.vstack([X, XIntersec]), np.hstack([y, yIntersec])
        predicted = classifiers.classify(X, y, Ut, K, classes, clfName)   
        #predicted = classifiers.clusterAndLabel(X, y, Ut, K, classes)

        # ***** Box 4 *****
        #pdfs from each new points from each class applied on new arrived points
        pdfsByClass = util.pdfByClass2(Ut, predicted, classes)

        # ***** Box 5 *****
        selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)

        # ***** Box 6 *****
        X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)

        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))

    # returns accuracy array and last selected points
    return arrAcc, X, y