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
    
    print("METHOD: Random Forest as classifier and GMM (all instances version)")

    arrAcc = []
    initialDataLength = 0
    excludingPercentage = 1-excludingPercentage
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch

    for t in range(batches):
        # ***** Box 2 *****
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        #print("Step: ", t)
        # ***** Box 3 *****
        predicted = classifiers.classify(X, y, Ut, K, classes, clfName)   

        # ***** Box 4 *****
        #pdfs from each new points from each class applied on new arrived points
        #pdfsByClass = util.pdfByClass(Ut, predicted, classes)

        # ***** Box 5 *****
        #selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)

        # ***** Box 6 *****
        #X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)
        X, y = util.pdfByClass3(X, y, Ut, predicted, classes, excludingPercentage)

        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))

    # returns accuracy array and last selected points
    return "GMM (all instances)", arrAcc, X, y