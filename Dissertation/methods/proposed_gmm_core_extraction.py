import numpy as np
from source import classifiers
from source import metrics
from source import util


def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    #initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    initialLabeledData = kwargs["initialLabeledData"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    classes = kwargs["classes"]
    K = kwargs["K_variation"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    excludingPercentage = kwargs["excludingPercentage"]
    clfName = kwargs["clfName"]
    densityFunction = kwargs["densityFunction"]
    
    print("METHOD: K-NN as classifier and {} as core support extraction with cutting data method".format(densityFunction))

    arrAcc = []
    initialDataLength = 0
    excludingPercentage = 1-excludingPercentage
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    for t in range(batches):
        initialDataLength=finalDataLength
        finalDataLength=finalDataLength+sizeOfBatch

        # ***** Box 2 *****
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        
        # ***** Box 3 *****
        predicted = classifiers.classify(X, y, Ut, K, classes, clfName)
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))

        # ***** Box 4 *****
        #pdfs from each new points from each class applied on new arrived points
        pdfsByClass = util.pdfByClass(Ut, predicted, classes, densityFunction)

        # ***** Box 5 *****
        selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)

        # ***** Box 6 *****
        X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)

    # returns accuracy array and last selected points
    return "KNN + Fixed cutting data percentage", arrAcc, X, y