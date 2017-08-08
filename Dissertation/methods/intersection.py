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
    #excludingPercentage = kwargs["excludingPercentage"]
    clfName = kwargs["clfName"]
    
    print("METHOD: Random Forest as classifier, GMM as core support extraction and mean of Battacharyya coefficient as excluding percentage")

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
        #print(len(Ut))
        # ***** Box 3 *****
        predicted = classifiers.classify(X, y, Ut, K, classes, clfName)   

        # ***** Box 4 *****
        pdfsByClass = util.pdfByClass(Ut, predicted, classes)
        #allInstancesByClass = util.unifyInstancesByClass(X, y, Ut, predicted, classes)
        instancesXByClass, instancesUtByClass = util.unifyInstancesByClass(X, y, Ut, predicted, classes)

        # ***** Box 5 *****
        keepPercentage = util.getBhattacharyyaScores(instancesUtByClass)
        #selectedIndexes = util.compactingDataScoreBased(scoresByClass, excludingPercentage)
        print("Excluding percentage: ",1-keepPercentage)
        selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, keepPercentage)

        # ***** Box 6 *****
        X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)

        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))

    # returns accuracy array and last selected points
    return "GMM + cutting data percentage", arrAcc, X, y