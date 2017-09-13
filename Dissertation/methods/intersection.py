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
    poolSize = kwargs["poolSize"]
    isBatchMode = kwargs["isBatchMode"]

    print("METHOD: K-NN as classifier and {} with Batacharyya distance as core support extraction with cutting data method".format(densityFunction))

    arrAcc = []
    arrX = []
    arrY = []
    arrUt = []
    arrYt = []
    arrClf = []
    keepPercentageByClass = {}
    initialDataLength = 0
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
        #predicted = classifiers.classify(X, y, Ut, K, classes, clfName)   
        clf = classifiers.labelPropagation(X, y, K)
        # for decision boundaries plot
        arrClf.append(clf)
        arrX.append(X)
        arrY.append(y)
        arrUt.append(np.array(Ut))
        arrYt.append(yt)
        predicted = clf.predict(Ut)
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))

        # ***** Box 4 *****
        pdfsByClass = util.pdfByClass(Ut, predicted, classes, densityFunction)
        instancesXByClass, instancesUtByClass = util.unifyInstancesByClass(X, y, Ut, predicted, classes)

        # ***** Box 5 *****
        keepPercentage = util.getBhattacharyyaScores(X, Ut, classes)
        #keepPercentageByClass = util.getBhattacharyyaScoresByClass(instancesXByClass, instancesUtByClass, classes)

        #print("Step: {} Excluding percentage for class 0: {}".format(t, keepPercentageByClass[0]))
        #print("Step: {} Excluding percentage for class 1: {}".format(t, keepPercentageByClass[1]))

        selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, 1-keepPercentage)
        #selectedIndexes = util.compactingDataDensityBased3(pdfsByClass, keepPercentageByClass)

        # ***** Box 6 *****
        X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)

    # returns accuracy array and last selected points
    return "KNN + Bhattacharyya", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf