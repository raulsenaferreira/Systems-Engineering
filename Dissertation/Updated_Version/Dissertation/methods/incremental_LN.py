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

    print("METHOD: Incremental LP as classifier")

    arrAcc = []
    arrX = []
    arrY = []
    arrUt = []
    arrYt = []
    arrClf = []
    arrPredicted = []
    Ut = []
    predicted = []
    initialDataLength = 0
    finalDataLength = initialLabeledData
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    clf = classifiers.labelPropagation(X, y, K)
    for t in range(batches):
        clf.fit(X, y)

        initialDataLength=finalDataLength
        finalDataLength=finalDataLength+sizeOfBatch
                   
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        
        # for decision boundaries plot
        arrClf.append(clf)
        arrX.append(X)
        arrY.append(y)
        arrUt.append(np.array(Ut))
        arrYt.append(yt)
        predicted = clf.predict(Ut)
        arrPredicted.append(predicted)
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
        # incremental
        X, y = np.vstack([X, Ut]), np.hstack([y, predicted])
    
    return "Incremental SSL", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted