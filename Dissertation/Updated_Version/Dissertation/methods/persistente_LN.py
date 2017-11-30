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

    print("METHOD: Persistent classifier")

    arrAcc = []
    arrX = []
    arrY = []
    arrUt = []
    arrYt = []
    arrClf = []
    arrPredicted = []
    initialDataLength = 0
    finalDataLength = initialLabeledData
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    lenLabels = len(y)
    inst = []
    labels = []
    clf = classifiers.labelPropagation(X, y, K)
    remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
    
    for i in range(len(remainingX)):
        clf.fit(X, y)
        initialDataLength=finalDataLength
        finalDataLength=finalDataLength+sizeOfBatch
        
        Ut, yt = remainingX[i], remainingY[i]
        predicted = clf.predict(Ut.reshape(1, -1))

        # for decision boundaries plot
        arrClf.append(clf)
        arrX.append(X)
        arrY.append(y)
        arrUt.append(np.array(Ut))
        arrYt.append(yt)
        arrPredicted.append(predicted)
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # persistent y^t = y(t-1)
        i+=1
        arrClf.append(clf)
        arrX.append(Ut)
        arrY.append(predicted)
        Ut, yt = remainingX[i], remainingY[i]
        arrUt.append(np.array(Ut))
        arrYt.append(yt)
        arrPredicted.append(predicted)
        # Evaluating the persistence
        arrAcc.append(metrics.evaluate(yt, predicted))
    
    return "Persistent SSL", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted