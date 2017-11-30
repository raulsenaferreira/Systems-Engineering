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

    print("METHOD: Sliding LP as classifier")

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
    clf = classifiers.labelPropagation(X, y, K)
    for t in range(batches):
        # sliding 
        clf.fit(X, y)

        initialDataLength=finalDataLength
        finalDataLength=finalDataLength+sizeOfBatch
        #print(initialDataLength)
        #print(finalDataLength)
                 
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
        
        X, y = Ut, predicted
    
    return "Sliding SSL", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted