from source import classifiers
from source import metrics
from source import util


def start(dataValues, dataLabels, **kwargs):
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

    print("METHOD: K-NN as classifier (STATIC)")

    arrAcc = []
    initialDataLength = 0
    excludingPercentage = 1-excludingPercentage
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=finalDataLength+sizeOfBatch
    
    for t in range(batches):
        clf = classifiers.knn(X, y, K)
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        predicted = clf.predict(Ut)
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        X = Ut
        y = predicted
    return "Static SVM", arrAcc, X, y