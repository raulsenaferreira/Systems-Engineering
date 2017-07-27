from source import classifiers
from source import metrics
from source import util


def start(dataValues, dataLabels, **kwargs):
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    usePCA = kwargs["usePCA"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]

    print("METHOD: Random Forest as classifier (STATIC)")
    
    arrAcc = []
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    clf = classifiers.randomForest(X, y)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    
    for t in range(batches):
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        predicted = clf.predict(Ut)
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
    
    return "Static RF", arrAcc, X, y