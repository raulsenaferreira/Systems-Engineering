from source import classifiers
from source import metrics
from source import util


def start(dataValues, dataLabels, **kwargs):
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    usePCA = kwargs["usePCA"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    K = kwargs["K_variation"]
    clfName = kwargs["clfName"]
    classes = kwargs["classes"]

    print("METHOD: Sliding SVM as classifier (UPDATING THROUGH THE TIME)")

    arrAcc = []
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    
    for t in range(batches):
        #print(t)
        #clf = classifiers.svmClassifier(X, y)
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        #predicted = clf.predict(Ut)
        predicted = classifiers.classify(X, y, Ut, K, classes, clfName)  
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        X = Ut
        y = predicted
    
    return "Sliding SVM", arrAcc, X, y