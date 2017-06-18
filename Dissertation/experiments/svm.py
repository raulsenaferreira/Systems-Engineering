from source import classifiers
from source import metrics
from source import util


def start(dataValues, dataLabels, **kwargs):
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    
    arrAcc = []
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    
    svmClf = classifiers.svmClassifier(X, y)
    
    for t in range(batches):
        #uncomment if you want update the model for each iteration
        #svmClf = classifiers.svmClassifier(X, y)
        
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        predicted = svmClf.predict(Ut)
        
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        #Updating classifier
        X = Ut
        y = predicted
        
    return arrAcc, Ut, predicted