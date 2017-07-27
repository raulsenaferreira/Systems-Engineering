import numpy as np
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
    
    for t in range(batches):
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        predicted = classifiers.randomForest(X, y, Ut)
        
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        #Updating classifier    
        X = Ut
        y = predicted
        
    return "Sliding RF", arrAcc, Ut, predicted