import numpy as np
from source import classifiers
from source import metrics
from source import util


def kmeans_svm(dataValues, dataLabels, classifier='kmeans', batches = 50, sizeOfBatch = 365, initialLabeledDataPerc=0.05, classes=[0, 1]):
    
    print(">>>>> STARTING TEST with ", classifier, " classifier <<<<<")
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = sizeOfLabeledData
    finalDataLength = sizeOfBatch
    
    X = dataValues.loc[:initialDataLength].copy()
    X = X.values
    y = dataLabels.loc[:initialDataLength].copy()
    y = y.values
    
    arrAcc = []
    
    for t in range(batches):
        #print("Step ",t+1)
        
        U = dataValues.loc[initialDataLength:finalDataLength].copy()
        Ut = U.values
        yt = dataLabels.loc[initialDataLength:finalDataLength].copy()
        yt = yt.values
        predicted=[]
        
        if classifier == 'kmeans':
            kmeans = classifiers.kMeans(classifiers.pca(X, 2), len(classes))
            predicted = util.baseClassifier(classifiers.pca(Ut, 2), kmeans)
        elif classifier == 'svm':
            svmClf = classifiers.svmClassifier(classifiers.pca(X, 2), y)
            predicted = util.baseClassifier(classifiers.pca(Ut, 2), svmClf)
        else:
            return
        
        arrAcc.append(metrics.evaluate(yt, predicted))
        initialDataLength=finalDataLength+1
        finalDataLength+=sizeOfBatch
        # keep a percentage from former distribution to train in next step 
        #X = Ut
        #y = yt
    
    print(metrics.finalEvaluation(arrAcc))
    
    print(">>>>> END OF TEST with ", classifier, " classifier <<<<<")