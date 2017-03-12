import numpy as np
from source import classifiers
from source import metrics
from source import util


def kmeans_svm(dataValues, dataLabels, usePCA=True, classifier='kmeans', batches = 50, sizeOfBatch = 365, initialLabeledDataPerc=0.05, classes=[0, 1]):
    
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
        
        if usePCA:
            X = classifiers.pca(X, 2)
            Ut = classifiers.pca(Ut, 2)

        if classifier == 'kmeans':
            kmeans = classifiers.kMeans(X, len(classes))
            predicted = util.baseClassifier(Ut, kmeans)
        elif classifier == 'svm':
            svmClf = classifiers.svmClassifier(X, y)
            predicted = util.baseClassifier(Ut, svmClf)
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