import numpy as np
from source import classifiers
from source import metrics
from source import util
from experiments.composeGMM import box1, box2


def start(dataValues, dataLabels, usePCA=True, classifier='svm', batches = 50, sizeOfBatch = 365, initialLabeledDataPerc=0.05, classes=[0, 1]):
    
    print(">>>>> STARTING TEST with ", classifier, " classifier <<<<<")
    
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = 0
    finalDataLength = sizeOfLabeledData
    arrAcc = []
    
    X, y = box1.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    for t in range(batches):
        #print("Step ",t+1)
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        Ut, yt = box2.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        predicted=[]

        if classifier == 'kmeans':
            kmeans = classifiers.kMeans(X, len(classes))
            predicted = util.baseClassifier(Ut, kmeans)
        elif classifier == 'svm':
            svmClf = classifiers.svmClassifier(X, y)
            predicted = util.baseClassifier(Ut, svmClf)
        else:
            return
        
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # keep a percentage from former distribution to train in next step 
        #X = Ut
        #y = yt
    
    #print(metrics.finalEvaluation(arrAcc))
    print(">>>>> END OF TEST with ", classifier, " classifier <<<<<")
    
    return np.mean(arrAcc)