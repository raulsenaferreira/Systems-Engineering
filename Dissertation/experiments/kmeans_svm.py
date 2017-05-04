import numpy as np
import matplotlib.pyplot as plt
from source import classifiers
from source import metrics
from source import util
from experiments.composeGMM import box1, box2


def start(dataValues, dataLabels, **kwargs):
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    classes = kwargs["classes"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    classifier='svm'
    isImbalanced=False
    
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = 0
    finalDataLength = sizeOfLabeledData
    arrAcc = []
    isStep1 = True
    
    X, y = box1.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    for t in range(batches):
        #print("Step ",t+1)
        initialDataLength=finalDataLength
        if isStep1:
            finalDataLength=sizeOfBatch
            isStep1 = False
        else:
            finalDataLength+=sizeOfBatch
        Ut, yt = box2.process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        predicted=[]

        if classifier == 'kmeans':
            kmeans = classifiers.kMeans(X, len(classes))
            predicted = util.baseClassifier(Ut, kmeans)
        elif classifier == 'svm':
            svmClf = classifiers.svmClassifier(X, y, isImbalanced)
            predicted = util.baseClassifier(Ut, svmClf)
        else:
            return
        
        arrAcc.append(metrics.evaluate(yt, predicted))
        
    #print data distribution in step t
    title = "Data distribution. Step {}".format(t+1)
    
    plt.scatter(class1[:,0], class1[:,1], c="b")
    plt.title(title)
    plt.show()
        
        # keep a percentage from former distribution to train in next step 
        #X = Ut
        #y = yt
    
    #print(metrics.finalEvaluation(arrAcc))
    
    return arrAcc