import matplotlib.pyplot as plt
from source import classifiers
from source import metrics
from source import util


def start(dataValues, dataLabels, **kwargs):
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    classes = kwargs["classes"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    useSVM = kwargs["useSVM"]
    isImbalanced=kwargs["isImbalanced"]
    clf=''
    arrAcc = []
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    
    if useSVM:
        clf = classifiers.svmClassifier(X, y, isImbalanced)
    else:
        clf = classifiers.kMeans(X, len(classes))
    
    for t in range(batches):
        
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        predicted=[]
        predicted = util.baseClassifier(Ut, clf)
        
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        
    #print data distribution in step t
    title = "Data distribution. Step {}".format(t+1)
    plt.scatter(Ut[:,0], Ut[:,1], c="b")
    plt.title(title)
    plt.show()
    
    #print(metrics.finalEvaluation(arrAcc))
    
    return arrAcc