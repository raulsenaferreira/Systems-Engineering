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

    print("METHOD: SCARGC with K-NN")
    '''
    %---dataset = path/name of dataset
    %--numini number of instances of initial labeled data
    %---max_pool_length = num of instances for perform the clustering in pool
    %data
    %example: [vet_bin_acc, acc_final, ~] = SCARGC_1NN('MC-2C-2D.txt', 50, 300, 2)
    %To see the results over time: plot100Steps(vet_bin_acc, '-r')
    function [vet_bin_acc, acc_final, elapsedTime] = SCARGC_1NN(dataset, numini, max_pool_length, nK)
    '''
    arrAcc = []
    initialDataLength = 0
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    initial_labeled_DATA, initial_labeled_LABELS = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    #in the beginning, labeled data are equal initially labeled data
    labeled_DATA = initial_labeled_DATA
    labeled_LABELS = initial_labeled_LABELS

    #unlabeled data used for the test phase
    unlabeled_DATA, unlabeled_LABELS = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
    
    nClass = length(classes)
    
    centroids_ant = []
    tmp_cent = []
    #first centroids
    if K == nClass: #for unimodal case, the initial centroid of each class is the mean of each feature
        for cl in range(1, nClass):
            




    initialDataLength=finalDataLength
    finalDataLength=finalDataLength+sizeOfBatch
    
    clf = classifiers.knn(X, y, K)

    for t in range(batches):
        
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        predicted = clf.predict(Ut)
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        #X = Ut
        #y = predicted
    return "Static KNN", arrAcc, X, y