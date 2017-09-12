from source import classifiers
from source import metrics
from source import util
import numpy as np

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
            tmp_cent = []
            for atts in range(1, len(initial_labeled_DATA)):
                tmp_cent = [tmp_cent, np.median(initial_labeled_DATA[initial_labeled_LABELS==classes(cl), atts])]
            
            centroids_ant = [centroids_ant; tmp_cent]
        
        centroids_ant = [centroids_ant, classes]
    else #for multimodal case, the initial centroids are estimated by kmeans
        centroids_ant = kmeans(initial_labeled_DATA, nK)
        #associate labels for first centroids
        centroids_ant_lab = []
        for core in range(1, len(centroids_ant,1)):
           pred_lab = knn_classify(initial_labeled_DATA, initial_labeled_LABELS, centroids_ant[core,:]) 
           centroids_ant_lab = [centroids_ant_lab; pred_lab]
        
        centroids_ant = [centroids_ant, centroids_ant_lab]
    cluster_labels = []
    pool_data = []
    vet_bin_acc = []

    updt=0

    for i in range(1, len(unlabeled_LABELS)):
       test_instance = unlabeled_DATA(i,:)
       actual_label = unlabeled_LABELS(i)
       
       #classify each stream's instance with 1NN classifier
       predicted_label = knn_classify(labeled_DATA, labeled_LABELS, test_instance)
  
       pool_data = [pool_data; test_instance, predicted_label]

       if len(pool_data) == max_pool_length:
           #FOR NOAA DATASET, COMMENT NEXT LINE
           centroids_cur = kmeans(pool_data[:,1:end-1], nK, 'start', centroids_ant[end-nK+1:end,1:end-1])        
           #FOR NOAA DATASET, REMOVE THE COMMENT OF THE NEXT LINE
           #[~, centroids_cur] = kmeans(pool_data[:,1:end-1], nK)        
           intermed = []
           cent_labels = []
           for p in range(1, len(centroids_cur)):
               clab,~, nearest = knn_classify(centroids_ant[:,1:end-1], centroids_ant[:,end], centroids_cur[p,:])
               intermed = [intermed; np.median([nearest; centroids_cur[p,:]]), clab]
               cent_labels = [cent_labels; clab]
           
           centroids_cur = [centroids_cur, cent_labels]
           
           #checks if any label is not associated with some cluster
           labelsIntermed = np.unique(intermed[:,end])
           if np.array_equal(labelsIntermed, classes) == False:
               atribuicoes = tabulate(intermed[:,end])
               posMax = np.max(atribuicoes[:,2])
               posMin = np.min(atribuicoes[:,2])
               labelFault = atribuicoes[posMin,1]
               intermed[posMin,end] = labelFault

            centroids_ant = intermed
            new_pool = []
            for p in range(1, len(pool_data)):
                new_pool = [new_pool; knn_classify([centroids_cur[:,1:end-1];centroids_ant[:,1:end-1], [centroids_cur[:,end]; centroids_ant[:,end], pool_data[p,1:end-1]]]
            
            concordant_labels = np.nonzero(pool_data[:,end] == new_pool)

            if len(concordant_labels)/max_pool_length < 1 or len(labeled_LABELS) < len(pool_data)
               pool_data(:,end) = new_pool(:,end);
               centroids_ant = [centroids_cur; intermed];

               labeled_DATA = pool_data(:,1:end-1);
               labeled_LABELS = pool_data(:,end);
               
               %number of updates
               %updt = updt+1
 
           end          

    '''
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
    '''