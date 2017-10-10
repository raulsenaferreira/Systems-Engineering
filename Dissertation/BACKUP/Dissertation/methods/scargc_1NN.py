from source import classifiers
from source import metrics
from source import util
import numpy as np
from scipy.stats import itemfreq

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

    print("METHOD: SCARGC with 1-NN")
    X,y = [], []
    arrAcc = []
    arrX = []
    arrY = []
    arrUt = []
    arrYt = []
    arrClf = []
    arrPredicted = []
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
    
    nClass = len(classes)
    nK = K
    max_pool_length = sizeOfBatch

    centroids_ant = []
    tmp_cent = []
    
    #first centroids
    if nK == nClass: #for unimodal case, the initial centroid of each class is the mean of each feature
      for cl in range(nClass):
        tmp_cent = np.median(initial_labeled_DATA[initial_labeled_LABELS==classes[cl], 0], axis=0)
        for atts in range(1, initial_labeled_DATA.shape[1]):
          aux = np.median(initial_labeled_DATA[initial_labeled_LABELS==classes[cl], atts], axis=0)
          tmp_cent = np.hstack([tmp_cent, aux])
        
        centroids_ant = np.vstack([centroids_ant, tmp_cent]) if len(centroids_ant) > 0 else tmp_cent

      e = np.array(classes)[np.newaxis]
      centroids_ant = np.hstack([centroids_ant, e.T])
      
    else: #for multimodal case, the initial centroids are estimated by kmeans
      centroids_ant = classifiers.kmeans_matlab(initial_labeled_DATA, nK)
      #associate labels for first centroids
      centroids_ant_lab = []
      
      centroids_ant_lab, a, b = classifiers.knn_classify(initial_labeled_DATA, initial_labeled_LABELS, centroids_ant[0,:])
      
      for core in range(1,centroids_ant.shape[0]):
        pred_lab, a, b = classifiers.knn_classify(initial_labeled_DATA, initial_labeled_LABELS, centroids_ant[core,:])
        centroids_ant_lab = np.vstack([centroids_ant_lab, pred_lab])
      
      centroids_ant = np.hstack([centroids_ant, centroids_ant_lab])

    cluster_labels = []
    pool_data = []
    vet_bin_acc = []

    updt=0

    test_instance = unlabeled_DATA[0,:]
    predicted_label, a, b = classifiers.knn_classify(labeled_DATA, labeled_LABELS, test_instance)
    pool_data = np.hstack([test_instance, predicted_label])

    for i in range(1,len(unlabeled_LABELS)):
      test_instance = unlabeled_DATA[i,:]
      actual_label = unlabeled_LABELS[i]

      #classify each stream's instance with 1NN classifier
      predicted_label, a, b = classifiers.knn_classify(labeled_DATA, labeled_LABELS, test_instance)
      #print("pool: ", pool_data)
      #print("test+predicted: ", np.hstack([test_instance, predicted_label]))
      aux = np.hstack([test_instance, predicted_label])
      pool_data = np.vstack([pool_data, aux]) if len(pool_data) > 0 else aux
      
      if pool_data.shape[0] == max_pool_length:
        #FOR NOAA DATASET, COMMENT NEXT LINE
        print("centroids_ant: ", centroids_ant)
        print("centroid: ",centroids_ant[-nK:,:-1])
        centroids_cur = classifiers.kmeans_matlab(pool_data[:,0:-1], nK, 'start', centroids_ant[-nK:,:-1])         
        #FOR NOAA DATASET, REMOVE THE COMMENT OF THE NEXT LINE
        #[~, centroids_cur] = kmeans(pool_data(:,1:end-1), nK)        

        intermed = []
        cent_labels = []
        
        clab, a, nearest = classifiers.knn_classify(centroids_ant[:,:-1], centroids_ant[:,-1], centroids_cur[0,:])
        #clab, a, nearest = classifiers.knn_scargc(centroids_ant[:,:-1], centroids_ant[:,-1], centroids_cur[0,:])
        
        intermed = np.hstack([np.median(np.vstack([nearest, centroids_cur[0,:]]), axis=0), clab])
        cent_labels = clab
        
        for p in range(1, centroids_cur.shape[0]):
          clab, a, nearest = classifiers.knn_classify(centroids_ant[:,:-1], centroids_ant[:,-1], centroids_cur[p,:])
          #clab, a, nearest = classifiers.knn_scargc(centroids_ant[:,:-1], centroids_ant[:,-1], centroids_cur[p,:])
          aux = np.hstack([np.median(np.vstack([nearest, centroids_cur[p,:]]), axis=0), clab])
          intermed = np.vstack([intermed, aux])
          cent_labels = np.vstack([cent_labels, clab])
         
        centroids_cur = np.hstack([centroids_cur, cent_labels])

        #checks if any label is not associated with some cluster
        labelsIntermed = np.unique(intermed[:,-1])
        
        if all(labelsIntermed == classes) == 0:
          atribuicoes = itemfreq(intermed[:,-1])
          print("atribuicoes: ", atribuicoes)
          posMax = int(np.max(atribuicoes[:,1]))
          posMin = int(np.min(atribuicoes[:,1]))
          labelFault = atribuicoes[posMin,0]
          intermed[posMin,-1] = labelFault               
           
        centroids_ant = intermed
        new_pool = []
        
        #pred, a, b = classifiers.knn_classify(np.vstack([centroids_cur[:,:-1], centroids_ant[:,:-1]]), np.vstack([centroids_cur[:,-1], centroids_ant[:,-1]]), pool_data[0,0:-1])
        pred, a, b = classifiers.knn_scargc(np.vstack([centroids_cur[:,:-1], centroids_ant[:,:-1]]), np.hstack([centroids_cur[:,-1], centroids_ant[:,-1]]), pool_data[0,0:-1])
        new_pool = np.hstack([pool_data[0,0:-1] ,pred])

        for p in range(1, pool_data.shape[0]):
          #pred, a, b = classifiers.knn_classify(np.vstack([centroids_cur[:,:-1], centroids_ant[:,:-1]]), np.vstack([centroids_cur[:,-1], centroids_ant[:,-1]]), pool_data[p,0:-1])
          pred, a, b = classifiers.knn_scargc(np.vstack([centroids_cur[:,:-1], centroids_ant[:,:-1]]), np.hstack([centroids_cur[:,-1], centroids_ant[:,-1]]), pool_data[p,0:-1])
          new_pool = np.vstack([new_pool, np.hstack([pool_data[p,0:-1], pred])])
          
        concordant_labels = np.nonzero(pool_data[:,-1] == new_pool[:,-1])[0]
        
        if len(concordant_labels)/max_pool_length < 1 or len(labeled_LABELS) < pool_data.shape[0]:
          pool_data[:,-1] = new_pool[:,-1]
          centroids_ant = np.vstack([centroids_cur, intermed])
          
          labeled_DATA = pool_data[:,0:-1]
          labeled_LABELS = pool_data[:,-1]
                   
        groundTruth = []
        pool_data = []
     
      #update vet_bin_acc for calculate the accuracy measure
      if predicted_label == actual_label:
        vet_bin_acc = np.hstack([vet_bin_acc, 1])
      else:
        vet_bin_acc = np.hstack([vet_bin_acc, 0])

    acc_final = (np.sum(vet_bin_acc)/len(unlabeled_DATA))*100
    print(acc_final)
    return "SCARGC", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted