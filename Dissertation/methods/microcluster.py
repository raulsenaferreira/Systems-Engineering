from source import classifiers
from source import metrics
from source import util
import numpy as np
from scipy.spatial import distance as euclidean_distance
from pprint import pprint
from methods import BIRCH_algorithm as bbb



def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]


def makeAccuracy(arrAllAcc, arrTrueY):
    arrAcc = []
    ini = 0
    end = ini
    for predicted in arrAllAcc:
        predicted = np.asarray(predicted)
        predicted = predicted.flatten()
        batchSize = len(predicted)
        ini=end
        end=end+batchSize

        yt = arrTrueY[ini:end]
        arrAcc.append(metrics.evaluate(yt, predicted))
        
    return arrAcc


def MC_statistics(X):
    N = len(X)
    LS = np.sum(X, axis=0)
    SS = np.sum(X**2, axis=0)
    centroid = np.divide(LS, N)
    radius = np.sqrt((np.sum(X**2)/N - np.sum(X)/N)**2)
    #print("SS: ", np.sum(X**2)/N)
    #print("LS: ", (np.sum(X)/N)**2)
    return centroid, radius


def MC_classification(MC_centroids, MC_labels, Ut):
    #print("MC_labels: ", MC_labels)
    centroid, radius = MC_statistics(Ut)
    best_distance = np.inf
    predicted_label = nearest = None
    
    for i in range(len(MC_centroids)):
        distance = euclidean_distance.euclidean(centroid, MC_centroids[i])
        if distance < best_distance:
            best_distance = distance
            predicted_label = MC_labels[i]
            nearest = MC_centroids[i]
    #print("predicted_label: ", predicted_label)
    #print("best_distance: ", best_distance)
    return predicted_label, best_distance, nearest


def MC_array_construction(X, y):
    MC = []
    classes = list(set(y))

    for i in range(len(X)):
        N = len(X[i])
        LS = np.sum(X[i], axis=0)
        SS = np.sum(X[i]**2, axis=0)
        MC.append((N, LS, SS, y[i]))
    
    return MC


def MC_construction(X, y):
    N = len(X)
    LS = np.sum(X, axis=0)
    SS = np.sum(X**2, axis=0)
    
    return (N, LS, SS, y)


def MC_classifier(MC_array, Ut, threshold):
    best_distance = np.inf
    distances = []
    distancesByClass = {}
    labels = []
    predicted_label = nearest_mc = None
    ind = 0
    #print("len(MC_array): ", len(MC_array))
    for i in range(len(MC_array)):
        #print("MC_array[i][1]: ", MC_array[i][1])
        centroid = np.divide(MC_array[i][1], MC_array[i][0]) # LS/N
        distance = euclidean_distance.euclidean(centroid, Ut)
        distances.append(distance)
        labels.append(MC_array[i][3])
        if distance < best_distance:
            best_distance = distance
            predicted_label = MC_array[i][3] #y
            nearest_mc = MC_array[i]
            ind = i

    radius = np.sqrt(nearest_mc[2]/nearest_mc[0] - ((nearest_mc[1]/nearest_mc[0])**2))
    #print(radius)
    if np.mean(radius) < threshold:
        # incrementality property
        mcUT = MC_construction(Ut, predicted_label)
        newN = nearest_mc[0]+mcUT[0]
        newLS = nearest_mc[1]+mcUT[1]
        newSS = nearest_mc[2]+mcUT[2]
        MC_array[ind] = (newN, newLS, newSS, predicted_label)
        #MC_array.append((newN, newLS, newSS, predicted_label))
    else:
        distancesByClass=[distances[i] for i in range(len(distances)) if labels[i] == predicted_label]
        indexes=[i for i in range(len(distances)) if labels[i] == predicted_label]
        #additivity property
        distancesByClass = np.array(distancesByClass)
        farthest_MCs_distances = (-distancesByClass).argsort()[:2] # two farthest MCs from the predicted class
        
        farthest_MCs_indexes1=indexes[farthest_MCs_distances[0]]
        farthest_MCs_indexes2=indexes[farthest_MCs_distances[1]]
        
        farthest_MC_1 = MC_array[farthest_MCs_indexes1]
        farthest_MC_2 = MC_array[farthest_MCs_indexes2]
        merged_N = farthest_MC_1[0]+farthest_MC_2[0]
        merged_LS = farthest_MC_1[1]+farthest_MC_2[1]
        merged_SS = farthest_MC_1[2]+farthest_MC_2[2]
        merged_y = predicted_label

        del farthest_MC_1
        del farthest_MC_2
        
        MC_array.append((merged_N, merged_LS, merged_SS, merged_y))

    return predicted_label, MC_array


def labeling_microclusters(transformed_X, y, clf):
    
    for i in range(len(transformed_X)):
        ind = (transformed_X[i]).argsort()[:1]
        clf.subcluster_labels_[ind] = y[i]
    
    return clf


def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
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
    poolSize = kwargs["poolSize"]
    isBatchMode = kwargs["isBatchMode"]

    print("METHOD: Microcluster(BIRCH) as classifier")

    arrAcc = []
    threshold = 0.1
    initialDataLength = 0
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    if isBatchMode:
        for t in range(batches):
            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+sizeOfBatch
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
            
            
            #predicted = clf.predict(Ut)
            predicted = clf.predict(Ut)
            #print("predicted: ", predicted)
            #print("real label: ", yt)
            arrAcc.append(metrics.evaluate(yt, predicted))
            #print(predicted)
            clf.partial_fit(Ut)
    else:
        memory = len(X)
        threshold = 1.1
        c = 0
        #y2 = np.copy(y)
        clf = bbb.BIRCH_algorithm(n_clusters=None, branching_factor=memory, threshold=threshold) #parameters from the article
        clf.fit(X)
        for i in range(len(clf.root_.subclusters_)):
            clf.root_.subclusters_[i].label_predicted = y[i]

        remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
        
        for Ut, yt in zip(remainingX, remainingY):
            #c+=1
            #print("Step: ", c)
            
            Ut = Ut.reshape(1, -1)

            mc_original_label = clf.predict(Ut)[0]
            similar_mc = clf.root_.subclusters_[mc_original_label]
            predicted_label = similar_mc.label_predicted
           
            arrAcc.append(predicted_label)
            #print("predicted: ",predicted_label)
            #print("real label: ", yt)
            #print("assert: ", predicted_label==yt)
            #clf.partial_fit() # global custering (fixed number of micro clusters)
            #clf.partial_fit(Ut) # adds new point to tree and makes clustering again
            
            #X = np.vstack([X, Ut])
            #y = np.hstack([y, predicted_label])
            #clf = labeling_microclusters(clf.transform(X), y, clf)

        arrAcc = split_list(arrAcc, batches)
        arrAcc = makeAccuracy(arrAcc, remainingY)
    print("Accuracy: ", np.mean(arrAcc))
    return "MClassification", arrAcc, X, y