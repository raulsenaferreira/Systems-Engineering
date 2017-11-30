from source import classifiers
from source import metrics
from source import util
import numpy as np
from scipy.spatial import distance as euclidean_distance
from methods import BIRCH_algorithm as bbb
from scipy.stats import itemfreq
import copy



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
        #SS = np.sum(X[i]**2, axis=0)
        SS = np.dot(LS, LS)
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


def storeLabelsForMC(predicted_by_MC, y):
            mapMCLabels = {}
            for i in range(len(predicted_by_MC)):
                mapMCLabels[predicted_by_MC[i]] = y[i]
            return mapMCLabels


def unionKMoreDissimilar(clf, Ut, predicted_label, K):
    distances = clf.transform(Ut) #matrix with one array containing the distances of the point from MCs
    distances = distances[0] 
    #print(distances)
    distancesByClass=[distances[i] for i in range(len(distances)) if clf.root_.subclusters_[i].label_predicted == predicted_label]
    indexes=[i for i in range(len(distances)) if clf.root_.subclusters_[i].label_predicted == predicted_label]
    
    distancesByClass = np.array(distancesByClass)
    farthest_MCs_distances = (-distancesByClass).argsort()[:K]

    #print("far",distances[indexes[farthest_MCs_distances[0]]])
    #print("far2",distances[indexes[farthest_MCs_distances[1]]])

    farthest_MCs_indexes1=indexes[farthest_MCs_distances[0]]
    farthest_MCs_indexes2=indexes[farthest_MCs_distances[1]]
    
    clf.root_.subclusters_[farthest_MCs_indexes1].update(clf.root_.subclusters_[farthest_MCs_indexes2])
    
    del clf.root_.subclusters_[farthest_MCs_indexes2]
    
    return clf.root_.subclusters_, farthest_MCs_indexes2



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
    arrAcc2 = []
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
        remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
        memory = len(X)
        threshold = 0.5
        c = 0
        #y2 = np.copy(y)
        clf = bbb.BIRCH_algorithm(n_clusters=None, branching_factor=memory, threshold=threshold) #parameters from the article
        clf.fit(X)
        #predicted_by_MC = clf.get_labels()
        #mapMCLabels = storeLabelsForMC(predicted_by_MC, y)
        #print("predicted_by_MC",predicted_by_MC)
        #print("clf.root_.subclusters_",dir(clf.root_.subclusters_))
        for i in range(len(clf.root_.subclusters_)):
            clf.root_.subclusters_[i].label_predicted = y[i]

        #print(clf.root_.init_centroids_)
        
        for Ut, yt in zip(remainingX, remainingY):
            c+=1
            
            Ut = Ut.reshape(1, -1)
            mc_original_label = clf.predict(Ut)[0]
            similar_mc = clf.root_.subclusters_[mc_original_label]
            predicted_label = similar_mc.label_predicted
            #print("centroid e sqnorm antes", similar_mc.centroid_, similar_mc.sq_norm_)
            #is the radius of MC U {mc} < threshold ?
            newMC = copy.deepcopy(similar_mc)
            newMC.n_samples_ = 1
            newMC.linear_sum_ = np.sum(Ut, axis=0)
            newMC.squared_sum_ = np.dot(newMC.linear_sum_, newMC.linear_sum_)
            newMC.label_predicted = predicted_label

            if similar_mc.merge_subcluster(newMC, threshold):
                #print("merged!")
                pass
            else:
                #print("dissimilar")
                # merging the two farthest MCs from the predicted class
                clf.root_.subclusters_, indexToInsert = unionKMoreDissimilar(clf, Ut, predicted_label, 2)
                newMC.centroid_ = newMC.linear_sum_ / newMC.n_samples_
                newMC.sq_norm_ = np.dot(newMC.centroid_, newMC.centroid_)
                clf.root_.subclusters_.insert(indexToInsert, newMC)

            #print("centroid e sqnorm depois", similar_mc.centroid_, similar_mc.sq_norm_)
            #print(clf.root_.subclusters_[mc_original_label].n_samples_)            
            #print(similar_mc.linear_sum_, np.sum(Ut, axis=0), LS)

            arrAcc.append(predicted_label)
            
            #clf.partial_fit(Ut)
            #clf.partial_fit()
            #print(c,len(np.unique(clf.subcluster_labels_)), len(clf.root_.subclusters_))
            #print("predicted", predicted_label)
            #print("assert: ", predicted_label==yt)
            #print("real label: ", yt)

            X = Ut
            y = [predicted_label]
            
        arrAcc = split_list(arrAcc, batches)
        arrAcc = makeAccuracy(arrAcc, remainingY)

    print("Accuracy 1: ", np.mean(arrAcc))
    
    return "MClassification", arrAcc, X, predicted_label