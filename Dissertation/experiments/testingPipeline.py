import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans
from sklearn import svm

def kMeans(X, classes):  
    numClasses = len(classes)
    kmeans = KMeans(n_clusters=numClasses).fit(X)
    
    return kmeans


def svm(X, y):
    clf = svm.SVC()
    clf.fit(X, y)
    
    svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    
    return SVM
    

def gmm(instances):
    gmm = {}
    for c, points in instances.items():
        clf = mixture.GaussianMixture(n_components=6, covariance_type='full')
        clf.fit(points)
        gmm[c] = clf.score_samples(points)
    return gmm


def kde(instances):
    kde={}
    for c, points in instances.items():
        kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(points)
        kde[c] = kernel.score_samples(points)
    return kde


def baseClassifier(instancesToPredict, classifier):
    return classifier.predict(instancesToPredict)


#Slicing instances according to their inferred clusters
def slicingClusteredData(X_train, clusters, classes):
    instances = {}
    for c in range(numClasses):
        instances[classes[c]]=[X_train[i] for i in range(len(clusters)) if clusters[i] == c]
    
    return instances


#Cutting data for next iteration
def compactingDataDensityBased(data, instances, criteria):
    
    return selectedInstances
    
    
def main():
    #current directory
    path = os.getcwd() 

    '''
    Reading NOAA dataset:
    Eight  features  (average temperature, minimum temperature, maximum temperature, dew
    point,  sea  level  pressure,  visibility,  average wind speed, maximum  wind  speed)
    are  used  to  determine  whether  each  day  experienced  rain  or no rain.
    '''
    dataValues = pd.read_csv(path+'\\noaa_data.csv',sep = ",")
    dataLabels = pd.read_csv(path+'\\noaa_label.csv',sep = ",")


    #Test 0: Predicting 10 instances
    unlabeledData = dataValues[initialDataLength:initialDataLength+10]

    # ***** Box 0 *****
    #starting labeled data with 5%
    initialDataLength = round((0.05)*len(dataValues))
    X_train = np.hstack((dataValues[:initialDataLength].as_matrix(), dataLabels[:initialDataLength].as_matrix()))

    #Starting the process
    for t in range(len(unlabeledData)):
        print("Step ",t)
        
        # ***** Box 1 *****
        Ut = unlabeledData[t]
        classes=[0, 1]

        # ***** Box 2 *****
        kmeans = kMeans(X_train, classes)
        clusters = kmeans.labels_
        predicted = baseClassifier(Ut, kmeans)

        instances = slicingClusteredData(X_train, np.vstack([clusters, predicted]), classes)

        # ***** Box 3 *****
        #Testing with two different methods
        gmm = gmm(instances)
        #kde = kde(instances)
        
        # ***** Box 4 *****
        instancesGMM = compactingDataDensityBased(instances, gmm, criteria)
        #instancesKDE = compactingDataDensityBased(instances, kde, criteria)
        
        # ***** Box 5 *****
        X_train = instancesGMM
        #X_train = instancesKDE
        print("Selected data: ", X_train)
        
        
main()
