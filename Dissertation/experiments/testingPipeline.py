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
    
    return clf
    

def gmm(points):
    clf = mixture.GaussianMixture(n_components=6, covariance_type='full')
    pdfs = clf.fit(points).score_samples(points)
        
    return pdfs

def loadGmmByClass(instances, indexesByClass):
    pdfs = [None] * len(instances)
    for c, indexes in indexesByClass.items():
        points = instances[indexes]
        pdfsByClass = gmm(points)
        a = 0
        for i in indexes:
            pdfs[i]=pdfsByClass[a]
            a+=1
        
    return pdfs  

def kde(instances, indexes):
    kde={}
    for c, points in instances.items():
        kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(points)
        kde[c] = kernel.score_samples(points)
    return kde


def baseClassifier(instancesToPredict, classifier):
    return classifier.predict(instancesToPredict)


#Slicing instances according to their inferred clusters
def slicingClusteredData(clusters, classes):
    indexes = {}
    for c in range(numClasses):
        indexes[classes[c]]=[i for i in range(len(clusters)) if clusters[i] == c]
    
    return indexes


#Cutting data for next iteration
def compactingDataDensityBased(instances, densities, criteria):
    maxPDF = max(densities)*criteria
    selectedInstances = [instances[i] for i in range(len(densities)) if densities[i] >= maxPDF]
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


    #Test 0: Predicting 10 instances. Starting labeled data with 5%
    initialDataLength = round((0.001)*len(dataValues))
    U = dataValues.loc[initialDataLength:(initialDataLength+10)].copy()
    U = U.values

    # ***** Box 0 *****
    X = dataValues.loc[:initialDataLength].copy()
    X = X.values
    
    #Starting the process
    for t in range(len(U)):
        print("Step ",t)
        print("Length: ", len(X))
        print("Selected data: ", X)
       
        # ***** Box 1 *****
        Ut = U[t]
        print("Selected unlabeled data: ", Ut)
        classes=[0, 1]

        # ***** Box 2 *****
        kmeans = kMeans(X, classes)
        clusters = kmeans.labels_
        predicted = baseClassifier(Ut, kmeans)

        indexesByClass = slicingClusteredData(np.hstack([clusters, predicted]), classes)
        instances = np.vstack([X, Ut])
        
        # ***** Box 3 *****
        #Testing with two different methods
        pdfGmm = loadGmmByClass(instances, indexesByClass)
        #pdfKde = kde(instances, indexes)
        
        # ***** Box 4 *****
        instancesGMM = compactingDataDensityBased(instances, pdfGmm, 0.8)
        #instancesKDE = compactingDataDensityBased(instances, pdfKde, criteria)
        
        # ***** Box 5 *****
        X = instancesGMM
        #X = instancesKDE
        
        
        
main()
