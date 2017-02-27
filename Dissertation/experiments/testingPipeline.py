import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.decomposition import PCA


def pca(X, numComponents):
    pca = PCA(n_components=numComponents)
    pca.fit(X)
    PCA(copy=True, iterated_power='auto', n_components=numComponents, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
    
    return pca.transform(X)
       
    
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
    pdfs = np.exp(clf.fit(points).score_samples(points))
        
    return pdfs


def kde(points):
    kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(points)
    pdfs = np.exp(kernel.score_samples(points))
    
    return pdfs


def baseClassifier(instancesToPredict, classifier):
    return classifier.predict(instancesToPredict)


def initializingData(X, y):
    c1=[]
    c2=[]
    for i in range(len(y)):
        if y[i]==0:
            c1.append(X[i])
        else:
            c2.append(X[i])
    
    return c1, c2
    
    
def loadDensitiesByClass(instances, indexesByClass, densityFunction):
    pdfsByClass = {}
    for c, indexes in indexesByClass.items():
        pdfs = [-1] * len(instances)
        points = instances[indexes]
        pdfsByPoints = densityFunction(points)
        a = 0
        for i in indexes:
            pdfs[i]=pdfsByPoints[a]
            a+=1
        pdfsByClass[c] = pdfs
        
    return pdfsByClass


#Slicing instances according to their inferred clusters
def slicingClusteredData(clusters, classes):
    indexes = {}
    for c in range(len(classes)):
        indexes[classes[c]]=[i for i in range(len(clusters)) if clusters[i] == c]
    
    return indexes


#Cutting data for next iteration
def compactingDataDensityBased(instances, densities, criteria):
    selectedInstances=[]
    
    for k in densities:
        arrPdf = densities[k]
        maxPDF = max(arrPdf)*criteria
        minPDF = min(arrPdf)*criteria
        selectedInstances.append([instances[i] for i in range(len(arrPdf)) if arrPdf[i] != -1 and (arrPdf[i] >= maxPDF or arrPdf[i] <= minPDF)])
    return selectedInstances
    

def plotDistributions(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['cluster 1', 'cluster 2']
    ax = fig.add_subplot(121)
    
    for X in distributions:
        #reducing to 2-dimensional data
        x=pca(X, 2)
        
        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1
    
    ax.legend(handles, classes)
    
    plt.show()
    
    
def plotDistributionByClass(instances, indexesByClass):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['cluster 1', 'cluster 2']
    ax = fig.add_subplot(121)
    
    for c, indexes in indexesByClass.items():
        X = instances[indexes]
        #reducing to 2-dimensional data
        x=pca(X, 2)
        
        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1
    
    ax.legend(handles, classes)
    
    plt.show()
    
    
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

    ''' Test 0: 
    Predicting 365 instances by step. 50 steps. Starting labeled data with 5%. Two classes.
    '''
    excludingPercentage = 0.5
    batches = 50
    sizeOfBatch = 365
    sizeOfLabeledData = round((0.05)*sizeOfBatch)
    initialDataLength = sizeOfLabeledData
    finalDataLength = sizeOfBatch
    
    # ***** Box 0 *****
    X = dataValues.loc[:initialDataLength].copy()
    X = X.values
    y = dataLabels.loc[:initialDataLength].copy()
    y = y.values
    X_class1, X_class2 = initializingData(X, y)
    
    #Starting the process
    for t in range(batches):
        print("Step ",t+1)
        
        # ***** Box 1 *****
        X = np.vstack([X_class1, X_class2])
        U = dataValues.loc[initialDataLength:finalDataLength].copy()
        Ut = U.values
        #Ut = [U[t], U[t+1]]
        #print("Selected unlabeled data: ", Ut)
        classes=[0, 1]

        # ***** Box 2 *****
        kmeans = kMeans(pca(X, 2), classes)
        clusters = kmeans.labels_
        predicted = baseClassifier(pca(Ut, 2), kmeans)
        instances = np.vstack([X, Ut])
        indexesByClass = slicingClusteredData(np.hstack([clusters, predicted]), classes)
        #Ploting some info
        print(len(instances), " Points")
        plotDistributions([X_class1, X_class2])
        
        # ***** Box 3 *****
        #Testing with two different methods
        #pdfGmmByClass = loadDensitiesByClass(instances, indexesByClass, gmm)
        pdfKdeByClass = loadDensitiesByClass(instances, indexesByClass, kde)
        # Plotting data distribution by class
        #plotDistributionByClass(instances, indexesByClass)
        
        # ***** Box 4 *****
        #instancesGMM = compactingDataDensityBased(instances, pdfGmmByClass, excludingPercentage)
        instancesKDE = compactingDataDensityBased(instances, pdfKdeByClass, excludingPercentage)
        
        # ***** Box 5 *****
        #X_class1 = instancesGMM[0]
        #X_class2 = instancesGMM[1]
        X_class1 = instancesKDE[0]
        X_class2 = instancesKDE[1]
        initialDataLength=finalDataLength+1
        finalDataLength+=sizeOfBatch
        
        
        
main()
