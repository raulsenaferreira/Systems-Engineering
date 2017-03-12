import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import mahalanobis
from math import sqrt
from math import floor


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
    numClasses = len(indexesByClass)
   
    for c, indexes in indexesByClass.items():
        pdfs = [-1] * len(instances)
        points = instances[indexes]
        pdfsByPoints = densityFunction(points, numClasses)
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
    selectedIndexes=[]
    for k in densities:
        arrPdf = densities[k]
        cutLine = max(arrPdf)*criteria
        selectedIndexes.append([i for i in range(len(arrPdf)) if arrPdf[i] != -1 and arrPdf[i] >= cutLine ])
    #if len(selectedIndexes) < 30:

    return selectedIndexes


def loadBestModelByClass(X, indexesByClass, densityFunction):
    bestModelForClass = {}
    numClasses = len(indexesByClass)

    for c, indexes in indexesByClass.items():
        points = X[indexes]
        bestModelForClass[c] = densityFunction(points)

    return bestModelForClass


def mahalanobisCoreSupportExtraction(Ut, indexesPredictedByClass, bestModelSelectedByClass, excludingPercentage):
    inf = 1e6
    selectedMinIndexesByClass={}

    for c in bestModelSelectedByClass:
        distsByComponent = []
        precisions = bestModelSelectedByClass[c].precisions_ #inverse of covariance matrix
        means = bestModelSelectedByClass[c].means_
        pointIndexes = indexesPredictedByClass[c]
        
        for i in range(len(means)):
            dists = []
            v = means[i]
            VI = precisions[i]

            for k in pointIndexes:
                u = Ut[k]
                dist = mahalanobis(u, v, VI)
                dists.append(dist)

            distsByComponent.append(dists)

        distsByComponent = np.array(distsByComponent)
        selectedMinIndexesByClass[c] = [inf]*len(Ut)

        for j in range(len(pointIndexes)):
            vals = distsByComponent[:, j]
            i = vals.argmin()
            selectedMinIndexesByClass[c][pointIndexes[j]] = distsByComponent[i][j]

        p = floor(excludingPercentage*np.max(selectedMinIndexesByClass[c])) 
        p = 70 #20% smallest distances per classs, based on paper
        selectedMinIndexesByClass[c] = np.array(selectedMinIndexesByClass[c])
        selectedMinIndexesByClass[c] = selectedMinIndexesByClass[c].argsort()[:p]
    #print(selectedMinIndexesByClass)
    return selectedMinIndexesByClass