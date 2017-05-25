import numpy as np
from scipy.spatial.distance import mahalanobis
from math import floor
import matplotlib.pyplot as plt
import random
#from experiments.methods import alpha_shape
from source import classifiers



def loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA):
    '''
    X = dataValues.loc[initialDataLength:finalDataLength-1].copy()
    X = X.values
    y = dataLabels.loc[initialDataLength:finalDataLength-1].copy()
    y = y.values[: , 0]
    '''
    X = np.copy(dataValues[initialDataLength:finalDataLength])
    y = np.copy(dataLabels[initialDataLength:finalDataLength])
    if usePCA:
        X = classifiers.pca(X, 2)
    
    return X, y


def selectedSlicedData(instances, labelsInstances, selectedIndexes):
    instances = np.array(instances)
    labelsInstances = np.array(labelsInstances)            
    X = instances[selectedIndexes]
    y = labelsInstances[selectedIndexes]
    
    return X, y


def loadGeometricCoreExtractionByClass(instances, indexesByClass, alpha, threshold):
    selectedPointsByClass = {}
    selectedIndexesByClass = {}
    
    for c in indexesByClass:
        points = instances[indexesByClass[c]]
        inst, indexes, edge_points = alpha_shape.alpha_compaction(points, alpha, threshold)
        selectedPointsByClass[c] = inst
        selectedIndexesByClass[c] = indexes

    return selectedPointsByClass, selectedIndexesByClass


def solve(m1,m2,s1,s2):
    x1 = (s1*s2*np.sqrt((-2*np.log(s1/s2)*s2**2)+2*s1**2*np.log(s1/s2)+m2**2-2*m1*m2+m1**2)+m1*s2**2-m2*s1**2)/(s2**2-s1**2)
    x2 = -(s1*s2*np.sqrt((-2*np.log(s1/s2)*s2**2)+2*s1**2*np.log(s1/s2)+m2**2-2*m1*m2+m1**2)-m1*s2**2+m2*s1**2)/(s2**2-s1**2)
    return x1,x2


def plotDistributionss(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['Class 1', 'Class 2']
    ax = fig.add_subplot(121)
    
    for k, v in distributions.items():
        points = distributions[k]
        points = np.array(points)
        print(points)
        handles.append(ax.scatter(points[:, 0], points[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1
    
    ax.legend(handles, classes)
    
    plt.show()


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
    
    
def loadDensitiesByClass(oldInstances, newInstances, allInstances, oldIndexesByClass, newIndexesByClass, densityFunction):
    previousPdfByClass = {}
    currentPdfByClass = {}
    numClasses = len(newIndexesByClass)
   
    for c in newIndexesByClass:
        oldPdfs = [-1] * len(oldInstances)
        newPdfs = [-1] * len(newInstances)
        
        oldPoints = oldInstances[oldIndexesByClass[c]]
        newPoints = newInstances[newIndexesByClass[c]]
        
        clf = densityFunction(allInstances, numClasses)
        oldPdfsByPoints = np.exp(clf.score_samples(oldPoints))
        newPdfsByPoints = np.exp(clf.score_samples(newPoints))
        
        a = 0
        for i in oldIndexesByClass[c]:
            oldPdfs[i]=oldPdfsByPoints[a]
            a+=1
        previousPdfByClass[c] = oldPdfs
        
        a = 0
        for i in newIndexesByClass[c]:
            newPdfs[i]=newPdfsByPoints[a]
            a+=1
        currentPdfByClass[c] = newPdfs
        
    return previousPdfByClass, currentPdfByClass


def loadDensitiesByClass2(instances, allInstances, indexesByClass, densityFunction):
    pdfsByClass = {}
    numClasses = len(indexesByClass)
   
    for c, indexes in indexesByClass.items():
        pdfs = [-1] * len(instances)
        points = instances[indexes]
        pdfsByPoints = densityFunction(points, allInstances, numClasses)
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
        if len(indexes[classes[c]]) < 1:
            #choose one index randomly if the array is empty
            indexes[classes[c]] = [random.randint(0,len(clusters))]
    
    return indexes


#Cutting data for next iteration
def compactingDataDensityBased(densities, criteria):
    selectedIndexes=[]
    
    for k in densities:
        arrPdf = densities[k]
        cutLine = max(arrPdf)*criteria
        #a = [i for i in range(len(arrPdf)) if arrPdf[i] != -1 and arrPdf[i] >= cutLine ]
        a = [i for i in range(len(arrPdf)) if arrPdf[i] >= cutLine ]
        if len(a) < criteria*len(arrPdf):
            #a=[i for i in range(len(arrPdf)) if arrPdf[i] != -1 and arrPdf[i] >= cutLine*criteria]
            a=[i for i in range(len(arrPdf)) if arrPdf[i] >= cutLine*criteria]
        selectedIndexes.append(a)
   
    stackedIndexes=selectedIndexes[0]
    
    for i in range(1, len(selectedIndexes)):
        stackedIndexes = np.hstack([stackedIndexes,selectedIndexes[i]])
    
    return stackedIndexes


def cuttingDataByIntersection3(x, x2, y):
    
    #getting intersection
    m1 = np.mean(x)
    std1 = np.std(x)
    m2 = np.mean(x2)
    std2 = np.std(x2)

    r = solve(m1,m2,std1,std2)[0]
    
    if np.min(x) < np.min(x2):
        #print("D1 < D2")
        indX = x[:,0]>r
        indX2 = x2[:,0]<r
    else:
        #print("D2 < D1")
        indX = x[:,0]<r
        indX2 = x2[:,0]>r
        
    y=np.array(y) 
    return x[indX], y[indX]


def getDistributionIntersection(X, Ut, indexesByClass, predictedByClass, densityFunction):
    pdfX = {}
    pdfUt = {}
    nComponents = 2

    for c, indexes in indexesByClass.items():
        arrX = []
        arrU = []
        oldPoints = X[indexes]
        newPoints = Ut[predictedByClass[c]]
        GMMX = densityFunction(oldPoints, nComponents)
        GMMU = densityFunction(newPoints, nComponents)

        for i in range(len(newPoints)):
            x = GMMX.predict_proba(newPoints[i])[0]
            x = np.array(x)
            x = x.reshape(1,-1)
            arrU.append(x)
        for i in range(len(oldPoints)):
            u = GMMU.predict_proba(oldPoints[i])[0]
            u = np.array(u)
            u = u.reshape(1, -1)
            arrX.append(u)
        
        pdfUt[c] = arrU
        pdfX[c] = arrX

    plotDistributionss(pdfX)
    plotDistributionss(pdfUt)


def loadBestModelByClass(X, indexesByClass, densityFunction):
    bestModelForClass = {}

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

        #20% smallest distances per class, based on paper
        p = floor(excludingPercentage*len(selectedMinIndexesByClass[c])) 
        #p = 70 
        selectedMinIndexesByClass[c] = np.array(selectedMinIndexesByClass[c])
        selectedMinIndexesByClass[c] = selectedMinIndexesByClass[c].argsort()[:p]
    #print(selectedMinIndexesByClass)
    return selectedMinIndexesByClass


def pdfByClass(oldInstances, oldLabels, newInstances, newLabels, allInstances, classes, densityFunction):
    oldIndexesByClass = slicingClusteredData(oldLabels, classes)
    newIndexesByClass = slicingClusteredData(newLabels, classes)
    
    if densityFunction == 'gmm':
        return loadDensitiesByClass(oldInstances, newInstances, allInstances, oldIndexesByClass, newIndexesByClass, classifiers.gmm)
    elif densityFunction == 'kde':
        return loadDensitiesByClass(oldInstances, newInstances, allInstances, oldIndexesByClass, newIndexesByClass, classifiers.kde)
    else:
        print ("Choose between 'gmm' or 'kde' function. Wrong name given: ", densityFunction)
        return
    
    
def pdfByClass2(instances, labels, classes):
    indexesByClass = slicingClusteredData(labels, classes)
    
    pdfsByClass = {}
    numClasses = len(indexesByClass)
    #print("{} instances".format(len(instances)))
    for c, indexes in indexesByClass.items():
        if len(indexes) > 0:
            pdfs = [-1] * len(instances)
            #print("class {} = {} points".format(c, len(indexes)))
            #print(indexes)
            points = instances[indexes]
            #points from a class, all points, number of components
            pdfsByPoints = classifiers.gmmWithPDF(points, points, numClasses)#(points, instances, numClasses)
            a = 0
            for i in indexes:
                pdfs[i]=pdfsByPoints[a]
                a+=1
            pdfsByClass[c] = pdfs
        
    return pdfsByClass


#Cutting data for next iteration
def compactingDataDensityBased2(densities, criteria):
    cut = 1-criteria
    selectedIndexes=[]
    
    for k in densities:
        arrPdf = np.array(densities[k])
        numSelected = int(np.ceil(cut*len(arrPdf)))
        ind = (-arrPdf).argsort()[:numSelected]
        selectedIndexes.append(ind)
   
    stackedIndexes=selectedIndexes[0]
    
    for i in range(1, len(selectedIndexes)):
        stackedIndexes = np.hstack([stackedIndexes,selectedIndexes[i]])
    
    return stackedIndexes