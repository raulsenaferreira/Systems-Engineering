import numpy as np
from source import util


def cuttingDataByPercentage(instances, pdfByClass, excludingPercentage):
    selectedIndexes = util.compactingDataDensityBased(instances, pdfByClass, excludingPercentage)
    selectedIndexes = np.hstack([selectedIndexes[0],selectedIndexes[1]])
    
    return selectedIndexes


def cuttingDataByDistance(Ut, predicted, bestModelSelectedByClass, excludingPercentage):
	predictedByClass = util.slicingClusteredData(predicted, [0,1])
	selectedIndexes = util.mahalanobisCoreSupportExtraction(Ut, predictedByClass, bestModelSelectedByClass, excludingPercentage)
	selectedIndexes = np.hstack([selectedIndexes[0],selectedIndexes[1]])
    
	return selectedIndexes


def cuttingDataByIntersection(previousX, currentX, previousLabels, currentLabels, classes):
    selectedLabels = []
    selectedPoints = []
    
    indexesByClassOld = util.slicingClusteredData(previousLabels, classes)
    indexesByClassNew = util.slicingClusteredData(currentLabels, classes)
    #print(previousLabels)
    #print(currentLabels)
    for c in indexesByClassOld:
        oldInd = indexesByClassOld[c]
        newInd = indexesByClassNew[c]
        
        x = previousX[oldInd]
        y = previousLabels[oldInd]
        x2 = currentX[newInd]
        y2 = currentLabels[newInd]
        
        #getting intersection
        m1 = np.mean(previousX[oldInd])
        std1 = np.std(previousX[oldInd])
        m2 = np.mean(currentX[newInd])
        std2 = np.std(currentX[newInd])
        r = util.solve(m1,m2,std1,std2)[0]
        
        if np.min(x) < np.min(x2):
        #print("D1 < D2")
            indX = x[:,0]>r
            indX2 = x2[:,0]<r
        else:
            #print("D2 < D1")
            indX = x[:,0]<r
            indX2 = x2[:,0]>r
    
        selectedPoints.append(np.vstack([x[indX], x2[indX2]]))
        #print("Intersection points: ",len(selectedPoints[0]))
        #print(y)
        selectedLabels.append(np.hstack([y[indX], y2[indX2]]))
        
    return selectedPoints, selectedLabels


def cuttingDataByIntersection2(x, x2, y, y2):
    #getting intersection
    m1 = np.mean(x)
    std1 = np.std(x)
    m2 = np.mean(x2)
    std2 = np.std(x2)

    r = util.solve(m1,m2,std1,std2)[0]

    if np.min(x) < np.min(x2):
        #print("D1 < D2")
        indX = x[:,0]>r
        indX2 = x2[:,0]<r
    else:
        #print("D2 < D1")
        indX = x[:,0]<r
        indX2 = x2[:,0]>r
        
    return np.vstack([x[indX], x2[indX2]]), np.hstack([y[indX], y2[indX2]])


def cuttingDataByIntersection3(x, x2, y):
    
    #getting intersection
    m1 = np.mean(x)
    std1 = np.std(x)
    m2 = np.mean(x2)
    std2 = np.std(x2)

    r = util.solve(m1,m2,std1,std2)[0]
    
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