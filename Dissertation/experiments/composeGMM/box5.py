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
    
    for c in indexesByClassOld:
        oldInd = indexesByClassOld[c]
        newInd = indexesByClassNew[c]
        
        x = previousX[oldInd]
        y = previousLabels[c]
        x2 = currentX[newInd]
        y2 = currentLabels[c]
        
        #getting intersection
        m1 = np.mean(previousX[oldInd])
        std1 = np.std(previousX[oldInd])
        m2 = np.mean(currentX[newInd])
        std2 = np.std(currentX[newInd])
        r = util.solve(m1,m2,std1,std2)[0]
        
        indX = x[:,0]<r
        indX2 = x2[:,0]>r
    
        selectedPoints.append(np.vstack([x[indX], x2[indX2]]))
        print("Intersection points: ",len(selectedPoints[0]))
        print(y)
        selectedLabels.append(np.vstack([y[indX], y2[indX2]]))
        
    return selectedPoints, selectedLabels