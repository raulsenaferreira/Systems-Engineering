from source import classifiers
from source import util
from source import alpha_shape


def geometricCoreExtraction(instances, labelsInstances, classes, alpha, threshold):
    indexesByClass = util.slicingClusteredData(labelsInstances, classes)
    
    return util.loadGeometricCoreExtractionByClass(instances, indexesByClass, alpha, threshold)


def pdfByClass(oldInstances, oldLabels, newInstances, newLabels, allInstances, classes, densityFunction):
    oldIndexesByClass = util.slicingClusteredData(oldLabels, classes)
    newIndexesByClass = util.slicingClusteredData(newLabels, classes)
    
    if densityFunction == 'gmm':
        return util.loadDensitiesByClass(oldInstances, newInstances, allInstances, oldIndexesByClass, newIndexesByClass, classifiers.gmm)
    elif densityFunction == 'kde':
        return util.loadDensitiesByClass(instances, indexesByClass, classifiers.kde)
    else:
        print ("Choose between 'gmm' or 'kde' function. Wrong name given: ", densityFunction)
        return
    
    
def pdfByClass2(instances, labelsInstances, allInstances, classes, densityFunction):
    indexesByClass = util.slicingClusteredData(labelsInstances, classes)

    if densityFunction == 'gmm':
        return util.loadDensitiesByClass2(instances, allInstances, indexesByClass, classifiers.gmmWithPDF)
    elif densityFunction == 'kde':
        return util.loadDensitiesByClass2(instances, allInstances, indexesByClass, classifiers.kde)
    else:
        print ("Choose between 'gmm' or 'kde' function. Wrong name given: ", densityFunction)
        return


def pdfByClass3(instances, labelsInstances, allInstances, classes, densityFunction):
    indexesByClass = util.slicingClusteredData(labelsInstances, classes)
    return util.loadDensitiesByClass2(instances, allInstances, indexesByClass, classifiers.gmm)
    
    
def bestModelSelectedByClass(X, y, classes, densityFunction):
	indexesByClass = util.slicingClusteredData(y, classes)
	if densityFunction == 'gmmBIC':
		return util.loadBestModelByClass(X, indexesByClass, classifiers.gmmWithBIC)

def intersection(X, y, Ut, predicted, classes, densityFunction):
	indexesByClass = util.slicingClusteredData(y, classes)
	predictedByClass = util.slicingClusteredData(predicted, classes)
	if densityFunction == 'gmm':
		return util.getDistributionIntersection(X, Ut, indexesByClass, predictedByClass, classifiers.gmmPure)