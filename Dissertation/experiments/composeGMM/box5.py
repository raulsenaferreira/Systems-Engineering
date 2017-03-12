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