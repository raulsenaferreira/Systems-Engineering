import numpy as np
from source import util


def cuttingData(instances, pdfByClass, excludingPercentage):
    selectedIndexes = util.compactingDataDensityBased(instances, pdfByClass, excludingPercentage)
    selectedIndexes = np.hstack([selectedIndexes[0],selectedIndexes[1]])
    
    return selectedIndexes