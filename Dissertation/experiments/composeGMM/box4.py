from source import classifiers
from source import util


def pdfByClass(instances, labelsInstances, classes, densityFunction):
    indexesByClass = util.slicingClusteredData(labelsInstances, classes)

    if densityFunction == 'gmm':
        return util.loadDensitiesByClass(instances, indexesByClass, classifiers.gmm)
    elif densityFunction == 'kde':
        return util.loadDensitiesByClass(instances, indexesByClass, classifiers.kde)
    else:
        print ("Choose between 'gmm' or 'kde' function. Wrong name given: ", densityFunction)
        return