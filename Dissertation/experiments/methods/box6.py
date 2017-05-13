import numpy as np


def selectedSlicedData(instances, labelsInstances, selectedIndexes):
    instances = np.array(instances)
    labelsInstances = np.array(labelsInstances)            
    X = instances[selectedIndexes]
    y = labelsInstances[selectedIndexes]
    
    return X, y


def gettingSelectedData(selectedPointsByClass, selectedIndexesByClass, labelsInstances):
    X = np.vstack([selectedPointsByClass[0], selectedPointsByClass[1]])
    y = np.hstack([labelsInstances[selectedIndexesByClass[0]], labelsInstances[selectedIndexesByClass[1]]])
    
    return X, y