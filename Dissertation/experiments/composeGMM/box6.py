import numpy as np


def selectedData(instances, labelsInstances, selectedIndexes):
    instances = np.array(instances)
    labelsInstances = np.array(labelsInstances)            
    X = instances[selectedIndexes]
    y = labelsInstances[selectedIndexes]
    
    return X, y