import numpy as np
from source import classifiers


def classify(X, y, Ut, K):
    predicted = classifiers.clusterAndLabel(X, y, Ut, K)
    return predicted
    

def stack(X, Ut, y, predicted):
    instances = np.vstack([X, Ut])
    labelsInstances = np.hstack([y, predicted])
    
    return instances, labelsInstances