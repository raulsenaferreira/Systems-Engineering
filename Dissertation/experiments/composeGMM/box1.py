from source import classifiers
import numpy as np

def process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA):
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