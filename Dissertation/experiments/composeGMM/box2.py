from source import classifiers
import numpy as np

def process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA):
    '''
    U = dataValues.loc[initialDataLength:finalDataLength-1].copy()
    Ut = U.values
    yt = dataLabels.loc[initialDataLength:finalDataLength-1].copy()
    yt = yt.values
    '''
    Ut = np.copy(dataValues[initialDataLength:finalDataLength])
    yt = np.copy(dataLabels[initialDataLength:finalDataLength])
    if usePCA:
        Ut = classifiers.pca(Ut, 2)
    
    return Ut, yt