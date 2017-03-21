from source import classifiers

def process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA):
    U = dataValues.loc[initialDataLength:finalDataLength].copy()
    Ut = U.values
    yt = dataLabels.loc[initialDataLength:finalDataLength].copy()
    yt = yt.values
    if usePCA:
        Ut = classifiers.pca(Ut, 2)
    
    return Ut, yt