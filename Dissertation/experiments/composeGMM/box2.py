from source import classifiers

def process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA):
    U = dataValues.loc[initialDataLength:finalDataLength-1].copy()
    Ut = U.values
    yt = dataLabels.loc[initialDataLength:finalDataLength-1].copy()
    yt = yt.values
    if usePCA:
        Ut = classifiers.pca(Ut, 2)
    
    return Ut, yt