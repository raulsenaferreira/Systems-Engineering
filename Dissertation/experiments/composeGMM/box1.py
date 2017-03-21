from source import classifiers

def process(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA):
    X = dataValues.loc[initialDataLength:finalDataLength-1].copy()
    X = X.values
    y = dataLabels.loc[initialDataLength:finalDataLength-1].copy()
    y = y.values[: , 0]
    if usePCA:
        X = classifiers.pca(X, 2)
    
    return X, y