
def process(dataValues, dataLabels, initialDataLength, finalDataLength):
    X = dataValues.loc[initialDataLength:finalDataLength].copy()
    X = X.values
    y = dataLabels.loc[initialDataLength:finalDataLength].copy()
    y = y.values[: , 0]
    
    return X, y