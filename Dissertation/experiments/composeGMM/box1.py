
def process(dataValues, dataLabels, initialDataLength):
    X = dataValues.loc[:initialDataLength].copy()
    X = X.values
    y = dataLabels.loc[:initialDataLength].copy()
    y = y.values[: , 0]
    
    return X, y