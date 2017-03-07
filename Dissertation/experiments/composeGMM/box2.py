
def process(dataValues, initialDataLength, finalDataLength):
    U = dataValues.loc[initialDataLength:finalDataLength].copy()
    Ut = U.values
    
    return Ut