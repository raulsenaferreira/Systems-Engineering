import numpy as np
from source import classifiers
from source import metrics
from source import util


def compose(dataValues, dataLabels, densityFunction='gmm', excludingPercentage = 0.2, batches = 50, sizeOfBatch = 365, initialLabeledDataPerc=0.05, classes = [0,1], K = 5):
    
    print(">>>>> STARTING TEST with K-Means, Cluster and label as classifier and ", densityFunction, " as cutting data <<<<<")
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = sizeOfLabeledData
    finalDataLength = sizeOfBatch
    arrAcc = []
    
    # ***** Box 1 *****
    X = dataValues.loc[:initialDataLength].copy()
    X = X.values
    y = dataLabels.loc[:initialDataLength].copy()
    y = y.values[: , 0]
    
    #Starting the process
    for t in range(batches):
        #print("Step ",t+1)
        
        # ***** Box 2 *****
        U = dataValues.loc[initialDataLength:finalDataLength].copy()
        Ut = U.values

        # ***** Box 3 *****
        predicted = classifiers.clusterAndLabel(X, y, Ut, K)
        instances = np.vstack([X, Ut])
        labelsInstances = np.hstack([y, predicted])
        # ***** Evaluating *****
        #print(len(instances), " Points")
        yt = dataLabels.loc[initialDataLength:finalDataLength].copy()
        yt = yt.values
        
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # ***** Box 4 *****
        indexesByClass = util.slicingClusteredData(labelsInstances, classes)
        
        pdfByClass=''
        if densityFunction == 'gmm':
            pdfByClass = util.loadDensitiesByClass(instances, indexesByClass, classifiers.gmm)
        elif densityFunction == 'kde':
            pdfByClass = util.loadDensitiesByClass(instances, indexesByClass, classifiers.kde)
        else:
            print ("Choose between 'gmm' or 'kde' function. Wrong name given: ", densityFunction)
            return 
        
        #Plotting data distribution by class
        #plotDistributionByClass(instances, indexesByClass)
        
        # ***** Box 5 *****
        selectedIndexes = util.compactingDataDensityBased(instances, pdfByClass, excludingPercentage)
        #X_class1 = selectedIndexes[0]
        #X_class2 = selectedIndexes[1]
        selectedIndexes = np.hstack([selectedIndexes[0],selectedIndexes[1]])
        
        # ***** Box 6 *****
        instances = np.array(instances)
        labelsInstances = np.array(labelsInstances)            
        X = instances[selectedIndexes]
        y = labelsInstances[selectedIndexes]      
        #updating indexes
        initialDataLength=finalDataLength+1
        finalDataLength+=sizeOfBatch
        
        
    metrics.finalEvaluation(arrAcc)
    print(">>>>> END OF TEST <<<<<")