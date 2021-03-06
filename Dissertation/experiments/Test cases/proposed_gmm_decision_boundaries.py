import numpy as np
from source import classifiers
from source import metrics
from source import util
import matplotlib.pyplot as plt


def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    #initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    initialLabeledData = kwargs["initialLabeledData"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    classes = kwargs["classes"]
    K = kwargs["K_variation"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    excludingPercentage = kwargs["excludingPercentage"]
    clfName = kwargs["clfName"]
    densityFunction = kwargs["densityFunction"]
    
    print("METHOD: K-NN as classifier and {} as core support extraction with cutting data method".format(densityFunction))

    arrAcc = []
    arrX = []
    arrY = []
    arrDecisionBoundaries = []
    initialDataLength = 0
    excludingPercentage = 1-excludingPercentage
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

    for t in range(batches):
        #print("passo: ",t)
        initialDataLength=finalDataLength
        finalDataLength=finalDataLength+sizeOfBatch
        #print(initialDataLength)
        #print(finalDataLength)
        # ***** Box 2 *****
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        
        # ***** Box 3 *****
        #predicted = classifiers.classify(X, y, Ut, K, classes, clfName)
        clf = classifiers.knn(X, y, K)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        predicted = clf.predict(Ut)
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))

        # ***** Box 4 *****
        #pdfs from each new points from each class applied on new arrived points
        pdfsByClass = util.pdfByClass(Ut, predicted, classes, densityFunction)

        # ***** Box 5 *****
        selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)

        # ***** Box 6 *****
        X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)

    plt.show()
    # returns accuracy array and last selected points
    return "KNN + Fixed cutting data percentage", arrAcc, X, y