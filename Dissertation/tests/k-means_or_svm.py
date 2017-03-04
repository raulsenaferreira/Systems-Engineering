import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt


def test1(dataValues, dataLabels, classifier='kmeans', batches = 50, sizeOfBatch = 365, initialLabeledDataPerc=0.05):
    print(">>>>> STARTING TEST 1 <<<<<")
    sizeOfLabeledData = round((initialLabeledDataPerc)*sizeOfBatch)
    initialDataLength = sizeOfLabeledData
    finalDataLength = sizeOfBatch
    classes=[0, 1]
    
    X = dataValues.loc[:initialDataLength].copy()
    X = X.values
    y = dataLabels.loc[:initialDataLength].copy()
    y = y.values
    
    arrRmse = []
    arrAcc = []
    
    for t in range(batches):
        #print("Step ",t+1)
        
        U = dataValues.loc[initialDataLength:finalDataLength].copy()
        Ut = U.values
        yt = dataLabels.loc[initialDataLength:finalDataLength].copy()
        yt = yt.values
        predicted=[]
        
        if classifier == 'kmeans':
            kmeans = kMeans(pca(X, 2), 2)
            predicted = baseClassifier(pca(Ut, 2), kmeans)
        elif classifier == 'svm':
            svmClf = svmClassifier(pca(X, 2), y)
            predicted = baseClassifier(pca(Ut, 2), svmClf)
        else:
            return
        
        arrRmse.append(evaluate(yt, predicted)[0])
        arrAcc.append(evaluate(yt, predicted)[1])
        initialDataLength=finalDataLength+1
        finalDataLength+=sizeOfBatch
        # keep a percentage from former distribution to train in next step 
        #X = Ut
        #y = yt
    
    print(finalEvaluation(arrRmse, arrAcc))
    
    print(">>>>> END OF TEST 1 <<<<<")