import numpy as np
from source import metrics
from source import util
from source import classifiers
from sklearn.base import BaseEstimator, ClassifierMixin

class proposed_gmm_core_svm_boundaries_extraction(BaseEstimator, ClassifierMixin):

    def __init__(self, excludingPercentage=0.05, K=1, sizeOfBatch=100, batches=50):
        #super(proposed_gmm_core_extraction,self).__init__()
        self.sizeOfBatch = sizeOfBatch
        self.batches = batches
        self.initialLabeledData=50
        #self.classes=[0, 1]
        self.usePCA=False
        #used only by gmm and cluster-label process
        self.densityFunction='gmm'
        self.excludingPercentage = excludingPercentage
        #self.C = C
        self.K = K
        self.clfName = 'knn'
        '''print(excludingPercentage)
        print(K)
        print(sizeOfBatch)
        print(batches)'''
        
        #print("{} excluding percecntage".format(excludingPercentage))    
    
    def get_params(self, deep=True):
        return {"excludingPercentage" : self.excludingPercentage, "K":self.K, "sizeOfBatch":self.sizeOfBatch, "batches":self.batches}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
            
    def fit(self, dataValues, dataLabels=None):
        arrAcc = []
        classes = list(set(dataLabels))
        initialDataLength = 0
        self.excludingPercentage=1-self.excludingPercentage
        finalDataLength = self.initialLabeledData
        # ***** Box 1 *****
        #Initial labeled data
        X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, self.usePCA)
        
        for t in range(self.batches):
            #print("passo: ",t)
            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+self.sizeOfBatch
            #print(initialDataLength)
            #print(finalDataLength)
            # ***** Box 2 *****            
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, self.usePCA)
            
            # ***** Box 3 *****
            clf = classifiers.svmClassifier(X, y)
            #predicted = clf.predict(Ut)
            predicted = classifiers.classify(X, y, Ut, self.K, classes, self.clfName)
            # Evaluating classification
            arrAcc.append(metrics.evaluate(yt, predicted))

            # ***** Box 4 *****
            #removing boundaryPoints for the next batch
            Ut, predicted = util.removeBoundaryPoints(clf.support_, Ut, predicted)
            #pdfs from each new points from each class applied on new arrived points
            pdfsByClass = util.pdfByClass(Ut, predicted, classes, self.densityFunction)
            
            # ***** Box 5 *****
            selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, self.excludingPercentage)
            
            # ***** Box 6 *****
            X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)
           
            
     
        # returns accuracy array and last selected points
        self.threshold_ = arrAcc
        return self
    
    def predict(self):
        try:
            getattr(self, "threshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return self.threshold_
    
    def score(self, X, y=None):
        accuracies = self.predict()
        N = len(accuracies)
        #print(self.K, self.excludingPercentage, sum(accuracies)/N)
        return sum(accuracies)/N