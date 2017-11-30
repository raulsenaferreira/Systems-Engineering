import numpy as np
from source import metrics
from source import util
from source import classifiers
from sklearn.base import BaseEstimator, ClassifierMixin

class label_propagation(BaseEstimator, ClassifierMixin):

    def __init__(self, K=1, sizeOfBatch=100, batches=50, poolSize=100, isBatchMode=True, initialLabeledData=50):
        #super(proposed_gmm_core_extraction,self).__init__()
        self.sizeOfBatch = sizeOfBatch
        self.batches = batches
        self.initialLabeledData=initialLabeledData
        #self.classes=[0, 1]
        self.usePCA=False
        #used only by gmm and cluster-label process
        #self.C = C
        self.K = K
        self.clfName = 'label'
        '''print(excludingPercentage)
        print(K)
        print(sizeOfBatch)
        print(batches)'''
        
        #print("{} excluding percecntage".format(excludingPercentage))    
    
    def get_params(self, deep=True):
        return {"K":self.K, "sizeOfBatch":self.sizeOfBatch, "batches":self.batches}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
            
    def fit(self, dataValues, dataLabels=None):
        arrAcc = []
        classes = list(set(dataLabels))
        initialDataLength = 0
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
            predicted = classifiers.classify(X, y, Ut, self.K, classes, self.clfName)
            # Evaluating classification
            arrAcc.append(metrics.evaluate(yt, predicted))

            # ***** Box 4 *****
            #removing boundaryPoints for the next batch
            
            # ***** Box 6 *****
            X, y = Ut, predicted
           
            
     
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