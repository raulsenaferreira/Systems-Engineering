from source import metrics
from source import util
from source import classifiers
from sklearn.base import BaseEstimator, ClassifierMixin

class raulClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clfName='rf', excludingPercentage=0.5, sizeOfBatch=200, batches=50):
        
        self.sizeOfBatch = sizeOfBatch
        self.batches = batches
        self.initialLabeledDataPerc=0.05
        #self.classes=[0, 1]
        self.usePCA=False
        #used only by gmm and cluster-label process
        self.densityFunction='gmm'
        self.excludingPercentage = excludingPercentage
        self.K_variation = 5
        self.clfName=clfName
        #print("{} excluding percecntage".format(excludingPercentage))    
    
    def get_params(self, deep=True):
        return {"excludingPercentage" : self.excludingPercentage, "clfName": self.clfName}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self 

    def fit(self, dataValues, dataLabels=None):
        text = 'Using {} as classifier and excluding percentage = {}'.format(self.clfName, self.excludingPercentage)
        #print(text)
        classes = list(set(dataLabels))
        arrAcc = []
        initialDataLength = 0
        finalDataLength = round((self.initialLabeledDataPerc)*self.sizeOfBatch)
        #print('>>> ',self.batches)
        # ***** Box 1 *****
        #Initial labeled data
        X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, self.usePCA)
        initialDataLength=finalDataLength
        finalDataLength=self.sizeOfBatch
        
        for t in range(self.batches):
            #print(t)
            # ***** Box 2 *****            
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, self.usePCA)
            
            # ***** Box 3 *****
            predicted = classifiers.classify(X, y, Ut, self.K_variation, classes, self.clfName)
            
            # ***** Box 4 *****
            #pdfs from each new points from each class applied on new arrived points
            pdfsByClass = util.pdfByClass2(Ut, predicted, classes)
            
            # ***** Box 5 *****
            selectedIndexes = util.compactingDataDensityBased(pdfsByClass, self.excludingPercentage)
            
            # ***** Box 6 *****
            X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)
            
            initialDataLength=finalDataLength
            finalDataLength+=self.sizeOfBatch
            # Evaluating classification
            arrAcc.append(metrics.evaluate(yt, predicted))
     
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
        return sum(accuracies)/N
