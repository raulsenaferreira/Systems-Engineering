import numpy as np
from source import metrics
from source import util
from source import classifiers
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import euclidean



_SQRT2 = np.sqrt(2)
def hellinger(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def cuttingPercentage(Xt_1, Xt, t):
    res = []
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]
        bins = int(np.sqrt(len(Xt_1)))
        hP = np.histogram(P+(-np.min(P)), bins=bins)
        hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
        res.append(hellinger(hP[1], hQ[1]))
    res = np.mean(res)
    x = np.sqrt(2)

    #similarity = abs((100 - ((100 * res)/x))/100) #ensure non-negativity
    similarity = (100 - ((100 * res)/x))/100
    #print("real value at time",t, (100 - ((100 * res)/x))/100)
    '''if similarity > 0.9:
        similarity = 0.9
    elif similarity < 0.5 and similarity > 0:
        similarity = 0.5'''
    #print(similarity)
    return similarity #percentage of similarity


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]


def makeAccuracy(arrAllAcc, arrTrueY):
    arrAcc = []
    ini = 0
    end = ini
    for predicted in arrAllAcc:
        predicted = np.asarray(predicted)
        predicted = predicted.flatten()
        batchSize = len(predicted)
        ini=end
        end=end+batchSize

        yt = arrTrueY[ini:end]
        arrAcc.append(metrics.evaluate(yt, predicted))
        
    return arrAcc


class proposed_gmm_decision_boundaries(BaseEstimator, ClassifierMixin):

    def __init__(self, K=1, sizeOfBatch=100, batches=50, poolSize=100, isBatchMode=True, initialLabeledData=50):
        #super(proposed_gmm_core_extraction,self).__init__()
        self.sizeOfBatch = sizeOfBatch
        self.batches = batches
        self.initialLabeledData=initialLabeledData
        #self.classes=[0, 1]
        self.usePCA=False
        #used only by gmm and cluster-label process
        self.densityFunction='kde'
        #self.excludingPercentage = excludingPercentage
        self.K = K
        self.clfName = 'label'
        self.poolSize = poolSize
        self.isBatchMode = isBatchMode
        
        #print("{} excluding percecntage".format(excludingPercentage))    
    
    def get_params(self, deep=True):
        return {"K":self.K, "sizeOfBatch":self.sizeOfBatch, "batches":self.batches, "poolSize":self.poolSize, "isBatchMode":self.isBatchMode}
    
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
        reset = False
        if self.isBatchMode:
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
                #clf = classifiers.labelPropagation(X, y, self.K)
                #predicted = clf.predict(Ut)
                # Evaluating classification
                arrAcc.append(metrics.evaluate(yt, predicted))

                # ***** Box 4 *****
                excludingPercentage = cuttingPercentage(X, Ut, t)
                if excludingPercentage < 0:
                    #print("negative, reseting points")
                    excludingPercentage = 0.8
                    reset = True
                else:
                    reset = False
                
                # ***** Box 5 *****
                if reset == True:
                    #Considers only the last distribution (time-series like)
                    pdfsByClass = util.pdfByClass(Ut, predicted, classes, self.densityFunction)
                else:
                    #Considers the past and actual data (concept-drift like)
                    allInstances = np.vstack([X, Ut])
                    allLabels = np.hstack([y, predicted])
                    pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, self.densityFunction)
                    
                selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, 1-excludingPercentage)
                
                # ***** Box 6 *****
                if reset == True:
                    #Considers only the last distribution (time-series like)
                    X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)
                else:
                    #Considers the past and actual data (concept-drift like)
                    X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)
        else:
            inst = []
            labels = []
            clf = classifiers.labelPropagation(X, y, self.K)
            remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), self.usePCA)
            
            for Ut, yt in zip(remainingX, remainingY):
                predicted = clf.predict(Ut.reshape(1, -1))
                arrAcc.append(predicted)
                inst.append(Ut)
                labels.append(predicted)
                
                if len(inst) == self.poolSize:
                    inst = np.asarray(inst)
                    pdfsByClass = util.pdfByClass(inst, labels, classes, self.densityFunction)
                    selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, self.excludingPercentage)
                    X, y = util.selectedSlicedData(inst, labels, selectedIndexes)
                    clf = classifiers.labelPropagation(X, y, self.K)
                    inst = []
                    labels = []
                
            arrAcc = split_list(arrAcc, self.batches)
            arrAcc = makeAccuracy(arrAcc, remainingY)   
            
     
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
