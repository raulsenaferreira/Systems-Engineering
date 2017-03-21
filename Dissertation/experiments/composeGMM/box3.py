import numpy as np
from source import classifiers
from source import util


def classify(X, y, Ut, K, classifier):

	if classifier == 'cluster_and_label':
		return classifiers.clusterAndLabel(X, y, Ut, K)
	elif classifier == 'svm':
		svmClf = classifiers.svmClassifier(X, y)
		return util.baseClassifier(Ut, svmClf)
	else:
		return
    

def stack(X, Ut, y, predicted):
    instances = np.vstack([X, Ut])
    labelsInstances = np.hstack([y, predicted])
    
    return instances, labelsInstances