import os
import sys
sys.path.append(os.getcwd()+"\\source")

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from math import sqrt
import numpy as np
from source.plotFunctions import plotAccuracy


def evaluate(y_actual, y_predicted):
    return accuracy_score(y_actual, y_predicted)*100
    

def F1(y_true, y_pred):
	return f1_score(y_true, y_pred, average=None)


def macroF1(y_true, y_pred):
	return f1_score(y_true, y_pred, average='macro')


def microF1(y_true, y_pred):
	return f1_score(y_true, y_pred, average='micro')


def mcc(Y_true, y_predicted):
	return matthews_corrcoef(y_true, y_predicted)


def finalEvaluation(arrAcc, batches):
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, 'Accuracy', batches)