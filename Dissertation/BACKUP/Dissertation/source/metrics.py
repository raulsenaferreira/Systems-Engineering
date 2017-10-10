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


def mcc(arr_y_true, arr_y_predicted):
	arrMcc = []
	m = min(len(arr_y_true), len(arr_y_predicted))

	for y_true, y_predicted in zip(arr_y_true, arr_y_predicted):
		y_true, y_predicted = y_true[:m], y_predicted[:m]
		arrMcc.append(matthews_corrcoef(y_true, y_predicted))
	return arrMcc


def finalEvaluation(arrAcc, batches):
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, 'Accuracy', batches)