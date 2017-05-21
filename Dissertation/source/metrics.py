import os
import sys
sys.path.append(os.getcwd()+"\\source")

from sklearn.metrics import accuracy_score
from math import sqrt
import numpy as np
from source.plotFunctions import plotAccuracy


def evaluate(y_actual, y_predicted):
    return accuracy_score(y_actual, y_predicted)
    

def finalEvaluation(arrAcc, batches):
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, 'Accuracy', batches)