from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt
import numpy as np


def evaluate(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    acc = accuracy_score(y_actual, y_predicted)
    #print("RMSE: ", rmse)
    return [rmse, acc]
    

def finalEvaluation(arrRmse, arrAcc):
    print("Average RMSE: ", np.mean(arrRmse))
    print("Standard Deviation: ", np.std(arrRmse))
    print("Variance: ", np.std(arrRmse)**2)
    plotAccuracy(arrRmse, 'RMSE')
    
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, 'Accuracy')