import numpy as np
import pandas as pd
from experiments.methods import checkerboard
 
                
                
def loadNOAADataset(path):
    #Test sets: Predicting 365 instances by step. 50 steps. Two classes.
    '''
    NOAA dataset:
    Eight  features  (average temperature, minimum temperature, maximum temperature, dew
    point,  sea  level  pressure,  visibility,  average wind speed, maximum  wind  speed)
    are  used  to  determine  whether  each  day  experienced  rain  or no rain.
    '''
    dataValues = pd.read_csv(path+'noaa_data.csv',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = pd.read_csv(path+'noaa_label.csv',sep = ",")
    dataLabels = pd.DataFrame.as_matrix(dataLabels)
    
    return dataValues, dataLabels


def loadCheckerBoard(path):
    #Test sets: Predicting N instances by step. T steps. Two classes.
    '''
    Rotated checkerboard dataset. Rotating 2*PI
    '''
    T = 50 #time steps
    N = 365 #instances
    a = np.linspace(0,2*np.pi,T)
    side = 0.25
    
    auxV, auxL = checkerboard.generateData(side, a, N, T)
    #print(dV[2][0])#tempo 2 da classe 0
    dataLabels = auxL[0]
    dataValues = auxV[0]
    
    for i in range(1, T):
        dataLabels = np.hstack([dataLabels, auxL[i]])
        dataValues = np.vstack([dataValues, auxV[i]])
    
    return dataValues, dataLabels 


def loadCDT(path):
    #Test set: One Class Diagonal Translation. 2 Dimensional data
    return loadData(pd.read_csv(path+'1CDT.txt',sep = ","))


def loadCHT(path):
    #Test set: One Class Horizontal Translation. 2 Dimensional data
    return loadData(pd.read_csv(path+'1CHT.txt',sep = ","))


def load2CDT(path):
    #Test set: Two Classes Diagonal Translation. 2 Dimensional data
    return loadData(pd.read_csv(path+'2CDT.txt',sep = ","))


def load2CHT(path):
    #Test set: Two Classes Horizontal Translation. 2 Dimensional data
    return loadData(pd.read_csv(path+'2CHT.txt',sep = ","))


def loadUG_2C_2D(path):
    #Test set: Bidimensional Unimodal Gaussian with Two Classes 
    return loadData(pd.read_csv(path+'UG_2C_2D.txt',sep = ","))


def loadUG_2C_3D(path):
    #Test set: Artificial 3-dimensional Unimodal Gaussian with Two Classes
    return loadData(pd.read_csv(path+'UG_2C_3D.txt',sep = ","))


def loadUG_2C_5D(path):
    #Test set: 5-dimensional Unimodal Gaussian with Two Classes
    return loadData(pd.read_csv(path+'UG_2C_5D.txt',sep = ","))


def loadMG_2C_2D(path):
    #Test set: Bidimensional Mulitimodal Gaussian with Two Classes
    return loadData(pd.read_csv(path+'MG_2C_2D.txt',sep = ","))


def loadData(dataValues):
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataValues = dataValues[:,0:2]
    
    dataLabels = dataLabels.astype(int) - 1
    dataLabels = dataLabels.reshape(len(dataLabels), 1)
    
    return dataValues, dataLabels