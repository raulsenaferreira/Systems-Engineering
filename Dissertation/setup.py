import numpy as np
import pandas as pd
from experiments import checkerboard
 
                
                
def loadNOAADataset(path):
    #Test sets: Predicting 365 instances by step. 50 steps. Two classes.
    '''
    NOAA dataset:
    Eight  features  (average temperature, minimum temperature, maximum temperature, dew
    point,  sea  level  pressure,  visibility,  average wind speed, maximum  wind  speed)
    are  used  to  determine  whether  each  day  experienced  rain  or no rain.
    '''
    #dataValues = pd.read_csv(path+'noaa_data.csv',sep = ",")
    #dataValues = pd.DataFrame.as_matrix(dataValues)
    #dataLabels = pd.read_csv(path+'noaa_label.csv',sep = ",")
    #dataLabels = pd.DataFrame.as_matrix(dataLabels)
    
    return 


def loadCheckerBoard(path):
    #Test sets: Predicting N instances by step. T steps. Two classes.
    '''
    Rotated checkerboard dataset. Rotating 2*PI
    '''
    T = 100 #time steps
    N = 200 #instances
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
    '''
    Artificial One Class Diagonal Translation. 2 Dimensional data
    '''
    dataValues = pd.read_csv(path+'1CDT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels


def loadCHT(path):
    #Test set: One Class Horizontal Translation. 2 Dimensional data
    '''
    Artificial One Class Horizontal Translation. 2 Dimensional data
    '''
    dataValues = pd.read_csv(path+'1CHT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels


def load2CDT(path):
    #Test set: Two Classes Diagonal Translation. 2 Dimensional data
    '''
    Artificial Two Classes Diagonal Translation. 2 Dimensional data
    '''
    dataValues = pd.read_csv(path+'2CDT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels


def load2CHT(path):
    #Test set: Two Classes Horizontal Translation. 2 Dimensional data
    '''
    Artificial Two Classes Horizontal Translation. 2 Dimensional data
    '''
    dataValues = pd.read_csv(path+'2CHT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels


def loadUG_2C_2D(path):
    #Test set: Two Bidimensional Unimodal Gaussian Classes 
    '''
    Artificial Two Bidimensional Unimodal Gaussian Classes 
    '''
    dataValues = pd.read_csv(path+'UG_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels


def loadUG_2C_3D(path):
    #Test set: Artificial Two 3-dimensional Unimodal Gaussian Classes
    '''
    Artificial Two 3-dimensional Unimodal Gaussian Classes
    '''
    dataValues = pd.read_csv(path+'UG_2C_3D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 3]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:3]
    #print(dataValues)
    
    return dataValues, dataLabels


def loadUG_2C_5D(path):
    #Test set: Two 5-dimensional Unimodal Gaussian Classes
    '''
    Artificial Two 5-dimensional Unimodal Gaussian Classes
    '''
    dataValues = pd.read_csv(path+'UG_2C_5D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 5]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:5]
    #print(dataValues)
    
    return dataValues, dataLabels


def loadMG_2C_2D(path):
    #Test set: Two Bidimensional Mulitimodal Gaussian Classes
    '''
    Artificial Two Bidimensional Mulitimodal Gaussian Classes
    '''
    dataValues = pd.read_csv(path+'MG_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels


def loadFG_2C_2D(path):
    dataValues = pd.read_csv(path+'FG_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels


def loadGEARS_2C_2D(path):
    dataValues = pd.read_csv(path+'GEARS_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels


def loadCSurr(path):
    dataValues = pd.read_csv(path+'1CSurr.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels


def loadKeystroke(path):
    dataValues = pd.read_csv(path+'keystroke.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 10]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:10]
    #print(dataValues)
    
    return dataValues, dataLabels