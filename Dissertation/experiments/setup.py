import numpy as np
import pandas as pd
#from experiments import checkerboard
 

#artificial datasets                
def loadCDT(path, sep):
    #Test set: One Class Diagonal Translation. 2 Dimensional data
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CDT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Artificial One Class Diagonal Translation. 2 Dimensional data.'


def loadCHT(path, sep):
    #Test set: One Class Horizontal Translation. 2 Dimensional data

    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CHT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Artificial One Class Horizontal Translation. 2 Dimensional data.'


def load2CDT(path, sep):
    #Test set: Two Classes Diagonal Translation. 2 Dimensional data
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'2CDT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Artificial Two Classes Diagonal Translation. 2 Dimensional data.'


def load2CHT(path, sep):
    #Test set: Two Classes Horizontal Translation. 2 Dimensional data
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'2CHT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Artificial Two Classes Horizontal Translation. 2 Dimensional data.'


def loadUG_2C_2D(path, sep):
    #Test set: Two Bidimensional Unimodal Gaussian Classes 
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'UG_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Artificial Two Bidimensional Unimodal Gaussian Classes.'


def loadUG_2C_3D(path, sep):
    #Test set: Artificial Two 3-dimensional Unimodal Gaussian Classes
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'UG_2C_3D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 3]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:3]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Artificial Two 3-dimensional Unimodal Gaussian Classes.'


def loadUG_2C_5D(path, sep):
    #Test set: Two 5-dimensional Unimodal Gaussian Classes
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'UG_2C_5D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 5]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:5]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Artificial Two 5-dimensional Unimodal Gaussian Classes.'


def loadMG_2C_2D(path, sep):
    #Test set: Two Bidimensional Mulitimodal Gaussian Classes
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'MG_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Artificial Two Bidimensional Mulitimodal Gaussian Classes.'


def loadFG_2C_2D(path, sep):
    dataValues = pd.read_csv(path+'sinthetic'+sep+'FG_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Two Bidimensional Classes as Four Gaussians.'


def loadGEARS_2C_2D(path, sep):
    dataValues = pd.read_csv(path+'sinthetic'+sep+'GEARS_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, 'Two Rotating Gears (Two classes. Bidimensional).'


def loadCSurr(path, sep):
    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CSurr.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, 'One Class Surrounding another Class. Bidimensional.'


def load5CVT(path, sep):
    dataValues = pd.read_csv(path+'sinthetic'+sep+'5CVT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), 'Five Classes Vertical Translation. Bidimensional.'


def load4CR(path, sep):
    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CR.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), 'Four Classes Rotating Separated. Bidimensional.'


def load4CRE_V1(path, sep):
    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CRE-V1.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), 'Four Classes Rotating with Expansion V1. Bidimensional.'


def load4CRE_V2(path, sep):
    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CRE-V2.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), 'Four Classes Rotating with Expansion V2. Bidimensional.'

def load4CE1CF(path, sep):
    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CE1CF.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), 'Four Classes Expanding and One Class Fixed. Bidimensional.'

def loadCheckerBoard(path, sep, T=300, N=2000):
    #Test sets: Predicting N instances by step. T steps. Two classes.
    '''
    Same parameters from original work
    '''
    a = np.linspace(0,2*np.pi,T)
    side = 0.25
    
    auxV, auxL = checkerboard.generateData(side, a, N, T)
    #print(dV[2][0])#tempo 2 da classe 0
    dataLabels = auxL[0]
    dataValues = auxV[0]
    
    for i in range(1, T):
        dataLabels = np.hstack([dataLabels, auxL[i]])
        dataValues = np.vstack([dataValues, auxV[i]])
    
    return dataValues, dataLabels, 'Rotated checkerboard dataset. Rotating 2*PI.'

#real datasets
def loadKeystroke(path, sep):
    dataValues = pd.read_csv(path+'keystroke'+sep+'keystroke.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 10]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:10]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), 'Keyboard patterns database. 10 features. 4 classes.'


def loadElecData(path, sep):
    #dataValues = pd.read_csv(path+'elecdata/elec2.data',sep = ",")
    dataValues = pd.read_csv(path+'elecdata'+sep+'elec2_data.dat',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = pd.read_csv(path+'elecdata'+sep+'elec2_label.dat',sep = ",")
    dataLabels = pd.DataFrame.as_matrix(dataLabels)
    #dataLabels = dataLabels-1
    print(dataLabels)
    return dataValues, dataLabels[:,0], 'Electricity data. 10 features. 2 classes.'


def loadNOAADataset(path, sep):
    #Test sets: Predicting 365 instances by step. 50 steps. Two classes.
    '''
    NOAA dataset:
    Eight  features  (average temperature, minimum temperature, maximum temperature, dew
    point,  sea  level  pressure,  visibility,  average wind speed, maximum  wind  speed)
    are  used  to  determine  whether  each  day  experienced  rain  or no rain.
    '''
    dataValues = pd.read_csv(path+'noaa'+sep+'noaa_data.csv',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = pd.read_csv(path+'noaa'+sep+'noaa_label.csv',sep = ",")
    dataLabels = pd.DataFrame.as_matrix(dataLabels)
    #dataLabels = np.squeeze(np.asarray(dataLabels))
    
    return dataValues, dataLabels[:,0], 'NOAA dataset. Eight  features. Two classes.'