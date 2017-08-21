import numpy as np
import pandas as pd
import checkerboard
 

#artificial datasets                
def loadCDT(path, sep):
    description = "One Class Diagonal Translation. 2 Dimensional data."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CDT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def loadCHT(path, sep):
    description = "One Class Horizontal Translation. 2 Dimensional data."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CHT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def load2CDT(path, sep):
    description = "Two Classes Diagonal Translation. 2 Dimensional data"
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'2CDT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def load2CHT(path, sep):
    description = "Two Classes Horizontal Translation. 2 Dimensional data."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'2CHT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def loadUG_2C_2D(path, sep):
    description = "Two Bidimensional Unimodal Gaussian Classes." 
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'UG_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def loadUG_2C_3D(path, sep):
    description = "Artificial Two 3-dimensional Unimodal Gaussian Classes."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'UG_2C_3D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 3]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:3]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def loadUG_2C_5D(path, sep):
    description = "Two 5-dimensional Unimodal Gaussian Classes."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'UG_2C_5D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 5]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:5]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def loadMG_2C_2D(path, sep):
    description = "Two Bidimensional Multimodal Gaussian Classes."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'MG_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def loadFG_2C_2D(path, sep):
    description = "Two Bidimensional Classes as Four Gaussians."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'FG_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def loadGEARS_2C_2D(path, sep):
    description = "Two Rotating Gears (Two classes. Bidimensional)."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'GEARS_2C_2D.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def loadCSurr(path, sep):
    description = "One Class Surrounding another Class. Bidimensional."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CSurr.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels, description


def load5CVT(path, sep):
    description = "Five Classes Vertical Translation. Bidimensional."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'5CVT.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), description


def load4CR(path, sep):
    description = 'Four Classes Rotating Separated. Bidimensional.'

    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CR.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), description


def load4CRE_V1(path, sep):
    description = 'Four Classes Rotating with Expansion V1. Bidimensional.'

    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CRE-V1.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataLabels)
    
    return dataValues, dataLabels.astype(int), description


def load4CRE_V2(path, sep):
    description = 'Four Classes Rotating with Expansion V2. Bidimensional.'

    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CRE-V2.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), description

def load4CE1CF(path, sep):
    description = 'Four Classes Expanding and One Class Fixed. Bidimensional.'

    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CE1CF.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), description

def loadCheckerBoard(path, sep, T=100, N=2000):
    description = 'Rotated checkerboard dataset. Rotating 2*PI.'

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
    
    return dataValues, dataLabels, description

#real datasets
def loadKeystroke(path, sep):
    description = 'Keyboard patterns database. 10 features. 4 classes.'

    dataValues = pd.read_csv(path+'real'+sep+'keystroke'+sep+'keystroke.txt',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 10]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:10]
    #print(dataValues)
    
    return dataValues, dataLabels.astype(int), description


def loadElecData(path, sep):
    description = 'Electricity data. 7 features. 2 classes.'

    dataValues = pd.read_csv(path+'real'+sep+'elecdata'+sep+'elec_data.csv',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = pd.read_csv(path+'real'+sep+'elecdata'+sep+'elec_label.csv',sep = ",")
    dataLabels = pd.DataFrame.as_matrix(dataLabels)
    
    #print(dataValues)
    return dataValues, dataLabels[:,0], description


def loadNOAADataset(path, sep):
    description = 'NOAA dataset. Eight  features. Two classes.'

    #Test sets: Predicting 365 instances by step. 50 steps. Two classes.
    '''
    NOAA dataset:
    Eight  features  (average temperature, minimum temperature, maximum temperature, dew
    point,  sea  level  pressure,  visibility,  average wind speed, maximum  wind  speed)
    are  used  to  determine  whether  each  day  experienced  rain  or no rain.
    '''
    dataValues = pd.read_csv(path+'real'+sep+'noaa'+sep+'noaa_data.csv',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = pd.read_csv(path+'real'+sep+'noaa'+sep+'noaa_label.csv',sep = ",")
    dataLabels = pd.DataFrame.as_matrix(dataLabels)
    #dataLabels = np.squeeze(np.asarray(dataLabels))
    
    return dataValues, dataLabels[:,0], description


def loadSCARGCBoxplotResults(path, sep):
    description = 'Results from SCARGC algorithm (for boxplot and accuracy timelime).'
    path = path+'results_scargc'+sep+'setting_1'+sep
    arrFiles = ['1CDT', '1CHT', '1CSurr', '2CDT', '2CHT', '4CE1CF', '4CR', '4CRE-V1', '4CRE-V2', '5CVT', 'FG_2C_2D', 'GEARS_2C_2D', 'MG_2C_2D', 'UG_2C_2D', 'UG_2C_3D', 'UG_2C_5D', 'keystroke']
    for i in range(len(arrFiles)):
        
    dataValues = pd.read_csv(+'elec_data.csv',sep = ",")
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = pd.read_csv(path+'real'+sep+'elecdata'+sep+'elec_label.csv',sep = ",")
    dataLabels = pd.DataFrame.as_matrix(dataLabels)
    
    #print(dataValues)
    return dataValues, dataLabels[:,0], description