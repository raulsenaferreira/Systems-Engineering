import matplotlib.pyplot as plt
import numpy as np
'''
Rotating checkerboard problem with a linear drift. 
The CB rotates over a range of 0 - 2pi with 0-pi being a recurring environment at pi-2pi.
Adapted from ConceptDriftData 2011 by Gregory Ditzler, Ryan Elwell, and Robi Polikar.
Original version written by Ludmila Kuncheva.
'''
T = 4 #time steps
N = 64 #instances
a = np.linspace(0,2*np.pi,T)
side = 0.5
xTrain = {}
yTrain = {}
xTest = {}
yTest = {}

def gendatcb(N,a,alpha):
    # N data points, uniform distribution,
    # checkerboard with side a, rotated at alpha
    d = np.random.rand(N,2)
    d_transformed = [d[:,0]*np.cos(alpha)-d[:,1]*np.sin(alpha), d[:,0]*np.sin(alpha)+d[:,1]*np.cos(alpha)]
    
    s = np.ceil(d_transformed[0]/a)+np.floor(d_transformed[1]/a)
    labd = 2-np.mod(s,2)
    labd = labd - 1
    return d, labd

def CBDAT(a,alpha,N):
    X,Y = gendatcb(N,a,alpha)
    X = np.transpose(X)
    Y = np.transpose(Y)
    return X, Y

for t in range(T):
    xTrain[t],yTrain[t] = CBDAT(side,a[t],N)
    xTest[t],yTest[t] = CBDAT(side,a[t],N)
    
print(xTrain[0])
print(yTrain[0])
