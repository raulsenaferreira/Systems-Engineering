import matplotlib.pyplot as plt
import numpy as np


def plotDistributions(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['cluster 1', 'cluster 2']
    ax = fig.add_subplot(121)
    
    for X in distributions:
        #reducing to 2-dimensional data
        x=pca(X, 2)
        
        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1
    
    ax.legend(handles, classes)
    
    plt.show()
    
    
def plotDistributionByClass(instances, indexesByClass):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['cluster 1', 'cluster 2']
    ax = fig.add_subplot(121)
    
    for c, indexes in indexesByClass.items():
        X = instances[indexes]
        #reducing to 2-dimensional data
        x=pca(X, 2)
        
        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1
    
    ax.legend(handles, classes)
    
    plt.show()
    
    
def plotAccuracy(arr, label):
    c = range(len(arr))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.legend(ax.plot(c, arr, 'k'), label)
    
    plt.grid()
    plt.show()


def plotDistributionss(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['Class 1', 'Class 2']
    ax = fig.add_subplot(121)
    
    for k, v in distributions.items():
        points = distributions[k]
        
        handles.append(ax.scatter(points[:, 0], points[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1
    
    ax.legend(handles, classes)
    
    plt.show()
    
    
def plot(X, y, coreX, coreY, t):
    fig = plt.figure()
    handles = []
    classes = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    
    class0 = X[np.where(y==0)[0]]
    class1 = X[np.where(y==1)[0]]
    coreClass0 = coreX[np.where(coreY==0)[0]]
    coreClass1 = coreX[np.where(coreY==1)[0]]
    
    handles.append(ax.scatter(class0[:, 0], class0[:, 1], c = 'r'))
    handles.append(ax.scatter(coreClass0[:, 0], coreClass0[:, 1], c = 'g'))
    handles.append(ax.scatter(class1[:, 0], class1[:, 1], c = 'b'))
    handles.append(ax.scatter(coreClass1[:, 0], coreClass1[:, 1], c = 'y'))
    
    ax.legend(handles, classes)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()
    
    
def finalEvaluation(arrAcc):
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, 'Accuracy')