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
    
    
def plot(X, y, t):
    #print data distribution in step t
    title = "Data distribution. Step {}".format(t+1)
    class0 = X[np.where(y==0)[0]]
    #print(len(X[np.where(y==0)]))
    #print(X[np.where(y==0)[0]])
    class1 = X[np.where(y==1)[0]]
    ax1 = plt.subplot(111)
    ax2 = plt.subplot(111)
    #ax3 = plt.subplot(111)
    ax1.scatter(class0[:, 0], class0[:, 1], c="b")
    ax2.scatter(class1[:, 0], class1[:, 1], c="r")
    #ax3.scatter(X[:, 0], X[:, 1], c="g")
    '''
    ax1.set_xlim([np.amin(class0[:, 0]), np.amax(class0[:, 0])])
    ax1.set_ylim([np.amin(class0[:, 1]), np.amax(class0[:, 1])])
    ax2.set_xlim([np.amin(class1[:, 0]), np.amax(class1[:, 0])])
    ax2.set_ylim([np.amin(class1[:, 1]), np.amax(class1[:, 1])])
    '''
    plt.title(title)
    plt.show()
    
    
def finalEvaluation(arrAcc):
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, 'Accuracy')