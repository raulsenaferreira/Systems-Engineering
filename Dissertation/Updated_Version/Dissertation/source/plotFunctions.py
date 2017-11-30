import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from source import classifiers



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
        x=classifiers.pca(X, 2)

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
        x=classifiers.pca(X, 2)

        handles.append(ax.scatter(x[:, 0], x[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1

    ax.legend(handles, classes)

    plt.show()


def plotAccuracy(arr, steps, label):
    arr = np.array(arr)
    c = range(len(arr))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, arr, 'k')
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(0, steps+1, 10))
    plt.title(label)
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
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
    classes = list(set(y))
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = X[np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color]))
        #core support points
        color+=1
        corePoints = coreX[np.where(coreY==cl)[0]]
        coreX1 = corePoints[:,0]
        coreX2 = corePoints[:,1]
        handles.append(ax.scatter(coreX1, coreX2, c = colors[color]))
        #labels
        classLabels.append('Class {}'.format(cl))
        classLabels.append('Core {}'.format(cl))
        color+=1

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def plotAnimation(i):
    classes = list(set(arrY[i]))
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = arrX[i][np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color]))
        #core support points
        color+=1
        corePoints = coreX[np.where(coreY==cl)[0]]
        coreX1 = corePoints[:,0]
        coreX2 = corePoints[:,1]
        handles.append(ax.scatter(coreX1, coreX2, c = colors[color]))
        #labels
        classLabels.append('Class {}'.format(cl))
        classLabels.append('Core {}'.format(cl))
        color+=1

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def plot2(X, y, t, classes):
    X = classifiers.pca(X, 2)
    fig = plt.figure()
    handles = []
    classLabels = []
    cmx = plt.get_cmap('Paired')
    colors = cmx(np.linspace(0, 1, (len(classes)*2)+1))
    #classLabels = ['Class 1', 'Core 1', 'Class 2', 'Core 2']
    ax = fig.add_subplot(111)
    color=0
    for cl in classes:
        #points
        points = X[np.where(y==cl)[0]]
        x1 = points[:,0]
        x2 = points[:,1]
        handles.append(ax.scatter(x1, x2, c = colors[color]))
        #core support points
        color+=1
        #labels
        classLabels.append('Class {}'.format(cl))

    ax.legend(handles, classLabels)
    title = "Data distribution. Step {}".format(t)
    plt.title(title)
    plt.show()


def finalEvaluation(arrAcc, steps, label):
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc)**2)
    plotAccuracy(arrAcc, steps, label)


def plotBoxplot(mode, data, labels):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=90)

    if mode == 'acc':
        plt.title("Accuracy - Boxplot")
        #plt.xlabel('step (s)')
        plt.ylabel('Accuracy')
    elif mode == 'mcc':
        plt.title('Mathews Correlation Coefficient - Boxplot')
        plt.ylabel("Mathews Correlation Coefficient")
    elif mode == 'f1':
        plt.title('F1 - Boxplot')
        plt.ylabel("F1")

    plt.show()


def plotAccuracyCurves(listOfAccuracies, listOfMethods):
    limit = len(listOfAccuracies[0])+1

    for acc in listOfAccuracies:
        acc = np.array(acc)
        c = range(len(acc))
        ax = plt.axes()
        ax.plot(c, acc)

    plt.title("Aggregated accuracy curve")
    plt.legend(listOfMethods)
    plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
    plt.xticks(range(0, limit, 10))
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotBars(listOfTimes, listOfMethods):
    
    for l in range(len(listOfTimes)):    
        ax = plt.axes()
        ax.bar(l, listOfTimes[l], label=listOfMethods[l], align='center')

    plt.title("Execution time to perform all stream")
    plt.legend(listOfMethods)
    plt.xlabel("Methods")
    plt.ylabel("Execution time")
    plt.xticks(range(len(listOfTimes)))
    plt.show()


def plotBars2(listOfTimes, listOfMethods):
    
    for l in range(len(listOfTimes)):    
        ax = plt.axes()
        ax.bar(l, listOfTimes[l], label=listOfMethods[l], align='center')

    plt.title("Average Accuracy")
    plt.legend(listOfMethods)
    plt.xlabel("Methods")
    plt.ylabel("Accuracy")
    plt.xticks(range(len(listOfTimes)))
    plt.show()