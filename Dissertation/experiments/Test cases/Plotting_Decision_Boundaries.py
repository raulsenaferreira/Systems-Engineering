import sys
import os
from pathlib import Path
os.chdir(Path(os.getcwd()).resolve().parents[1])

import matplotlib.pyplot as plt
from matplotlib import animation
from source import plotFunctions
from timeit import default_timer as timer
import numpy as np
import setup
from source import metrics
from methods import sliding_svm
from methods import sliding_knn
from methods import sliding_random_forest
from methods import static_rf
from methods import proposed_gmm_decision_boundaries
from methods import improved_intersection
from methods import compose
from methods import compose_gmm_version
from methods import fast_compose
from methods import intersection
from methods import testing



class Experiment():
    def __init__(self, method, densityFunction='gmm'):
        #commom for all experiments
        self.method = method
        #self.initialLabeledDataPerc=0.05 #150 instances for keystroke database and 0.05 % for artificial databases
        #self.classes=[0, 1]
        self.usePCA=False
        #used only by gmm / kde process
        self.densityFunction=densityFunction
        self.excludingPercentage = 0.75
        self.K_variation = 3
        self.classifier='cluster_and_label'
        #used in alpha-shape version only
        self.CP=0.65
        self.alpha=0.5
        #used in kmeans_svm and compose only
        self.useSVM=False
        self.isImbalanced=False


def doExperiments(dataValues, dataLabels, datasetDescription, arrAccSCARGC, finalAccSCARGC, experiments, numberOfTimes, batches, labeledData):
    listOfAccuracies = []
    listOfMethods = []
    sizeOfBatch = int((len(dataLabels)-labeledData)/batches)#int(len(dataLabels)/batches)
    arrX = []
    arrClf = []
    arrY = []
    arrUt = [] 
    arrYt = []
    
    print(datasetDescription)
    print("{} batches of {} instances".format(batches, sizeOfBatch))
    print("\n\n")
    
    for name, e in experiments.items():
        CoreX = []
        CoreY = []
        elapsedTime = []
        accTotal = []
        accuracies=[]
        classes = list(set(dataLabels))#getting all possible classes existent in data
        e.sizeOfBatch = sizeOfBatch
        e.batches = batches
        e.dataLabels = dataLabels
        e.dataValues = dataValues
        e.clfName = 'knn' #rf = random forests, cl = cluster and label, knn = k-nn, svm = svm

        for i in range(numberOfTimes):
            start = timer()
            #accuracy per step
            algorithmName, accuracies, CoreX, CoreY, arrX, arrY, arrUt, arrYt, arrClf = e.method.start(dataValues=e.dataValues, dataLabels=e.dataLabels, usePCA=e.usePCA, classes=classes, classifier=e.classifier, densityFunction=e.densityFunction, batches=e.batches, sizeOfBatch = e.sizeOfBatch, initialLabeledData=labeledData, excludingPercentage=e.excludingPercentage, K_variation=e.K_variation, CP=e.CP, alpha=e.alpha, clfName=e.clfName , useSVM=e.useSVM, isImbalanced=e.isImbalanced)
            end = timer()
            averageAccuracy = np.mean(accuracies)

            #elapsed time per step
            elapsedTime.append(end - start)
            
            accTotal.append(averageAccuracy)
        
        listOfAccuracies.append(accuracies)
        listOfMethods.append(algorithmName)
        #print("Total of ", numberOfTimes, " experiment iterations with an average accuracy of ", np.mean(accTotal))
        print("Average execution time: ", np.mean(elapsedTime))
        metrics.finalEvaluation(accuracies, batches)
    
        #print data distribution in step t
        initial = (batches*sizeOfBatch)-sizeOfBatch
        final = initial + sizeOfBatch
        plotFunctions.plot(dataValues[initial:final], dataLabels[initial:final], CoreX, CoreY, batches)
        print("\n\n")
    '''
    print("SCARGC")
    metrics.finalEvaluation(arrAccSCARGC, batches)
    listOfAccuracies.append(arrAccSCARGC)
    listOfMethods.append("SCARGC")
    
    plotFunctions.plotBoxplot(listOfAccuracies, listOfMethods)'''
    startAnimation(arrX, arrY, arrUt, arrYt, arrClf)
    
        
def accSCARGC(path, sep, key, steps):
    steps = 8
    resultsSCARGC_1, resultsSCARGC_2 = setup.loadSCARGCBoxplotResults(path, sep)
    res = resultsSCARGC_1[key]
    res = [ res[i::steps] for i in range(steps) ]
    arrAcc = []
    for i in range(steps):
        arrAcc.append(sum(res[i])/len(res[i])*100)
        #print(r[i])
    #print(sum(r)/steps)
    finalAcc = sum(arrAcc)/steps
    return arrAcc, finalAcc



def startAnimation(arrX, arrY, arrUt, arrYt, arrClf):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    #ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    #line, = ax.plot([], [], lw=2)
    #fig, ax = plt.subplots()
    
    # initialization function: plot the background of each frame
    def init():
        '''X = arrX[0]
        y = arrY[0]
        clf = arrClf[0]
        #decision boundaries
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        contour = plt.contourf(xx, yy, Z, alpha=0.4)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')'''
        scatter = plt.scatter([], [], s=20, edgecolor='k')
        return scatter,
        
    
    # animation function.  This is called sequentially
    def animate(i):
        X = arrX[i]
        y = arrY[i]
        clf = arrClf[i]
        Ut = arrUt[i]
        yt = arrYt[i] #gabarito
        
        #decision boundaries
        x_min, x_max = Ut[:, 0].min() - 1, Ut[:, 0].max() + 1
        y_min, y_max = Ut[:, 1].min() - 1, Ut[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        contour = plt.contourf(xx, yy, Z, alpha=0.4)
        scatter = plt.scatter(Ut[:, 0], Ut[:, 1], c=yt, s=30)
        cores = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, marker ='v', edgecolor='k')
        plt.title("Step {}".format(i+1))
        #plt.show()
        return scatter,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=8, interval=2000, blit=True)
    anim.save('results/keystroke_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()
    
    
    
def main():
    experiments = {}
    is_windows = sys.platform.startswith('win')
    sep = '\\'

    if is_windows == False:
        sep = '/'

    path = os.getcwd()+sep+'data'+sep
    
    steps = 100
    arrAccSCARGC, finalAccSCARGC = accSCARGC(path, sep, 'keystroke', steps)
    
    #sinthetic
    dataValues, dataLabels, description = setup.loadKeystroke(path, sep)
    

    '''
    Paper: Core  Support  Extraction  for  Learning  from  Initially  Labeled Nonstationary  Environments  using  COMPOSE
    link: http://s3.amazonaws.com/academia.edu.documents/45784667/2014_-_Core_Support_Extraction_for_Learning_from_Initially_Labeled_NSE_using_COMPOSE_-_IJCNN.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1489296600&Signature=9Z5DQZeDxcCtHUw7445uELSkgBg%3D&response-content-disposition=inline%3B%20filename%3DCore_support_extraction_for_learning_fro.pdf
    '''
    #experiments[0] = Experiment(compose_gmm_version)

    '''
    Original compose (alpha-shape version)
    '''
    #experiments[1] = Experiment(compose)

    '''
    SVM / Random Forest
    '''
    #experiments[2] = Experiment(static_svm)
    #experiments[3] = Experiment(sliding_knn)

    ''' Proposed Method 1 (density function core extraction) '''
    experiments[2] = Experiment(proposed_gmm_decision_boundaries, "gmm", 159, True))
    #experiments[4] = Experiment(proposed_gmm_core_extraction, "kde")

    '''
    Proposed method 2 (Intersection between two distributions with Batacharrya distance + density function)
    '''
    #experiments[5] = Experiment(intersection, "gmm")
    #experiments[6] = Experiment(intersection, "kde")
    
    '''
    Proposed method 3 (GMM All instances)
    '''
    #experiments[6] = Experiment(testing)

    #params: X, y, method, num of experiment repetitions, num of batches
    doExperiments(dataValues, dataLabels, description, arrAccSCARGC, finalAccSCARGC, experiments, 1, steps, 150)



if __name__ == "__main__":
    main()