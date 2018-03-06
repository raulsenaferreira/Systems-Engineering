import numpy as np
from source import classifiers
from source import metrics
from source import util
from source import plotFunctions
from timeit import default_timer as timer


def run(dataValues, dataLabels, datasetDescription, isBinaryClassification, isImbalanced, experiments, batches, labeledData, isBatchMode, poolSize, externalResults={}):
    listOfAccuracies = []
    listOfMethods = []
    listOfMCCs = []
    listOfF1s = []
    listOfTimeExecutions = []
    avgAccuracies = []
    
    sizeOfBatch = int((len(dataLabels)-labeledData)/batches)
    arrYt = dataLabels[labeledData:]
    arrYt = [ arrYt[i::batches] for i in range(batches) ]
    
    print(datasetDescription)
    if isBatchMode:
        print("{} batches of {} instances".format(batches, sizeOfBatch))
    else:
        print("Stream mode with pool size = {}".format(poolSize))
    print("\n\n")

    F1Type = 'macro' #for balanced datasets with 2 classes
    if isImbalanced or not isBinaryClassification:
        F1Type='micro'
    
    for name, e in experiments.items():
        try:
            CoreX = []
            CoreY = []
            accTotal = []
            accuracies=[]
            classes = list(set(dataLabels))#getting all possible classes in data

            start = timer()
            #accuracy per step
            algorithmName, accuracies, CoreX, CoreY, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted = e.method.start(
                dataValues=dataValues, dataLabels=dataLabels, classes=classes, densityFunction=e.densityFunction, 
                batches=batches, sizeOfBatch = sizeOfBatch, initialLabeledData=labeledData, excludingPercentage=e.excludingPercentage, 
                K_variation=e.K_variation, clfName=e.clfName, poolSize=poolSize, isBatchMode=isBatchMode)
            end = timer()
            averageAccuracy = np.mean(accuracies)

            #elapsed time per step
            elapsedTime = end - start
            
            accTotal.append(averageAccuracy)

            arrF1 = metrics.F1(arrYt, arrPredicted, F1Type)
            listOfAccuracies.append(accuracies)
            listOfMethods.append(algorithmName)
            listOfF1s.append(arrF1)
            listOfTimeExecutions.append(elapsedTime)
            
            print("Execution time: ", elapsedTime)
            
            if isBinaryClassification:
                arrMCC = metrics.mcc(arrYt, arrPredicted)
                listOfMCCs.append(arrMCC)
                print("Average MCC: ", np.mean(arrMCC))

            print("Average error:", 100-averageAccuracy)
            print("Average {}-F1: {}".format(F1Type, np.mean(arrF1)))
            plotFunctions.finalEvaluation(accuracies, batches, algorithmName)
            plotFunctions.plotF1(arrF1, batches, algorithmName)
            avgAccuracies.append(np.mean(accuracies))
            
            #print data distribution in step t
            initial = (batches*sizeOfBatch)-sizeOfBatch
            final = initial + sizeOfBatch
            #plotFunctions.plot(dataValues[initial:final], dataLabels[initial:final], CoreX, CoreY, batches)
            print("\n\n")
        except Exception as e:
            print(e)
            raise e
        
    
    # Beginning of external results plottings
    for extResult in externalResults:
        print("Method: {}".format(extResult['name']))
        print("Execution time: ", elapsedTime)
        
        if isBinaryClassification:
            MCCs = metrics.mcc(arrYt, extResult['predictions'])
            print("Average MCC: ", np.mean(MCCs))
            listOfMCCs.append(MCCs)

        arrF1External = metrics.F1(arrYt, extResult['predictions'], F1Type)
        print("Average {}-F1: {}".format(F1Type, np.mean(arrF1External)))

        plotFunctions.finalEvaluation(extResult['accuracies'], batches, extResult['name'])
        plotFunctions.plotF1(arrF1External, batches, extResult['name'])
        
        listOfMethods.append(extResult['name'])
        listOfAccuracies.append(extResult['accuracies'])
        
        listOfF1s.append(arrF1External)
        listOfTimeExecutions.append(extResult['time'])
        avgAccuracies.append(np.mean(extResult['accuracies']))
    # End of external results plottings

    plotFunctions.plotBoxplot('acc', listOfAccuracies, listOfMethods)
    
    if isBinaryClassification:
        plotFunctions.plotBoxplot('mcc', listOfMCCs, listOfMethods)

    plotFunctions.plotBoxplot('f1', listOfF1s, listOfMethods)
    plotFunctions.plotAccuracyCurves(listOfAccuracies, listOfMethods)
    plotFunctions.plotBars(listOfTimeExecutions, listOfMethods)
    plotFunctions.plotBars2(avgAccuracies, listOfMethods)
    plotFunctions.plotBars3(avgAccuracies, listOfMethods)
    plotFunctions.plotBars4(avgAccuracies[0], avgAccuracies, listOfMethods)