# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:36:59 2015

@author: Raul Sena Ferreira
"""
from nltk.stem.porter import *
from pprint import pprint as pp
import os
import ast
import logging
import time
import math
import numpy as np
#from sklearn.metrics import metrics

#globals
PATH = os.path.dirname(__file__)
evaluatorLog = ''

#def main():
def executeEvaluator(PATH):
    path=PATH
    configFile = '/Evaluator/evaluator.cfg'
    pathVector = readData(configFile, '=')
    begin = time.time()
    global evaluatorLog
    #logging instantiate
    logPath = path+'/Evaluator/evaluator.log'
    log('evaluator', logPath)
    evaluatorLog = logging.getLogger('evaluator')
    evaluatorLog.info('Processing Evaluator module...')
    
    configFile = '/Evaluator/evaluator.cfg'
    pathVector = readData(configFile, '=')
    #STEMMER OR NOSTEMMER    
    use_mode = pathVector[0][1]
    
    expectedResultsString = readData('/'+pathVector[0][1], ';')
    expectedResults = strToDictExpectedResults(expectedResultsString)
    use_mode = 'STEMMER'
    resultsStr = ''
    if use_mode == 'NOSTEMMER':
        resultsStr = readData(pathVector[1][1], ';')
    elif use_mode == 'STEMMER':
        resultsStr = readData(pathVector[2][1], ';')
    else:
        print("Use mode undefined")
        
    results = strToDictResults(resultsStr)
    
    relevanceList = selectRelevantDocs(expectedResults)
    #pp(relevanceList)
    graphic11points(results, relevanceList)
    #executing metrics
    #k=10 documents
    #PK=[precisionK(results[query], relevanceList[query]) for query in expectedResults.keys() if precisionK(results[query], relevanceList[query]) is not None]#doc 1234 in query 00040 with problem
    #pp(PK)    
    #AVP=[averagePrecision(results[query], relevanceList[query]) for query in expectedResults.keys()]
    #pp(AVP)
    #MAP=meanAveragePrecision(AVP)
    #pp(MAP)
    #DCG=discountedCumulativeGain(results, expectedResults, range(4,8))
    #pp(DCG)
    #nDCG=normalizedDiscountedCumulativeGain(DCG, results, expectedResults, range(4,8))
    #pp(nDCG)
    #f1=[F1(results[query], relevanceList[query], query) for query in expectedResults.keys()]
    #pp(f1)    
    #writeReport(path+pathVector[4][1])
    
    #stemmer = PorterStemmer()
    #stemmer.stem(lists)
    end = time.time() - begin
    evaluatorLog.info('End of Evaluator Module. Total of %s elapsed.' % str(end))
    


def graphic11points(results, relevanceList):
    globalRelevants = set()
    globalResults = set()
    arrayPoints = []
    
    for k in relevanceList:
        for doc in relevanceList[k]:
            globalRelevants.add(doc)
        
    kRels = math.ceil(len(globalRelevants)/10)
    cont=kRels

    for key in results:
        for res in results[key]:
            globalResults.add(res[1])
    
    nDocs = 0
    nRels = 0
    vetAux = []
    for res in globalResults:
        nDocs+=1
        if res in globalRelevants:
            nRels+=1
            vetAux.append(nRels/nDocs)
            if nRels == cont:
                arrayPoints.append(max(vetAux))
                vetAux = []
                cont+=kRels
            
    if len(vetAux) > 0:
        arrayPoints.append(nRels/nDocs)
            
    pp(arrayPoints)

    

def selectRelevantDocs(expectedResults, minRelevanceScore=4):
    relevantsByQuery = {}
    
    for query in expectedResults.keys():
        relevants=[]
        for doc in expectedResults[query]:
            if doc[1] >= minRelevanceScore:
                relevants.append(doc[0])
        relevantsByQuery.update({query: relevants})
    return relevantsByQuery



#measures methods
def precisionK(results, relevants, k=10):
    numDocs=k
    rels=0
    
    for doc in results:
        if k > 0:
            k-=1
            if doc[1] in relevants:
                rels+=1
        else:
            return rels/numDocs



def averagePrecision(results, relevants):
    
    relevantsOfK=[]
    total=0
    p=0
    
    for docResults in results:
        total+=1
        if docResults[1] in relevants:
            relevantsOfK.append(docResults[1])
            try: p+=precisionK(results, relevantsOfK, total)
            except TypeError: pp("")
    if len(relevantsOfK) > 0:
        p /= len(relevantsOfK)
        return p
    else:
        return 0
        
        
    
def meanAveragePrecision(avgPrecisionVector):
    return sum(avgPrecisionVector)/len(avgPrecisionVector)
    
    
    
def discountedCumulativeGain(results, expectedResults, relevanceScale=range(0,8)):
    arrayDCG = {}
    for query in expectedResults.keys():
        relevants={}
        
        for doc in expectedResults[query]:
            if doc[1] in relevanceScale:
                relevants.update({doc[0]: doc[1]})
        
        DCG=0
        for doc in results[query]:
            
            if doc[1] in relevants:
                rank = float(doc[0])
                if rank > 1:
                    logRank = math.log(rank, 2)
                else:
                    logRank = 1
                
                scale = float(relevants[doc[1]])
                DCG += scale/logRank
        arrayDCG.update({query: DCG})
    return arrayDCG
    
    
    
def normalizedDiscountedCumulativeGain(dcg, results, expectedResults, relevanceScale=range(0,8)):
    arrayIDCG = {}
    for query in expectedResults.keys():
        relevants={}
        
        for doc in expectedResults[query]:
            if doc[1] in relevanceScale:
                relevants.update({doc[0]: doc[1]})
        relevants=sorted(relevants.items(),key=operator.itemgetter(1), reverse=True)
        #pp(relevants) 
        DCG=0
        for doc in results[query]:
            if doc[1] in relevants:
                rank = float(doc[0])
                if rank > 1:
                    logRank = math.log(rank, 2)
                else:
                    logRank = 1
                
                scale = float(relevants[doc[1]])
                DCG += scale/logRank
        arrayIDCG.update({query: DCG})
    return arrayIDCG



def F1(results, relevants, query):
    f1List={}
    precision=0
    recall=0
    rel=0
    nDoc=0
    nRel=len(relevants)
    
    for doc in results:
        nDoc+=1
        if doc[1] in relevants:
            rel+=1
            
    if rel > 0:
        precision=rel/nDoc
        recall=rel/nRel
        f1 = 2 * ((precision*recall)/(precision+recall))
        f1List.update({query: f1})
    else:
        f1List.update({query: 0})

    return f1List
    
    
    
def writeReport(pathReport):
    return 0



def compareResults(results, expectedResults):
    return 0



def interpolated_precision_recall_curve(queries_ranking, queries_similarities, relevants):
    
    queries_count = np.shape(queries_ranking)[0]
    interpolated_precision = np.zeros(11,dtype = np.float128) 
    
    for qindex in range(0,queries_count):

        tp = 0
        precision, recall = [0],[0]
        
        relevants_count = np.shape(np.nonzero(relevants[qindex]))[1]
        retrieved_count = 1
        
        for ranki in queries_ranking[qindex]:
            if (queries_similarities[qindex][ranki] > 0) and (relevants[qindex][ranki] == 1):
                tp += 1
                
            precisioni = tp / retrieved_count
            if relevants_count == 0:
                recalli = 1
            else:
                recalli = tp / relevants_count
            
            retrieved_count += 1

            precision += [precisioni]
            recall += [recalli]
              
        # query's 11 levels of precision recall precision_levels[0] = max precision in recall > 0                  
        precision_levels = []
        
        for rank in range(0,11):
            prec_ati = 0
            for j in range(0,len(recall)):
                if rank <= recall[j]*10:
                    prec_ati =  max(prec_ati,precision[j])
                    
            precision_levels.append(prec_ati)
            interpolated_precision[rank] += prec_ati/queries_count
            
        del precision
        del recall
                 
    
    auc = float("{0:1.4f}".format(metrics.auc([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],interpolated_precision)))
    
    return interpolated_precision, auc
    
    
    
#utils
def strToDictResults(resultsStr):
    results = {}
    for r in resultsStr:
        lstString = r[1].lstrip('[').rstrip(']').split('],[')
        lst=[]
        for l in lstString:
            lst.append(l.replace(' ','').split(','))
        results.update({r[0]: lst})
    return results
    
    
    
def strToDictExpectedResults(expResStr):
    expectedResults = {}
    for e in expResStr:
        expectedResults.update({e[0]: ast.literal_eval(e[1])})
    return expectedResults
    
    
    
def readData(filepath, symbol):
    directory = open(filepath.strip(), 'r')
    lines=[]
    
    for line in directory:
        line = line.strip()
        lines.append(line.split(symbol))
    
    directory.close()
    return lines



def log(name, logFile):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(logFile)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(streamHandler)
    
    
    
def readData(filepath, symbol):
    directory = open(PATH+filepath.strip(), 'r')
    lines=[]
    
    for line in directory:
        line = line.strip()
        lines.append(line.split(symbol))
    
    directory.close()
    return lines
    

executeEvaluator(PATH)