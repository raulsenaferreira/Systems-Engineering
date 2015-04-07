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

#globals
PATH = os.path.dirname(__file__)
evaluatorLog = ''

def main():
    begin = time.time()
    global evaluatorLog
    #logging instantiate
    logPath = path+'/Evaluator/evaluator.log'
    log('evaluator', logPath)
    evaluatorLog = logging.getLogger('evaluator')
    evaluatorLog.info('Processing Evaluator module...')
    
    configFile = '/Evaluator/evaluator.cfg'
    pathVector = readData(PATH+configFile, '=')
    #STEMMER OR NOSTEMMER    
    use_mode = pathVector[0][1]
    
    expectedResultsString = readData(PATH+pathVector[1][1], ';')
    expectedResults = strToDictExpectedResults(expectedResultsString)
    
    resultsStr = ''
    if use_mode == 'NOSTEMMER':
        resultsStr = readData(PATH+pathVector[2][1], ';')
    elif use_mode == 'STEMMER':
        resultsStr = readData(PATH+pathVector[3][1], ';')
    else:
        print("Use mode undefined")
    results = strToDictResults(resultsStr)
    
    writeReport(PATH+pathVector[4][1])
    
    stemmer = PorterStemmer()
    #stemmer.stem(lists)
    end = time.time() - begin
    evaluatorLog.info('End of Evaluator Module. Total of %s elapsed.' % str(end))


def strToDictResults(resultsStr):
    results = {}
    for r in resultsStr:
        l = r[1].lstrip('[').rstrip(']').split('],[')
        results.update({r: l})
    return results
    
    
    
def strToDictExpectedResults(expResStr):
    expectedResults = {}
    for e in expResStr:
        expectedResults.update({e[0]: ast.literal_eval(e[1])})
    return expectedResults



def writeReport(pathReport):
    return 0



def compareResults(results, expectedResults):
    return 0



def readData(filepath, symbol):
    directory = open(filepath.strip(), 'r')
    lines=[]
    
    for line in directory:
        line = line.strip()
        lines.append(line.split(symbol))
    
    directory.close()
    return lines



if __name__ == '__main__':
    main() 