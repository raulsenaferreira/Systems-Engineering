# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:36:59 2015

@author: Raul Sena Ferreira
"""
from nltk.stem.porter import *
from pprint import pprint as pp
import os
import ast

#globals
PATH = os.path.dirname(__file__)

def main():
    configFile = '/main.cfg'
    pathVector = readData(configFile, '=')
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


def strToDictResults(resultsStr):
    return 0
    
    
    
def strToDictExpectedResults(expResStr):
    expectedResults = {}
    
    pp(expectedResults)
    return expectedResults



def writeReport(pathReport):
    return 0



def compareResults(results, expectedResults):
    return 0



def readData(filepath, symbol):
    directory = open(PATH+filepath.strip(), 'r')
    lines=[]
    
    for line in directory:
        line = line.strip()
        lines.append(line.split(symbol))
    
    directory.close()
    return lines



if __name__ == '__main__':
    main() 