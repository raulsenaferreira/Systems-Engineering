# -*- coding: utf-8 -*-
"""
@author: Raul Sena Ferreira
"""
from InvertedIndex import processInvertedIndexGenerator
from Indexer import tf_idf_metric
from Indexer import writeVectorModel
from QueryProcessor import executeQueryProcessor
from Searcher import strToDict
from Searcher import makeSearch
from Searcher import writeResults
import math
from nltk.corpus import stopwords
import os
import logging


#globals
PATH = os.path.dirname(__file__)

def main():
    ''' You can execute each module separated if you want '''
    #log
    log(PATH+'/InvertedIndexGenerator/invertedIndexGenerator.log')
    
    '''     ***** InvertedIndexGenerator *****     '''
    configFile = '/InvertedIndexGenerator/gli.cfg'
    pathVector = readData(configFile, '=')
    processInvertedIndexGenerator(pathVector)
    
    '''     ****** Indexer *****    '''
    configFile = '/Indexer/index.cfg'
    pathVector = readData(configFile, '=')
    invertedIndex = readData(pathVector[0][1], ';')
    listOfWeights = tf_idf_metric(invertedIndex)
    writeVectorModel(listOfWeights, PATH+pathVector[1][1])
    
    '''     ****** QueryProcessor *****     '''
    configFile = '/QueryProcessor/pc.cfg'
    pathVector = readData(configFile, '=')
    executeQueryProcessor(pathVector)
    
    '''     ****** Searcher *****     '''
    configFile = '/Searcher/busca.cfg'
    pathVector = readData(configFile, '=')
    indexes = readData(pathVector[0][1], ';')
    indexes = strToDict(indexes)
    queries = readData(pathVector[1][1], ';')
    queries = strToDict(queries, False)
    stop = stopwords.words('english')
    rankings = makeSearch(indexes, queries, stop)
    writeResults(rankings, pathVector[2][1])
    #v1,v2 = [3, 45, 7, 2], [2, 54, 13, 15]
    #pp(cosine_similarity(v1,v2))
    
    

def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
    

    
def readData(filepath, symbol):
    directory = open(PATH+filepath.strip(), 'r')
    lines=[]
    
    for line in directory:
        line = line.strip()
        lines.append(line.split(symbol))
    
    directory.close()
    return lines
    


def log(logFile):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(logFile)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info('Logging iniated')


                      
main() 