# -*- coding: utf-8 -*-
"""
@author: Raul Sena Ferreira
"""
from InvertedIndex import processInvertedIndexGenerator
from Indexer import processIndexer
from QueryProcessor import executeQueryProcessor
from Searcher import processSearcher
from Evaluator import executeEvaluator
from nltk.corpus import stopwords
import os
import logging
import time

#globals
PATH = os.path.dirname(os.path.abspath(__file__))

def main():
    ''' You can execute each module separated if you want '''

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('main.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('System initiated!')

    s = time.time()

    '''     ***** InvertedIndexGenerator *****     '''
    configFile = '/InvertedIndexGenerator/gli.cfg'
    pathVector = readData(configFile, '=')
    use_mode = pathVector[0][1]

    processInvertedIndexGenerator(PATH, pathVector, use_mode)


    '''     ****** Indexer *****    '''
    configFile = '/Indexer/index.cfg'
    pathVector = readData(configFile, '=')
    invertedIndex = readData(pathVector[0][1], ';')
    processIndexer(invertedIndex, PATH, pathVector)


    '''     ****** QueryProcessor *****     '''
    configFile = '/QueryProcessor/pc.cfg'
    pathVector = readData(configFile, '=')
    executeQueryProcessor(PATH, pathVector)


    '''     ****** Searcher *****     '''
    configFile = '/Searcher/busca.cfg'
    pathVector = readData(configFile, '=')
    indexes = readData(pathVector[0][1], ';')
    queries = readData(pathVector[1][1], ';')
    stop = stopwords.words('english')
    processSearcher(PATH, indexes, queries, pathVector, stop, use_mode)


    '''     ****** Evaluator *****     '''
    configFile = '/Evaluator/evaluator.cfg'
    pathVector = readData(configFile, '=')

    #STEMMER OR NOSTEMMER
    resultsStr=''

    if use_mode == 'NOSTEMMER':
        resultsStr = readData(pathVector[1][1], ';')
    elif use_mode == 'STEMMER':
        resultsStr = readData(pathVector[2][1], ';')
    else:
        print("Use mode undefined")
    expectedResultsString = readData(pathVector[0][1], ';')

    executeEvaluator(PATH, pathVector, expectedResultsString, resultsStr, use_mode)


    logger.info('End of System. Total of %s elapsed.' % str(time.time() - s))



# ******    Main Methods

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
