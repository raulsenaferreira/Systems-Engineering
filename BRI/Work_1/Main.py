# -*- coding: utf-8 -*-
"""
@author: Raul Sena Ferreira
"""
from InvertedIndex import processInvertedIndexGenerator
from Indexer import processIndexer
from QueryProcessor import executeQueryProcessor
from Searcher import processSearcher
from nltk.corpus import stopwords
import os
import logging
import time

#globals
PATH = os.path.dirname(__file__)

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
    processInvertedIndexGenerator(PATH, pathVector)
    
    
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
    processSearcher(PATH, indexes, queries, pathVector, stop)
    
    
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