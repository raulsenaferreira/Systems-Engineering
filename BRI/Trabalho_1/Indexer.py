# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:33:52 2015

@author: Raul Sena Ferreira
"""
import re
import ast
import nltk
import math
import logging
import time

#global
indexer = ''

def processIndexer(invertedIndex, path, pathVector):
    begin = time.time()
    global indexer
    #logging instantiate
    logPath = path+'/Indexer/indexer.log'
    log('indexer', logPath)
    indexer = logging.getLogger('indexer')
    indexer.info('Processing Indexer module...')
    indexer.info('Using TF_IDF Metric to calculate weights...')
    
    ini = time.time()
    
    listOfWeights = tf_idf_metric(invertedIndex)
    
    timeElapsed = time.time()-ini
    indexer.info('Total time of the weight calculus: %s' % str(timeElapsed))
    indexer.info('Writing Vector Model on file.')
    ini = time.time()
    
    writeVectorModel(listOfWeights, path+pathVector[1][1])
    timeElapsed = time.time()-ini
    indexer.info('Total time to write Vector Model on file: %s' % str(timeElapsed))
    
    end = time.time() - begin
    indexer.info('End of Indexer Module. Total of %s elapsed.' % str(end))



def tf_idf_metric(invertedIndex, minLength=2, regex="^[A-Z]+$"):
    '''
    regex by defult only allows uppercase ASCII words
    minimun length of word by default to be analyzed = 2
    N = total of collections
    n_i = number of documents where the ki term occurs
    freq_ij = frequence of ki term in dj document
    
    tf = (freq_ij[k]/maxFreq_ij[0][1])
    idf = math.log(N/n_i)
    w_ij = term-document weight, e.g. w_ij = tf * idf
    '''
    regex = re.compile(regex)
    N = set()
    w_ij = {}
    invInd = invertedIndex
    for ind in invInd:
        ind[1] = ast.literal_eval(ind[1])
        if len(ind[0]) >= minLength and regex.match(ind[0]):
            N = N | set(ind[1])
        
    N = len(N)#1183 documents with abstrac or extract
    #N = 1239
    for ind in invertedIndex:
        if len(ind[0]) >= minLength and regex.match(ind[0]):
            term = ind[0]
            docs = ind[1]
            w = {}
            freq_ij = nltk.FreqDist(docs)
            #maxFreq_ij = freq_ij.most_common(1)
            n_i = len(set(docs))
            for k in freq_ij.keys():
                #r = (freq_ij[k]/maxFreq_ij[0][1]) * math.log(N/n_i, 10)
                r = (1+math.log(freq_ij[k], 10)) * math.log(N/n_i, 10)
                w.update({k: r})
            w_ij.update({term: w})
    return w_ij
    
    
    
def writeVectorModel(listOfWeights, filename):
    
    f = open(filename, 'w+')
    for k in listOfWeights.keys():
        f.write(k+";%s\n" % listOfWeights[k])
    f.close()
    
    
    
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