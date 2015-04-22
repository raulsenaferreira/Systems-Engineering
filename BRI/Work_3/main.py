# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:27:23 2015

@author: Raul Sena Ferreira
"""

import time
import nltk
import logging
import os
from pprint import pprint as pp 
from nltk.stem.porter import *
from nltk.corpus import stopwords
from xml.dom.minidom import parse
from Indexer import indexer
from Retriever import retriever

#globals
PATH = os.path.dirname(__file__)
luceneSearcher = ''
def main():
    begin = time.time()
    global luceneSearcher
    #logging instantiate
    logPath = PATH+'/luceneSearcher.log'
    log('luceneSearcher', logPath)
    luceneSearcher = logging.getLogger('luceneSearcher')
    luceneSearcher.info('Processing Lucene...')
    configFile = '/config.cfg'
    pathVector = readData(configFile, '=')
    
    for i in range(0, len(pathVector)):
        if pathVector[i][0] == "DOCS":
            dictionary=readXML(PATH+str(pathVector[i][1]).strip())
            for k in dictionary.keys():
                indexer(k, dictionary[k])
        elif pathVector[i][0] == "QUERIES":    
            queries = readData(str(pathVector[i][1]).strip(), ';')
            for q in queries:
                retriever(q[0], q[1])
    
#Utils
def readData(filepath, symbol):
    directory = open(PATH+filepath.strip(), 'r')
    lines=[]
    
    for line in directory:
        line = line.strip()
        lines.append(line.split(symbol))
    
    directory.close()
    return lines        
    
    
    
def readXML(filename):
    luceneSearcher.info('Reading '+filename+' file')
    
    dictionary = {}
    DOMTree = parse(filename)
    collection = DOMTree.documentElement
    
    records = collection.getElementsByTagName("RECORD")
    
    for record in records:
        recordNumber = record.getElementsByTagName('RECORDNUM')[0].childNodes[0].data
        
        try:
            dictionary[recordNumber] = record.getElementsByTagName('ABSTRACT')[0].childNodes[0].data
        except IndexError:
            try:
                dictionary[recordNumber] = record.getElementsByTagName('EXTRACT')[0].childNodes[0].data
            except IndexError:
                luceneSearcher.warning("Document["+recordNumber+"] doesn't have abstract neither extract!")
    
    luceneSearcher.info('%s records read succesfully.' % str(len(dictionary)))
    return dictionary
    
    
    
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
    
    
    
if __name__ == '__main__':
    main()