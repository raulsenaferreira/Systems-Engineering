# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:22:42 2015

@author: Raul Sena Ferreira
"""
import nltk
import logging
from nltk.corpus import stopwords
from xml.dom.minidom import parse

#global
invIndGen = ''

def processInvertedIndexGenerator(path, vectorPath):
    global invIndGen
    #logging instantiate
    logPath = path+'/InvertedIndexGenerator/invertedIndexGenerator.log'
    log('invIndGen', logPath)
    invIndGen = logging.getLogger('invIndGen')
    
    invIndGen.info('Processing Inverted Index Generator Module...')
    invertedIndex = {}
    stop = stopwords.words('english')
    for line in vectorPath:
        if str(line[0]) == 'LEIA':
            dictionary=readXML(path+str(line[1]).strip())
            invertedIndex.update(invertedIndexGenerator(dictionary, stop))
        elif str(line[0]) == 'ESCREVA':
            writeInvertedIndex(path+str(line[1]).strip(), invertedIndex)
    
    invIndGen.info('End of Inverted Index Generator processing.')
            
            
            
def invertedIndexGenerator(dictionaries, stopWords):
    invIndGen.info('Making inverted index...')
    
    invertedIndex = {}
    
    for key in dictionaries.keys():
        for token in nltk.word_tokenize(dictionaries[key]):
            key = key.strip()
            tokenList = []
            if token not in stopWords and len(token) > 1:
                try:
                    invertedIndex[token].append(key)
                    invertedIndex.update({token:invertedIndex[token]})
                except KeyError:
                    tokenList.append(key)
                    invertedIndex.update({token:tokenList})
    return invertedIndex
    
    
    
def writeInvertedIndex(filepath, invertedIndex):
    invIndGen.info('Writing Inverted Index on file...')
    
    f = open(filepath, 'w+')
    
    for key in invertedIndex.keys():
        f.write(key.upper()+";%s\n" % invertedIndex[key])        
    f.close()
    
    invIndGen.info('Write operation finished.')
    
    
    
def readXML(filename):
    invIndGen.info('Reading '+filename+' file')
    
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
                invIndGen.warning("Document["+recordNumber+"] doesn't have abstract neither extract!")
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
