# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:22:42 2015

@author: Raul Sena Ferreira
"""
import time
import nltk
import logging
from nltk.corpus import stopwords
from xml.dom.minidom import parse

#global
invIndGen = ''

def processInvertedIndexGenerator(path, vectorPath):
    begin = time.time()
    global invIndGen
    #logging instantiate
    logPath = path+'/InvertedIndexGenerator/invertedIndexGenerator.log'
    log('invertedIndexGenerator', logPath)
    invIndGen = logging.getLogger('invertedIndexGenerator')
    invIndGen.info('Processing Inverted Index Generator Module...')
    meanTimeXML = 0
    meanTimeIIG = 0
    
    invertedIndex = {}
    stop = stopwords.words('english')
    for line in vectorPath:
        if str(line[0]) == 'LEIA':
            ini = time.time()
            
            dictionary=readXML(path+str(line[1]).strip())
            
            meanTimeXML+=time.time()-ini
            ini = time.time()
            
            invertedIndex.update(invertedIndexGenerator(dictionary, stop))
            
            meanTimeIIG+=time.time()-ini
        elif str(line[0]) == 'ESCREVA':
            invIndGen.info('Writing Inverted Index on file...')
            ini = time.time()
            
            writeInvertedIndex(path+str(line[1]).strip(), invertedIndex)
            
            timeElapsed = time.time()-ini
            invIndGen.info('Write operation finished with %s' % str(timeElapsed))
            
    invIndGen.info('XML reading operation finished with %s of time average.' % str(meanTimeXML/len(line)-1))        
    invIndGen.info('Inverted Index Generator Method finished with %s of time average.' % str(meanTimeIIG/len(line)-1))
    
    end = time.time() - begin
    invIndGen.info('End of Inverted Index Generator Module. Total of %s elapsed.' % str(end))
            
            
            
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
    
    f = open(filepath, 'w+')
    
    for key in invertedIndex.keys():
        f.write(key.upper()+";%s\n" % invertedIndex[key])        
    f.close()
    
    
    
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
    
    invIndGen.info('%s records read succesfully.' % str(len(dictionary)))
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
