# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:22:42 2015

@author: Raul Sena Ferreira
"""
import time
import nltk
import logging
from pprint import pprint as pp 
from nltk.stem.porter import *
from nltk.corpus import stopwords
from xml.dom.minidom import parse

#global
invIndGen = ''

def processInvertedIndexGenerator(path, vectorPath, use_mode):
    begin = time.time()
    global invIndGen
    #logging instantiate
    logPath = path+'/InvertedIndexGenerator/invertedIndexGenerator.log'
    log('invertedIndexGenerator', logPath)
    invIndGen = logging.getLogger('invertedIndexGenerator')
    invIndGen.info('Processing Inverted Index Generator Module...')
    meanTimeXML = 0
    meanTimeIIG = 0
    arrayOfDictionaries = []
    
    #stop words
    stop = stopwords.words('english')
    
    for line in vectorPath:
        if str(line[0]) == 'LEIA':
            ini = time.time()
            
            dictionary=readXML(path+str(line[1]).strip())
            
            meanTimeXML+=time.time()-ini
            ini = time.time()
            
            #STEMMER OR NOSTEMMER
            if use_mode == 'STEMMER':
                stemmer = PorterStemmer()
                arrayOfDictionaries.append(ListOfTermsByFile(dictionary, stop, stemmer))
                #stemmer.stem(lists)
            elif use_mode == 'NOSTEMMER':
                arrayOfDictionaries.append(ListOfTermsByFile(dictionary, stop, None))
            else: print("Use mode undefined")
            
            
            meanTimeIIG+=time.time()-ini
        #elif str(line[0]) == 'ESCREVA':
    invIndGen.info('Writing Inverted Index on file...')
    ini = time.time()
    
    writeInvertedIndex(path+'/InvertedIndexGenerator/invertedIndexOut.csv', invertedIndexGenerator(arrayOfDictionaries))
    
    timeElapsed = time.time()-ini
    invIndGen.info('Write operation finished with %s' % str(timeElapsed))
            
    invIndGen.info('XML reading operation finished with %s of time average.' % str(meanTimeXML/len(line)-1))        
    invIndGen.info('Inverted Index Generator Method finished with %s of time average.' % str(meanTimeIIG/len(line)-1))
    
    end = time.time() - begin
    invIndGen.info('End of Inverted Index Generator Module. Total of %s elapsed.' % str(end))
            
            
            
def ListOfTermsByFile(dictionaries, stopWords, stemmer):
    invIndGen.info('Making inverted index...')
    
    arrayOfDictionaries = {}
    
    for key in dictionaries.keys():
        tokens = nltk.word_tokenize(dictionaries[key])
        for token in tokens:
            token = token.upper()
            key = key.strip()
            key = str(int(key))
            
            if token not in stopWords and len(token) > 1:
                tokenList = []
                if stemmer is not None:
                    #pp("Using stemmer...")
                    token=stemmer.stem(token)
                try:
                    arrayOfDictionaries[token].append(key)
                    arrayOfDictionaries.update({token:arrayOfDictionaries[token]})
                except KeyError:
                    tokenList.append(key)
                    arrayOfDictionaries.update({token:tokenList})
    return arrayOfDictionaries
    
    
    
def invertedIndexGenerator(arrayOfDictionaries):
    
    invertedIndex = {}
    
    for inv in arrayOfDictionaries:
        for token in inv.keys():
            try:
                invertedIndex[token]+=inv[token]
                invertedIndex.update({token:invertedIndex[token]})
            except KeyError:
                invertedIndex.update({token:inv[token]})
    
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
