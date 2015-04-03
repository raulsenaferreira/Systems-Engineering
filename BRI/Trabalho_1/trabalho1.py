# -*- coding: utf-8 -*-
"""
@author: raul
"""
import math
import nltk
import os
import logging
import re
from xml.dom.minidom import parse
from pprint import pprint as pp

#globals
path = os.path.dirname(__file__)

def main():
    pathVector = []
    '''     ***** InvertedIndexGenerator *****     ''' 
    configFile = path+'/InvertedIndexGenerator/gli.cfg'
    pathVector = readData(configFile, '=')
    #log
    log(path+'/InvertedIndexGenerator/invertedIndexGenerator.log')
    processInvertedIndexGenerator(pathVector)
    
    
    '''     ****** Indexer *****    '''
    configFile = path+'/Indexer/index.cfg'
    pathVector = readData(configFile, '=')
    invertedIndex = readData(path+pathVector[0][1], ';')
    tf_idf_metric(invertedIndex)  
    #processIndexer(metric, list)


def readData(filepath, symbol):
    directory = open(filepath, 'r')
    lines=[]
    
    for line in directory:
        line = line.strip().replace(" ","")
        lines.append(line.split(symbol))
    
    directory.close()
    return lines



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
    
    for ind in invertedIndex:
        if len(ind[0]) >= minLength and regex.match(ind[0]):
            N = N | set(ind[1].split(','))
        
    N = len(N)

    for ind in invertedIndex:
        if len(ind[0]) >= minLength and regex.match(ind[0]):
            term = ind[0]
            docs = ind[1].split(',')
            diffDocs = set(docs)
            
            freq_ij = nltk.FreqDist(docs)
            maxFreq_ij = freq_ij.most_common(1)
            n_i = len(diffDocs)
            w_ij.update({term: [(freq_ij[k]/maxFreq_ij[0][1]) * math.log(N/n_i) for k in freq_ij.keys()]})
        
    return w_ij
    

    
def processInvertedIndexGenerator(vectorPath):
    dictionary = {}
    invertedIndex = {}
    
    for line in vectorPath:
        if str(line[0]) == 'LEIA':
            dictionary=readXML(path+str(line[1]))
            invertedIndex=invertedIndexGenerator(dictionary, [])
        elif str(line[0]) == 'ESCREVA':
            writeInvertedIndex(path+str(line[1]), invertedIndex)
 


def readXML(filename):
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
                pp("Document["+recordNumber+"] don't have abstract neither extract!")
    return dictionary
    


def writeInvertedIndex(filepath, invertedIndex):
    f = open(filepath, 'w+')
    
    for key in invertedIndex.keys():
        listString=','.join(str(i) for i in invertedIndex[key])
        #f.write(key.upper()+';['+listString+']\n')
        f.write(key.upper()+';'+listString+'\n')
    f.close()
    


def invertedIndexGenerator(dictionaries, stopWords):
    invertedIndex = {}
    
    for key in dictionaries.keys():
        for token in nltk.word_tokenize(dictionaries[key]):
            tokenList = []
            if token not in stopWords:
                try:
                    invertedIndex[token].append(key)
                    invertedIndex.update({token:invertedIndex[token]})
                except KeyError:
                    tokenList.append(key)
                    invertedIndex.update({token:tokenList})
    return invertedIndex



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