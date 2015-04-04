# -*- coding: utf-8 -*-
"""
@author: Raul Sena Ferreira
"""
import math
import nltk
from nltk.corpus import stopwords
import os
import logging
import re
import ast
from xml.dom.minidom import parse
from pprint import pprint as pp

#globals
PATH = os.path.dirname(__file__)

def main():
    '''     ***** InvertedIndexGenerator *****     
    configFile = '/InvertedIndexGenerator/gli.cfg'
    pathVector = readData(configFile, '=')
    #log
    #log(PATH+'/InvertedIndexGenerator/invertedIndexGenerator.log')
    processInvertedIndexGenerator(pathVector)
    '''
    '''     ****** Indexer *****    
    configFile = '/Indexer/index.cfg'
    pathVector = readData(configFile, '=')
    invertedIndex = readData(pathVector[0][1], ';')
    listOfWeights = tf_idf_metric(invertedIndex)
    writeVectorModel(listOfWeights, PATH+pathVector[1][1])
    '''
    '''     ****** QueryProcessor *****     
    configFile = '/QueryProcessor/pc.cfg'
    pathVector = readData(configFile, '=')
    excuteQueryProcessor(pathVector)
    '''
    '''     ****** Searcher *****     '''
    configFile = '/Searcher/busca.cfg'
    pathVector = readData(configFile, '=')
    indexes = readData(pathVector[0][1], ';')
    indexes = strToDict(indexes)
    queries = readData(pathVector[1][1], ';')
    queries = strToDict(queries, False)
    makeSearch(indexes, queries)
    #pp(indexes)
    


def makeSearch(indexes, queries):
    return 0



def strToDict(listString, hasListInside=True):
    dictionary = {}
    if hasListInside is True:
        for s in listString:
            dictionary.update({s[0]: ast.literal_eval(s[1])})
    else:
        for s in listString:
            dictionary.update({s[0]: s[1]})
    return dictionary



def writeVectorModel(listOfWeights, filename):
    f = open(filename, 'w+')
    for k in listOfWeights.keys():
        f.write(k+";%s\n" % listOfWeights[k])
    f.close()
    
  
  
def excuteQueryProcessor(pathVector):
    queries = readQueriesXML(PATH+pathVector[0][1])
    writeQueryProcessorData(pathVector, queries)
    


def writeQueryProcessorData(pathVector, queries):
    
    for line in pathVector:
        if str(line[0]) == 'CONSULTAS':
            f = open(PATH+str(line[1]), 'w+')
            for key in queries.keys():
                f.write(key+';'+queries[key][0]+'\n')
            f.close()
        elif str(line[0]) == 'ESPERADOS':
            f = open(PATH+str(line[1]), 'w+')
            for key in queries.keys():
                f.write(key+';%s\n' % queries[key][1])
            f.close()
        


def readData(filepath, symbol):
    directory = open(PATH+filepath.strip(), 'r')
    lines=[]
    
    for line in directory:
        line = line.strip()
        lines.append(line.split(symbol))
    
    directory.close()
    return lines



def readInvertedIndex(filepath):
    directory = open(PATH+filepath.strip(), 'r')
    lines=[]
    
    for line in directory:
        line = line.strip()
        line = line.split(';')
        line = ast.literal_eval(line[1])
        lines.append(line.split(';'))
    
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
        ind[1] = ast.literal_eval(ind[1])
        if len(ind[0]) >= minLength and regex.match(ind[0]):
            N = N | set(ind[1])
        
    N = len(N)

    for ind in invertedIndex:
        if len(ind[0]) >= minLength and regex.match(ind[0]):
            term = ind[0]
            docs = ind[1]
            
            freq_ij = nltk.FreqDist(docs)
            maxFreq_ij = freq_ij.most_common(1)
            n_i = len(set(docs))
            w_ij.update({term: [{k: (freq_ij[k]/maxFreq_ij[0][1]) * math.log(N/n_i)} for k in freq_ij.keys()]})
        
    return w_ij
    

    
def processInvertedIndexGenerator(vectorPath):
    stop = stopwords.words('english')
    for line in vectorPath:
        if str(line[0]) == 'LEIA':
            dictionary=readXML(PATH+str(line[1]).strip())
            invertedIndex=invertedIndexGenerator(dictionary, stop)
        elif str(line[0]) == 'ESCREVA':
            writeInvertedIndex(PATH+str(line[1]).strip(), invertedIndex)
 


def readQueriesXML(filename):
    dictionary = {}
    
    DOMTree = parse(filename)
    collection = DOMTree.documentElement
    
    queries = collection.getElementsByTagName("QUERY")
    
    for query in queries:
        queryNumber = query.getElementsByTagName('QueryNumber')[0].childNodes[0].data
        expected = {}
        try:
            queryText = query.getElementsByTagName('QueryText')[0].childNodes[0].data
            queryText = queryText.replace('\n', '').upper()
            try:
                records = query.getElementsByTagName('Item')
                for record in records:
                    if record.hasAttribute("score"):
                        score = record.getAttribute("score")
                        key = record.childNodes[0].data
                        expected.update({key: score})
                dictionary[queryNumber] = [queryText, expected]
            except IndexError:
                pp("Document["+queryNumber+"] doesn't have records!")
        except IndexError:
            pp("Document["+queryNumber+"] doesn't have query text!")
        
    return dictionary
    
    
    
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
                pp("Document["+recordNumber+"] doesn't have abstract neither extract!")
    return dictionary
    


def writeInvertedIndex(filepath, invertedIndex):
    f = open(filepath, 'w+')
    
    for key in invertedIndex.keys():
        f.write(key.upper()+";%s\n" % invertedIndex[key])        
    f.close()
    


def invertedIndexGenerator(dictionaries, stopWords):
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