# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:22:42 2015

@author: raul
"""
import os
import nltk
from nltk.corpus import stopwords
from xml.dom.minidom import parse
from pprint import pprint as pp

PATH = os.path.dirname(__file__)

def processInvertedIndexGenerator(vectorPath):
    invertedIndex = {}
    stop = stopwords.words('english')
    for line in vectorPath:
        if str(line[0]) == 'LEIA':
            dictionary=readXML(PATH+str(line[1]).strip())
            invertedIndex.update(invertedIndexGenerator(dictionary, stop))
        elif str(line[0]) == 'ESCREVA':
            writeInvertedIndex(PATH+str(line[1]).strip(), invertedIndex)
            
            
            
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
    
    
    
def writeInvertedIndex(filepath, invertedIndex):
    f = open(filepath, 'w+')
    
    for key in invertedIndex.keys():
        f.write(key.upper()+";%s\n" % invertedIndex[key])        
    f.close()
    
    
    
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