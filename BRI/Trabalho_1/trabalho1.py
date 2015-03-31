# -*- coding: utf-8 -*-
"""
@author: raul
"""

import nltk
import os
import logging

path = os.path.dirname(__file__)

def main():
    log(path+'/log.txt')
    readData(path)
    
class InvertedIndex:
    def __init__(self, docs):
        self.invIndex = {}
        self.root = docs
        for i,t in enumerate(docs):
            self.parse(i, t)
    
    def parse(self, idoc, doc):
        words = [w.lower() for w in doc.split(' ')]
        
        for w in words:
            if w in self.invIndex:
                self.invIndex[w].append(idoc)
            else:
                self.invIndex[w] = [idoc]
    
    def search(self, word):
        if word in self.invIndex:
            return self.invIndex[word][:]
        else:
            return []
            
            
            
def readData(filename):
    directory = open(path+filename, 'r')
    print(directory.readline())
    directory.close()
 

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