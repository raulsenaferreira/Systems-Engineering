# -*- coding: utf-8 -*-
"""
@author: raul
"""


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