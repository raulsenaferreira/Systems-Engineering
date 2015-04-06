# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:40:08 2015

@author: raul
"""
import os
from pprint import pprint as pp
from xml.dom.minidom import parse
#globals
PATH = os.path.dirname(__file__)

def executeQueryProcessor(pathVector):
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