# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:40:08 2015

@author: Raul Sena Ferreira
"""
import time
import operator
from xml.dom.minidom import parse
import logging

#global
queryProcess = ''

def executeQueryProcessor(path, pathVector):
    begin = time.time()
    global queryProcess
    #log
    logPath = path+'/QueryProcessor/queryProcessor.log'
    log('queryProcessor', logPath)
    queryProcess = logging.getLogger('queryProcessor')
    queryProcess.info('Processing Query Processor Module...')
    ini = time.time()
    
    queries = readQueriesXML(path+pathVector[0][1])
    
    timeElapsed = time.time()-ini
    queryProcess.info('Read queries operation finished with %s' % str(timeElapsed))
    
    queryProcess.info('Writing Query Processor and Expected results on their respective files...')
    ini = time.time()
    
    writeQueryProcessorData(path, pathVector, queries)
    
    timeElapsed = time.time()-ini
    queryProcess.info('Write operation finished with %s' % str(timeElapsed))
    
    end = time.time() - begin
    queryProcess.info('End of Query Processor Module. Total of %s elapsed.' % str(end))
    
    
    
def writeQueryProcessorData(path, pathVector, queries):
    
    for line in pathVector:
        if str(line[0]) == 'CONSULTAS':
            f = open(path+str(line[1]), 'w+')
            for key in queries.keys():
                f.write(key+';'+queries[key][0]+'\n')
            f.close()
        elif str(line[0]) == 'ESPERADOS':
            f = open(path+str(line[1]), 'w+')
            for key in queries.keys():
                f.write(key+';%s\n' % sorted(queries[key][1].items(), key=operator.itemgetter(1), reverse=True))
            f.close()
            
            

def scoreCounter(scoreList):
    score = 0  
    s = list(scoreList)
    for i in s:
        score += int(i)
    return score


            
def readQueriesXML(filename):
    queryProcess.info('Reading '+filename+'...')
    
    dictionary = {}
    
    DOMTree = parse(filename)
    collection = DOMTree.documentElement
    
    queries = collection.getElementsByTagName("QUERY")
    
    queryProcess.info('Processing score list of Expected results...')
    
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
                        expected.update({key: scoreCounter(score)})
                dictionary[queryNumber] = [queryText, expected]
            except IndexError:
                queryProcess.warning('Document["+queryNumber+"] doesn\'t have records!')
        except IndexError:
            queryProcess.warning("Document["+queryNumber+"] doesn't have query text!")
        
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