# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:44:07 2015

@author: Raul Sena Ferreira
"""
import ast
import nltk
import operator
import math
import logging
import time
#from pprint import pprint as pp
from nltk.stem.porter import *

#globals
searcher = ''

def processSearcher(path, indexes, queries, pathVector, stop, use_mode):
    begin = time.time()
    global searcher

    #log
    logPath = path+'/Searcher/searcher.log'
    log('Searcher', logPath)
    searcher = logging.getLogger('Searcher')
    searcher.info('Processing Searcher Module...')

    indexes = strToDict(indexes)
    queries = strToDict(queries, False)

    searcher.info('Making search and calculating rankings...')
    ini = time.time()

    rankings=''
    if use_mode == 'STEMMER':
        stemmer = PorterStemmer()
        rankings = makeSearch(indexes, queries, stemmer, stop)
    elif use_mode == 'NOSTEMMER':
        rankings = makeSearch(indexes, queries, None, stop)
    else: print("Use mode undefined")

    timeElapsed = time.time()-ini
    searcher.info('Searching and ranking calculus operation finished with %s' % str(timeElapsed))

    pathToSaveFile  = ''
    if use_mode == 'NOSTEMMER':
        pathToSaveFile = pathVector[2][1]
    elif use_mode == 'STEMMER':
        pathToSaveFile = pathVector[3][1]
    else:
        print("Use mode undefined")

    writeResults(path, rankings, pathToSaveFile)

    end = time.time() - begin
    searcher.info('End of Searcher Module processing. Total of %s elapsed.' % str(end))
    #v1,v2 = [3, 45, 7, 2], [2, 54, 13, 15]
    #pp(cosine_similarity(v1,v2))



def cosine_similarity(v1,v2):
    searcher.info('Calculating Cosine Similarity of array...')
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)



def strToDict(listString, hasListInside=True):
    searcher.info('Transforming String to Dictionary...')

    dictionary = {}
    if hasListInside is True:
        for s in listString:
            dictionary.update({s[0]: ast.literal_eval(s[1])})
    else:
        for s in listString:
            dictionary.update({s[0]: s[1]})
    return dictionary



def makeSearch(indexes, queries, stemmer, stop=[]):

    rankings = {}
    for queryNumber in queries.keys():
        #N = len(queries)
        tokens = nltk.word_tokenize(queries[queryNumber])
        #freq_iq = nltk.FreqDist(tokens)
        #using only valid tokens and cleaning invalid ones, e.g. (?, brackets and etc)
        ranking = {}
        for token in tokens:
            if token in stop or len(token) < 2:
                tokens.remove(token)
        for token in tokens:
            if stemmer is not None:
                #pp("Using stemmer...")
                token=stemmer.stem(token)
            #updating weights based on terms frequence
            try:
                for k in indexes[token].keys():
                    #n_q = len(indexes[token])
                    #maxFreq_iq = freq_iq.most_common(1)
                    w_iq = 1
                    #w_iq=(freq_iq[token]/maxFreq_iq[0][1])*math.log(N/n_q)
                    tf_idf = indexes[token][k]*w_iq
                    #cosine = math.sqrt(math.pow(w_iq, 2))*math.sqrt(math.pow(indexes[token][k], 2))
                    #sim=tf_idf/cosine
                    try:
                        weight = ranking[k] + tf_idf
                        ranking.update({k: 1/weight})
                    except KeyError:
                        if tf_idf > 0:
                            ranking.update({k: 1/tf_idf})
            except KeyError:
                searcher.warning("word '"+token+"' not found!")
        rankings.update({queryNumber: sorted(ranking.items(), key=operator.itemgetter(1), reverse=False)})
    return rankings



def writeResults(rootPath, rankings, path):
    searcher.info('Writing search results on file...')

    f = open(rootPath+path, 'w+')
    for ident in rankings.keys():
        line = ''
        line+= ident+';'
        i=0
        for docNum in rankings[ident]:
            #don't print documents umbers with 0 weight (e.g. no occurrence)
            if docNum[1] > 0.0:
                i+=1
                line+='['
                line+= str(i)+', '
                line+= str(docNum[0])+', '
                line+= str('%.2f' % round(docNum[1], 2))#distance
                line+='],'
        line=line.rstrip(',')+'\n'
        f.write(line)
    f.close()



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
