# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:29:07 2015

@author: Raul Sena Ferreira
"""

import lucene
 
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
 
def retriever(queryNumber, queryText):
    lucene.initVM()
    analyzer = StandardAnalyzer(Version.LUCENE_4_10_1)
    reader = IndexReader.open(SimpleFSDirectory(File("index/")))
    searcher = IndexSearcher(reader)
 
    query = QueryParser(Version.LUCENE_4_10_1, "docText", analyzer).parse(QueryParser.escape(queryText))
    MAX = 1000
    hits = searcher.search(query, MAX)
 
    print "Found %d document(s) that matched query '%s':" % (hits.totalHits, query)
    for hit in hits.scoreDocs:
        print hit.score, hit.doc, hit.toString()
        doc = searcher.doc(hit.doc)
        print doc.get("docNumber").encode("utf-8")
