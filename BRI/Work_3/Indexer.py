# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:27:23 2015

@author: Raul Sena Ferreira
"""

import lucene
 
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
 
def indexer(docNumber, docText):
    lucene.initVM()
    indexDir = SimpleFSDirectory(File("index/"))
    writerConfig = IndexWriterConfig(Version.LUCENE_4_10_1, StandardAnalyzer())
    writer = IndexWriter(indexDir, writerConfig)
    doc = Document()
    doc.add(Field("docNumber", docNumber, Field.Store.YES, Field.Index.ANALYZED))
    doc.add(Field("docText", docText, Field.Store.YES, Field.Index.ANALYZED))
    writer.addDocument(doc)
    print "Closing index of %d docs..." % writer.numDocs()
    writer.close()
