### Doing the previous system using Lucene

The objective of this work is make the same exercises made previously but now with Apache Lucene / PyLucene.

This work also is processed in memory, and makes uses of Stemmer. Next, we compare new results with previous work results.

#### Dependences:
Install PyLucene http://lucene.apache.org/pylucene/install.html

####Instructions:
Delete all files in index directory and execute main.py file

####Folders and files:

Indexer.py -> Indexing implementation with PyLucene

Retriever.py-> Retrieving implementation with PyLucene

Metrics.py-> Metrics used to measure PyLucene results

data-> Fibrose cystic xml files

index-> Lucene indexed Files

ResultsLucene-> Searcher results

MetricResults-> Individual Results of all Metrics

REPORT.txt -> All Results condensed in one text file

####References:

http://svn.apache.org/viewvc/lucene/pylucene/trunk/samples/

http://www.lucenetutorial.com/lucene-in-5-minutes.html

http://lucene.apache.org/pylucene/

http://graus.nu/blog/pylucene-4-0-in-60-seconds-tutorial/
