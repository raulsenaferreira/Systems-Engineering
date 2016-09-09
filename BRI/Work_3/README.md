### Doing the previous system using Lucene

The objective of this work is make the same exercises made previously but now with Apache Lucene / PyLucene.

This work also is processed in memory, and makes uses of Stemmer. Next, we compare new results with previous work results.

#### Dependences:
Install PyLucene http://lucene.apache.org/pylucene/install.html

Test the installation:

$ python2.7

`>>> import lucene `

`>>> lucene.initVM()`

If no exceptions are raised you're done!  You've now installed PyLucene 4 !

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

#### Troubleshooting
Some errors and solutions, copied from (http://bendemott.blogspot.com.br/2013/11/installing-pylucene-4-451.html)

Error:

`Traceback (most recent call last):  File "setup.py", line 418, in 
    main('--debug' in sys.argv)  File "setup.py", line 363, in main    raise type(e), "%s: %s" %(e, args)OSError: [Errno 2] No such file or directory: ['javac', '-d', 'jcc/classes', 'java/org/apache/jcc/PythonVM.java', 'java/org/apache/jcc/PythonException.java']`

Solution:

What this error is really telling us is it couldn't find "javac" ... This is a program provided by the debian/ubuntu package "openjdk-7-jdk" Install this package and try again.

Error:
 `jcc/sources/jcc.cpp:18:17: fatal error: jni.h: No such file or directory #include 
                 ^compilation terminated.`
  
Solution:

This error happens when the sources path can't be found for  your jdk (or JRE)... or you have the wrong jdk installed. If you get this error it means you are using a jdk that the installer isn't looking for.  Open pylucene-4.5.1-1/jcc/setup.py with a text editor.  Look for a line similar to this: 

    'linux2': '/usr/lib/jvm/java-7-openjdk-amd64',

Whatever the value of the JDK here is.. is the one you need installed.  In my case "openjdk-7-jdk"




Error:
*** ANT is not defined, please edit Makefile as required at top. Stop.

Solution:
This means you didn't edit a section of the file named Makefile to uncomment the variables needed to run the makefile.
