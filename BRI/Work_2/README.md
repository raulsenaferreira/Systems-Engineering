### Evaluating an information retrieval model

This second work is a sequence of the implementation of the Work 1, thus, in this directory has only the additional files and directories to complete the work 2.

The instructions are contained inside the pdf file in this folder.

The metric used to classify a expected result as relevant or non relevant was the sum of votes in score attribute. Values greather than 3 is considered relevant, the others is non relevant. Ex:
doc 530 1010 --> 1+0+1+0=2 --> Non relevant
doc 490 1110 --> 1+1+1+0=3 --> Non relevant
doc 555 1111 --> 1+1+1+1=4 --> Relevant
doc 490 2200 --> 2+2+0+0=4 --> Relevant
