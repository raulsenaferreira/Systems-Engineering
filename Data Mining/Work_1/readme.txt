The objective is mining data in movielens database and thus, answer following questions:
1) Who watches movies X watches Y.
2) Who like movies X likes  Y.
3) Who dislikes movies X dislikes Y.
4) Who dislikes X but likes Y.(Try to find a interesting pattern)
P.S.: From the questions above, find 10 most frequent patterns.
This work uses SVD or PCA to reduce dimensionality data(about 5 or 10 collumns) and K-Means or DB Scan to clustering groups that watched the same movies(about 10 and 15 clusters).
For learning and personal challenge purposes this work is under development using julia language.
