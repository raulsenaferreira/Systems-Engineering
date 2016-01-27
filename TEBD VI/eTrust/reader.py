# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:26:25 2015

@author: ygorc
Dataset available on http://www.public.asu.edu/~jtang20/datasetcode/truststudy.htm
==============================================================================================
rating_with_timestamp.mat 

rating.mat includes the rating information with time points when ratings are created.
The time points are splited into 11 timestamps.
There are six columns and they are:
  * userid
  * productid
  * categoryid
  * rating
  * helpfulness
  * timestamp
===============================================================================================
epinion_trust_with_timestamp.mat

epinion_trust_with_timestamp.mat includes the trust relations 
between users with the time stamps when the relations were established.
There are three columns:
  * truster (userid)
  * trustee (userid)
  * timestamp
===============================================================================================
"""

import scipy.io as sio
import numpy as np
from sklearn.preprocessing import LabelEncoder



#ratings = ratings.astype([('userId', '<i8'), ('productId', '<i8'), ('categoryId', '<i8'), ('rating', '<i8'), ('helpsfullness', '<i8'), ('timepoint', '<i8')])
#trust = trust.astype([('truster', '<i8'), ('trustee', '<i8'), ('timepoint', '<i8')])

#categories = np.histogram(ratings[:, 2], bins=np.max(ratings[:, 2])-1)
#times = np.histogram(ratings[:, 5], bins=np.max(ratings[:, 5])-1)


#ratings = ratings_data[:,3]

def load_ratings_data():
	
	ratings_data = sio.loadmat("rating_with_timestamp.mat")["rating"]
	
	#users = ratings_data[:,0]
	#items = ratings_data[:,1]
	#times = ratings_data[:,5]
	
	le = LabelEncoder()
	ratings_data[:,0] = le.fit_transform(ratings_data[:,0])
	
	le = LabelEncoder()
	ratings_data[:,1] = le.fit_transform(ratings_data[:,1])
	
	le = LabelEncoder()
	ratings_data[:,5] = le.fit_transform(ratings_data[:,5])
	
	U = np.sort(np.unique(ratings_data[:,0]))
	I = np.sort(np.unique(ratings_data[:,1]))
	T = np.sort(np.unique(ratings_data[:,5])) 
	
	tui = {}
		
	O = []
	R = {}
	for t in range(T.shape[0]):
		O.append({})
		
	for line in ratings_data:
		i = line[0]
		j = line[1]
		t = line[5]
		rijt = line[3]
		
		if i in tui:
			if rijt < tui[i]:
				tui[i] = rijt
		else:
			tui[i] = rijt
		
		O[t][(i, j)] = rijt
		R[(i, j)] = (t, rijt)
	
	O_train = O[:-1]
	O_test = [O[-1]]
	return O_train, O_test, R, U, I, T, tui

def load_trust_data():
	
	trust_data = sio.loadmat("epinion_trust_with_timestamp.mat")["trust"]

	#n_users = np.unique(users).shape[0]
	#n_items = np.unique(items).shape[0]
	#n_times = np.unique(times).shape[0]
	Trust = {}
	N = {}
	for line in trust_data:
		truster = line[0]
		trustee = line[1]
		timestamp = line[2]
		
		Trust[(truster, trustee, timestamp)] = 1
		if (truster, timestamp) not in N:  		
			N[(truster, timestamp)] = []
		N[(truster, timestamp)].append(trustee)
		
		
	return Trust, N