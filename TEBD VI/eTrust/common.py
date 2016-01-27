# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:51:38 2016

@author: ygorc
"""
import numpy as np
from numpy import random
import reader
import math
from numpy.linalg import norm
from numpy import dot, transpose

def euclidean_norm(v):
	return norm(v, 2)

L = 9
K = 10

alpha = 0.1
beta = 0.05
lmbda = 0.2

print("Ready for load ratings")
O_train, O_test, R, U, I, T, tui = reader.load_ratings_data()
print("Ratings loaded")
print("Ready for load trust data")
Trust, N = reader.load_trust_data()
print("Trust loaded")

eta = random.random_sample(U.shape[0])

def sivkt(pikt, pvkt):
	s = np.zeros(L)
	s[0] = pikt
	s[1] = pvkt
	s[2] = pikt + pvkt
	s[3] = pikt - pvkt
	s[4] = pvkt - pikt
	s[5] = (pikt - pvkt)**2
	s[6] = pikt*pvkt
	s[7] = pikt/pvkt
	s[8] = pvkt/pikt
	return s
	
def diff_sivkt_partial_pikt(pikt, pvkt):
	s = np.zeros(L)
	s[0] = 1
	s[1] = 0
	s[2] = 1
	s[3] = 1
	s[4] = -1
	s[5] = -2
	s[6] = pvkt
	s[7] = 1/pvkt
	s[8] = -pvkt/(pikt**2)
	return s

def c_of(pitk, pit_minus_1k):
	return (pitk - pit_minus_1k)**2

def diff_c_of_partial_pit_minus_1k(pitk, pit_minus_1k):
	return -2 * (pitk - pit_minus_1k)
	
def diff_c_of_partial_pitk(pitk, pit_minus_1k):
	return 2 * (pitk - pit_minus_1k)
	
def f_of(w, pikt, pvkt, bi):
	wTsivkt = dot(transpose(w), sivkt(pikt, pvkt))
	return 1/(1 + np.exp(-wTsivkt + bi))

def diff_f_of_partial_w(w, pikt, pvkt, bi):
	f = f_of(w, pikt, pvkt, bi)
	return f * (1-f) * sivkt(pikt, pvkt)
	
def diff_f_of_partial_bi(w, pikt, pvkt, bi):
	f = f_of(w, pikt, pvkt, bi)
	return f * (1-f)

def diff_f_of_partial_pikt(w, pikt, pvkt, bi):
	f = f_of(w, pikt, pvkt, bi)
	diff_sivkt = diff_sivkt_partial_pikt(pikt, pvkt)
	return f * (1-f) * dot(transpose(w), diff_sivkt)
	
def rating_decay(t, tvj, eta, r):
	return math.exp( -eta*(t-tvj)) * r
	
	
def EPQijt(i, j, t, b, p, q, w, eta):
	Qijt = 0
	Pijt = 0
	rijt = O_train[t][(i, j)]
	
	pq = dot(transpose(q[j]), p[t][i])
	
	for v in N[(i,t)]:
		if (v,j) in R:
			for k in range(K):
				tvj = R[(v,j)][0]
				rvj = R[(v,j)][1]
				Qijt += f_of(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k]
				Pijt += f_of(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k] * rating_decay(t, tvj, eta[i], rvj)
	if Qijt != 0:
		rating_prediction_by_trust = Pijt/Qijt
	else:
		rating_prediction_by_trust = 0
		
	Eijt = (rijt - (alpha*pq) - ((1-alpha) * rating_prediction_by_trust))
	
	return Eijt, Pijt, Qijt