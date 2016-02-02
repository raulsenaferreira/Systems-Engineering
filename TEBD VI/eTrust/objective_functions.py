# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:25:23 2016

@author: ygorc
"""
from common import O_train, Nit, R, K, U, f_of, c_of, alpha, beta, lmbda, rating_decay, euclidean_norm, tui
from numpy import dot, transpose

def g_sum(b, p, q, w, eta):
	acc = 0
	for t in range(len(O_train)):
		for (i,j), rijt in O_train[t].items():
			pq = dot(transpose(q[j]), p[t][i])

			dividend = 0
			divisor = 0
			if (i,t) in Nit:
				for v in Nit[(i,t)]:
					if (v,j) in R:
						for k in range(K):
							tvj = R[(v,j)][0]
							rvj = R[(v,j)][1]
							divisor += f_of(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k]
							dividend += f_of(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k] * rating_decay(t, tvj, eta[i], rvj)
				if divisor != 0:
					rating_prediction_by_trust = dividend/divisor
				else:
					rating_prediction_by_trust = 0
					
				acc += (rijt - (alpha*pq) - ((1-alpha) * rating_prediction_by_trust))**2
			else:
				acc += rijt - (alpha*pq)
	return acc

def g_of_b(x, parameters):
	b = x
	p = parameters[1]
	q = parameters[2]
	w = parameters[3]
	eta = parameters[4]
	acc = g_sum(b, p, q, w, eta)
	
	return acc + beta * euclidean_norm(b)**2

def g_of_p(x, parameters):
	b = parameters[0]
	p = x
	q = parameters[2]
	w = parameters[3]
	eta = parameters[4]
	acc = g_sum(b, p, q, w, eta)
	acc2 = 0
	for t in range(len(p)):
		for i in range(p[t].shape[0]):
			acc2 += euclidean_norm(p[t][i])**2
	acc3 = 0
	for i in U: 
		for t in range(tui[i] + 1, len(O_train)):
			for k in range(K):
				acc3 += c_of(p[t][i,k],p[t-1][i,k])
	
	return acc + beta * acc2 + lmbda * acc3
	
def g_of_q(x, parameters):
	b = parameters[0]
	p = parameters[1]
	q = x
	w = parameters[3]
	eta = parameters[4]
	acc = g_sum(b, p, q, w, eta)
	acc2 = 0
	for i in range(q.shape[0]):
		acc2 += euclidean_norm(q[i])**2
			
	return acc + beta * acc2
	
def g_of_w(x, parameters):
	b = parameters[0]
	p = parameters[1]
	q = parameters[2]
	w = x
	eta = parameters[4]
	acc = g_sum(b, p, q, w, eta)
	return acc + beta * euclidean_norm(w)**2
	
def g_of_eta(x, parameters):
	b = parameters[0]
	p = parameters[1]
	q = parameters[2]
	w = parameters[3]
	eta = x
	acc = g_sum(b, p, q, w, eta)
	acc2 = 0
	for i in range(eta.shape[0]):
		acc2 += euclidean_norm(eta[i])**2
	return acc + beta * acc2