# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:24:38 2016

@author: ygorc
"""
from numpy import zeros
from common import O_train, alpha, beta, f_of, diff_f_of_partial_w, diff_f_of_partial_bi, diff_f_of_partial_pikt, EPQijt, Hitk, Nit, Tit, R, K, U

def nabla_b(x, parameters):
	b = x
	p = parameters[1]
	q = parameters[2]
	w = parameters[3]
	eta = parameters[4]
	
	acc = 0
	for t in range(len(O_train)):
		for (i,j), rijt in O_train[t].items():
			
			Eijt, Pijt, Qijt = EPQijt(i, j, t, b, p, q, w, eta)
			
			accQ = 0
			accP = 0
			
			if (i,t) in Nit:
				for v in Nit[(i,t)]:
					if (v,j) in R:
						for k in range(K):					
							rvj = R[(v,j)][1]
						
							accP += diff_f_of_partial_bi(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k]
							accQ += diff_f_of_partial_bi(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k] * rvj
					if Qijt != 0:		
						acc += Eijt * (1/(Qijt)**2) * ((Qijt * accQ) - (Pijt * accP) )
	return 2*beta*b - 2*(1-alpha)*acc
	
def nabla_p(x, parameters):
	b = parameters[0]
	p = x
	q = parameters[2]
	w = parameters[3]
	eta = parameters[4]
	
	acc = [] 
	for t in range(len(O_train)):
		acc.append(zeros( (U.shape[0], K) ))
	
	for k in range(K):
		for t in range(len(O_train)):
						
			for (i,j), rijt in O_train[t].items():
				
				Eijt, Pijt, Qijt = EPQijt(i, j, t, b, p, q, w, eta)
				
				accQ = 0
				accP = 0
				accZ = 0
				
				if (i,t) in Nit:
					for v in Nit[(i,t)]:
						if (v,j) in R:
							
							rvj = R[(v,j)][1]

							accQ += diff_f_of_partial_pikt(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k] * rvj 
							accP += diff_f_of_partial_pikt(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k]
		
				for z in Tit[i]:
					if (z,j) in O_train[t]:
						Ezjt, Pzjt, Qzjt = EPQijt(z, j, t, b, p, q, w, eta)
						 
						Q_term = Qzjt * diff_f_of_partial_pikt(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k] * rijt
						P_term = Pzjt * diff_f_of_partial_pikt(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k]
					
						accZ +=  (1/Qzjt**2) * (Q_term - P_term)
						
				acc[t][i,k] += Eijt *( (alpha * q[j,k] + (1-alpha) * (1 / Qijt**2) * (Qijt*accQ - Pijt * accP) ) ) -2 * (1-alpha) * accZ + Hitk(i, t, k, p)
				acc[t][i,k] = 2 * beta * p[t][i,k] - 2 * acc[t][i,k]
	return acc
	
def nabla_q(x, parameters):
	b = parameters[0]
	p = parameters[1]
	q = x
	w = parameters[3]
	eta = parameters[4]
	
	acc = zeros(q.shape)
	for k in range(K):
		for t in range(len(O_train)):
			for (i,j), rijt in O_train[t].items():
				
				Eijt, Pijt, Qijt = EPQijt(i, j, t, b, p, q, w, eta)
				
				accQ = 0
				accP = 0
				
				if (i,t) in Nit:
					for v in Nit[(i,t)]:
						if (v,j) in R:
							
							rvj = R[(v,j)][1]
					
							accQ += f_of(w, p[t][i,k], p[t][v,k], b[i]) * rvj 
							accP += f_of(w, p[t][i,k], p[t][v,k], b[i])
		
								
						acc[i,k] += Eijt * (alpha * p[t][i,k] + (1-alpha) * (1 / Qijt**2) * (Qijt*accQ - Pijt * accP) )
	return 2*beta*eta + 2*(1-alpha)*acc
	
def nabla_w(x, parameters):
	b = parameters[0]
	p = parameters[1]
	q = parameters[2]
	w = x
	eta = parameters[4]
	
	acc = 0
	for t in range(len(O_train)):
		for (i,j), rijt in O_train[t].items():
			
			Eijt, Pijt, Qijt = EPQijt(i, j, t, b, p, q, w, eta)
			
			accQ = 0
			accP = 0
			
			if (i,t) in Nit:
				for v in Nit[(i,t)]:
					if (v,j) in R:
						for k in range(K):					
							rvj = R[(v,j)][1]
						
							accP += diff_f_of_partial_w(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k]
							accQ += diff_f_of_partial_w(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k] * rvj
							
					acc += Eijt * (1/(Qijt)**2) * ((Qijt * accQ) - (Pijt * accP) )
	return 2*beta*w - 2*(1-alpha)*acc
	
def nabla_eta(x, parameters):
	b = parameters[0]
	p = parameters[1]
	q = parameters[2]
	w = parameters[3]
	eta = x
	acc = 0
	
	for t in range(len(O_train)):
		for (i,j), rijt in O_train[t].items():
			
			Eijt, Pijt, Qijt = EPQijt(i, j, t, b, p, q, w, eta)
			
			acc2 = 0
			
			if (i,t) in Nit:
				for v in Nit[(i,t)]:
					if (v,j) in R:
						for k in range(K):
							tvj = R[(v,j)][0]
							rvj = R[(v,j)][1]
					
							acc2 += f_of(w, p[t][i,k], p[t][v,k], b[i]) * q[j,k] * rvj * (t - tvj)
	
							
					acc += (Eijt/Qijt) * acc2
	return 2*beta*eta + 2*(1-alpha)*acc
