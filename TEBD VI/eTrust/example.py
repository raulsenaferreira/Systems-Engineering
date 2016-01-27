# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:26:25 2015

@author: ygorc
"""
from scipy import stats
from sklearn import preprocessing
import scipy.io as sio
import numpy as np


def common_rated_items(ratings, a, u):
	a_ratings = ratings[a] > 0
	u_ratings = ratings[u] > 0
	commom_rated_itens = np.bitwise_and(a_ratings, u_ratings)
	n_common_rated_items = np.sum(commom_rated_itens)
	index_common_rated_items = np.argsort(commom_rated_itens)[::-1][0:n_common_rated_items]
	return index_common_rated_items
	
def sim(ratings, a, u):
	a_mean_rating = np.sum(ratings[a])/np.count_nonzero(ratings[a])
	u_mean_rating = np.sum(ratings[u])/np.count_nonzero(ratings[u])
	
	cri = common_rated_items(ratings, a, u)
	dividendo = 0
	divisor_a = 0
	divisor_u = 0
	for i in cri:
		dividendo += (ratings[a,i] - a_mean_rating) * (ratings[u,i] - u_mean_rating)
		divisor_a += np.power(ratings[a,i] - a_mean_rating, 2) 
		divisor_u += np.power(ratings[u,i] - u_mean_rating, 2)
	
	divisor = np.sqrt(divisor_a) * np.sqrt(divisor_u)
	return dividendo / divisor
	
def calcular_pesos_beta(ratings, T):
	pesos = np.zeros(ratings.shape)
	
	for a in range(ratings.shape[0]):
		for u in range(ratings.shape[1]):
			s = stats.pearsonr(ratings[a], ratings[u])[0]
			
			if (s + T[a,u]) != 0 and (s * T[a,u]) != 0:
				pesos[a,u] = (2*s*T[a,u])/(s + T[a,u])
			elif s == 0 and T[a,u] != 0:
				pesos[a,u] = T[a,u]
			elif s != 0 and T[a,u] == 0:
				pesos[a,u] =  s
	
	return pesos
	
def calcular_pesos(ratings, T):
	pesos = np.zeros(ratings.shape)
	
	for a in range(ratings.shape[0]):
		for u in range(ratings.shape[1]):
			s = sim(ratings, a, u)
			
			if (s + T[a,u]) != 0 and (s * T[a,u]) != 0:
				pesos[a,u] = (2*s*T[a,u])/(s + T[a,u])
			elif s == 0 and T[a,u] != 0:
				pesos[a,u] = T[a,u]
			elif s != 0 and T[a,u] == 0:
				pesos[a,u] =  s
	
	return pesos
	
def k_vizinhos_mais_proximos(W, k):
	min_max_scaler = preprocessing.MinMaxScaler( (0,1) )
	distancias = 1 - min_max_scaler.fit_transform(W)
	vizinhos_mais_proximos = np.argsort( distancias )
	return vizinhos_mais_proximos[:,1:k+1]
	
def predict_initial_ratings(W, ratings, k):
	K = k_vizinhos_mais_proximos(W, k)
	P = np.zeros(W.shape)
	mean_ratings = np.sum(ratings, axis=1)/np.count_nonzero(ratings, axis=1)
	for a in range(ratings.shape[0]):
		for i in range(ratings.shape[1]):
			dividendo = 0
			divisor = 0
			for u in K[a]:
				if ratings[u,i] != 0:
					dividendo += W[a,u] * (ratings[u,i] - mean_ratings[u])
					divisor += W[a,u]
			if divisor != 0:
				P[a,i] = mean_ratings[a] + (dividendo/divisor)
			else:
				P[a,i] = mean_ratings[a]
	return P

def metrica_de_confiabilidade(P, W, k):
	K = k_vizinhos_mais_proximos(W, k)
	S = np.zeros(W.shape)
	V = np.zeros(W.shape)
	mean_ratings = np.sum(ratings, axis=1)/np.count_nonzero(ratings, axis=1)
	for a in range(ratings.shape[0]):
		for i in range(ratings.shape[1]):
			dividendo = 0
			S[a,i] = 0
			for u in K[a]:
				if ratings[u,i] != 0:
					dividendo += W[a,u] * np.power(ratings[u,i] - mean_ratings[u] - P[a,i] + mean_ratings[a], 2)
					S[a,i] += W[a,u]
			if S[a,i] != 0:
				V[a,i] = dividendo/S[a,i]
			else:
				V[a,i] = mean_ratings[a]

	max_r = np.max(ratings)
	min_r = np.min(ratings)
	mean_v = np.mean(V)
	
	gamma = np.log(0.5) / np.log( (max_r - min_r - mean_v) / (max_r - min_r) )
	Fv = np.power((max_r - min_r - V) / (max_r - min_r), gamma)
	
	mean_s = np.mean(S)
	Fs = 1 - mean_s/(mean_s + S)
	
	R = np.power(Fs * np.power(Fv, Fs), 1 / (1 + Fs))
	return R
	
def reconstruir_rede_de_confianca(W, P, ratings):
	V = np.zeros(ratings.shape)
	for u in range(ratings.shape[0]):
		for a in ([x for x in range(ratings.shape[0]) if x not in [u]]):
			V[u] = (W[a,u] * np.power(ratings[u] - np.mean(ratings[u]) - P[a] + np.mean(ratings[a]),2) ) / 4 * np.power(np.max(ratings) - np.min(ratings), 2)
	return V	
ratings = sio.loadmat("rating_with_timestamp.mat")["rating"]
d = sio.loadmat("epinion_trust_with_timestamp.mat")["trust"]


ratings = np.array([[0, 2, 0, 5, 0],
			   [0, 2, 3, 4, 0],
			   [5, 0, 0, 1, 3],
			   [4, 5, 2, 1, 0],
			   [0, 1, 5, 2, 4]])
						
d = np.array([[0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 0],
			 [0, 1, 0, 1, 1],
			 [1, 0, 1, 0, 0],
			 [1, 0, 1, 0, 1]])
				

log_tam_matrix = np.log(d.shape[0])
grau_medio = np.log(2*np.sum(d)/d.shape[0])
d_max = log_tam_matrix/grau_medio

T = (d_max - d + 1)/d_max
k = 2

W = calcular_pesos(ratings, T)d5
P = predict_initial_ratings(W, ratings, k)
R = metrica_de_confiabilidade(P, W, k)
V = reconstruir_rede_de_confianca(W, P, ratings)

theta = 0.7



