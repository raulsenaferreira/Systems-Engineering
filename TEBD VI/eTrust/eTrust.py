# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:59:31 2016

@author: ygorc
"""
from common import T, U, K, I, L, eta
from nablas import nabla_b, nabla_p, nabla_q, nabla_w, nabla_eta
from objective_functions import g_of_b, g_of_p, g_of_q, g_of_w, g_of_eta
from numpy import random, transpose, dot

n_iter = 10
epsilon = 0.01#random.uniform(0, 0.5)
rho = 1/10

w = random.random_sample(L)
b = random.random_sample(U.shape[0])
q = random.random_sample( (I.shape[0], K) )
p = []

for t in range(T.shape[0]):
	p.append(random.random_sample( (U.shape[0], K) ))

def zero_if_negative(x):
	return x - (x * (x < 0))

def back_to_the_bounded_feasible_region(x):
	return zero_if_negative(x)	
	
def search_gamma_with_goldstein_conditions(parameters, nablas, objective_functions):
	gammas = []
	for parameter, nabla_x, g_of_x in zip(parameters, nablas, objective_functions):
		print("Searching gamma for: %s" % str(nabla_x) )
		gamma = 1 - random.random() #the number always be greater than 0	
		x = parameter.copy()
		while True:
			old_x = x
			x = back_to_the_bounded_feasible_region(x - gamma * nabla_x(x, parameters))
			z = x - old_x
			gamma = rho * gamma
			lower_value = g_of_x(old_x, parameters) + (1 - epsilon)*gamma*dot(transpose(nabla_x(old_x, parameters)), z)
			upper_value = g_of_x(old_x, parameters) + epsilon*gamma*dot(transpose(nabla_x(old_x, parameters)), z)
			our_value = g_of_x(x, parameters)
			
			print ("\n\nlower:\t%f\nupper:\t%f\nour:\t%f" % (lower_value, upper_value, our_value))
			if lower_value <= our_value and our_value <= upper_value:
				gammas.append(gamma)
				break
	return gammas

def projected_gradient(b, p, q, w, eta):
	for i in range(n_iter):
		print("Projected gradient iteration: %d" % i)
		
		parameters = [b, p, q, w, eta]
		nablas = [nabla_b, nabla_p, nabla_q, nabla_w, nabla_eta]
		objective_functions = [g_of_b, g_of_p, g_of_q, g_of_w, g_of_eta]
		gammas = search_gamma_with_goldstein_conditions(parameters, nablas, objective_functions)
		gamma_b = gammas[0]
		gamma_p = gammas[1]
		gamma_q = gammas[2]
		gamma_w = gammas[3]
		gamma_eta = gammas[4]
		
		old_b = b.copy()
		old_p = p.copy()
		old_q = q.copy()
		old_w = w.copy()
		old_eta = eta.copy()
		
		parameters = [old_b,old_p,old_q,old_w,old_eta]
		w = old_w - gamma_w*nabla_w(old_w,parameters)
		b = old_b - gamma_b*nabla_b(old_b,parameters)
		p = old_p - gamma_p*nabla_p(old_p,parameters)
		q = old_q - gamma_q*nabla_q(old_q,parameters)
		eta = old_eta - gamma_eta*nabla_eta(old_eta,parameters)
	return b, p, q, w, eta

	
trained_parameters = projected_gradient(b, p, q, w, eta)
print(trained_parameters)