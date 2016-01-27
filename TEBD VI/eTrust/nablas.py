# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:24:38 2016

@author: ygorc
"""

def nabla_b(x, parameters):
	b = x
	p = parameters[1]
	q = parameters[2]
	w = parameters[3]
	eta = parameters[4]
	return 0.01*b
	
def nabla_p(x, parameters):
	b = parameters[0]
	p = x
	q = parameters[2]
	w = parameters[3]
	eta = parameters[4]
	return 0.01*p
	
def nabla_q(x, parameters):
	b = parameters[0]
	p = parameters[1]
	q = x
	w = parameters[3]
	eta = parameters[4]
	return 0.01*q
	
def nabla_w(x, parameters):
	b = parameters[0]
	p = parameters[1]
	q = parameters[2]
	w = x
	eta = parameters[4]
	return 0.01*w
	
def nabla_eta(x, parameters):
	b = parameters[0]
	p = parameters[1]
	q = parameters[2]
	w = parameters[3]
	eta = x
	return 0.01*eta
