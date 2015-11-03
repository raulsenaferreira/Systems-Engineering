#!/usr/bin/env python
"""
Random graph from given degree sequence.
Draw degree rank plot and graph with matplotlib.
"""
'''
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:05:14 2015

@author: raul
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib
except:
    raise

import networkx as nx

z=nx.utils.create_degree_sequence(100,nx.utils.powerlaw_sequence,exponent=2.1)
nx.is_valid_degree_sequence(z)

print ("Configuration model")
G=nx.configuration_model(z)  # configuration model

degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
#print "Degree sequence", degree_sequence
dmax=max(degree_sequence)

plt.loglog(degree_sequence,'b-',marker='o')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

# draw graph in inset
plt.axes([0.45,0.45,0.45,0.45])
Gcc=nx.connected_component_subgraphs(G)[0]
pos=nx.spring_layout(Gcc)
plt.axis('off')
nx.draw_networkx_nodes(Gcc,pos,node_size=20)
nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

plt.savefig("degree_histogram.png")
plt.show()
'''
'''
import networkx as nx
G=nx.path_graph(4)
H=nx.read_gml('web-Google.txt')
'''
import numpy as np
import matplotlib.pyplot as plt

# some fake data
data = np.random.randn(1000)
# evaluate the histogram
values, base = np.histogram(data, bins=40)
#evaluate the cumulative
cumulative = np.cumsum(values)
# plot the cumulative function
#plot the survival function
plt.plot(base[:-1], len(data)-cumulative, c='green')

plt.show()