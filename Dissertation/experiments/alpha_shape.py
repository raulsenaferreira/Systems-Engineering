# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:38:42 2016

@author: raul
"""
import numpy as np
from scipy.spatial import Delaunay
import math
import pylab as pl
from matplotlib.collections import LineCollection


def alpha_shape(points, alpha):

    def add_edge(edges, edge_points, points, i, j):
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add( (i, j) )
        edge_points.append(points[ [i, j] ])
        
    tri = Delaunay(points)
    edges = set()
    edge_points = []    
        
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
    
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
    
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
    
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
    
        # Here's the radius filter.
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, points, ia, ib)
            add_edge(edges, edge_points, points, ib, ic)
            add_edge(edges, edge_points, points, ic, ia)
            
    return edge_points



#generating data
mean = (1, 2)
cov = [[1, 0], [0, 1]]
points = np.random.multivariate_normal(mean, cov, 100)
   
#testing for different values of alpha
for i in range(5):
    alpha = (i+1)*.5
    edge_points = alpha_shape(points, alpha=alpha)
    lines = LineCollection(edge_points)
    pl.figure()
    pl.title('Alpha={0} Delaunay triangulation'.format(alpha))
    pl.gca().add_collection(lines)
    pl.plot(points[:,0], points[:,1], 'o', hold=1, color='#f16824')
            
