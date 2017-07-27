import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import math
import pylab as plt
from matplotlib.collections import LineCollection


def alpha_shape(points, alpha, hull_simplices=[]):

    def add_edge(edges, edge_points, points, i, j):
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add( (i, j) )
        edge_points.append(points[ [i, j] ])
        
    tri = Delaunay(points)
    edges = set()
    edge_points = []    
    instancias = set()   
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        if ia not in hull_simplices and ib not in hull_simplices and ic not in hull_simplices:
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
            if (circum_r < 1.0/alpha):
                add_edge(edges, edge_points, points, ia, ib)
                add_edge(edges, edge_points, points, ib, ic)
                add_edge(edges, edge_points, points, ic, ia)
                instancias.add(ia)
                instancias.add(ib)
                instancias.add(ic)
    
    a=[]    
    for k in instancias:
        a.append(points[k])
    
    instancias = np.array(a)
    
    return (instancias, edge_points)


def alpha_compaction(points, alpha, threshold):
    iterations=0
    inst = np.copy(points)
    edge_points=[]
    inst, edge_points = alpha_shape(inst, alpha)
    while len(inst)>threshold:
        iterations=iterations+1
        inst, edge_points = alpha_shape(inst, alpha, ConvexHull(inst).simplices)
        
        #print('Iteration[{0}] Alpha={1} Instancias={2}'.format(iterations, str(alpha), str(len(inst))))
        #alpha=alpha+0.1
        
    return inst, edge_points