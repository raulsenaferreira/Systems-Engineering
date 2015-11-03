#%%
from __future__ import print_function
from graph_tool.all import *
import snap;
#%%
#Google graph
googleGraph = snap.LoadEdgeList(snap.PNGraph, "/home/raul/workspace/web-Google.txt", 0, 1)
#snap.PrintInfo(googleGraph, "Python type PUNGraph", "/home/filipebraida/workspace/info-pungraph.txt")

#num de componentes conexas
ComponentDist = snap.TIntPrV()
sz = snap.GetWccSzCnt(googleGraph, ComponentDist)
print (size(ComponentDist))
#snap.DrawGViz(googleGraph, snap.gvlDot, "/home/filipebraida/workspace/google.png", "Google graph")
#%%
#%%
#grau maximo e minimo e medio
print snap.GetMxDegNId(googleGraph)
print snap.CntDegNodes(googleGraph, 1)
#%%

#%%
#distancia media
NIdToDistH = snap.TIntH()
snap.GetShortPath(googleGraph, 1, NIdToDistH)
count = 0
for item in NIdToDistH:
    count += NIdToDistH[item]
print count/len(NIdToDistH)
#%%




#%%
#dolphin graph
ug = Graph(directed=False)
#ug = GraphView(ug, vfilt=lambda v: v.out_degree() > 4)
ug = load_graph("/home/filipebraida/workspace/dolphins/dolphins.gml")
#%%

#tree = min_spanning_tree(ug)
#u = GraphView(ug, efilt=tree)
graph_draw(ug, output="composed-filter.png")
#%%
#%%
ug.degree_property_map("out")
#%%
# Numero de vertices e arestas
#%%
'''
itrV = 0
itrA = 0
for v in ug.vertices():
    itrV +=1
for e in ug.edges():
    itrA +=1
'''
print ug
#%%

# Grau minimo e maximo
#%%
minimum = graph_tool.incident_edges_op(ug, "out", "min", ug.edge_index)
print minimum.a
#%%

#%%
#Grau  medio
res = graph_tool.stats.vertex_average(ug, "total")
print res
#%%

#Maior componente conexa
#%%
l = graph_tool.topology.label_largest_component(ug)
print l

u = graph_tool.topology.GraphView(ug, vfilt=l)
print u.num_vertices()
#%%

#Diametro
#%%
print graph_tool.topology.pseudo_diameter(ug)[0]
#%%

#Coeficiente de clusterizacao global
#%%
gC, std = graph_tool.clustering.global_clustering(ug)
print gC
print std
#%%

#Coeficiente de clusterizacao local
#%%
gt = graph_tool
lC = gt.clustering.local_clustering(ug)
print lC
print std
#%%

#distancia
#%%
dist = average(gt.topology.shortest_distance(ug)[1])
dist
#%%

#Page Rank
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
pr = gt.centrality.pagerank(ug)

a=[]
for p in range(0, 62):
    a.append(pr[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])

#grafo da metrica
graph_draw(ug, vertex_fill_color=pr, vorder=pr, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/dolphins_pagerank.png")
#%%

#Betweeness
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
vb, eb = gt.centrality.betweenness(ug)

a=[]
for p in range(0, 62):
    a.append(vb[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])

#metrica
graph_draw(ug, vertex_fill_color=vb, vorder=vb, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/dolphins_betweeness.png")
#%%

#Closeness
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
cl = gt.centrality.closeness(ug)

a=[]
for p in range(0, 62):
    a.append(cl[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])
#grafo da metrica
#graph_draw(ug, vertex_fill_color=cl, vorder=cl, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/dolphins_closeness.png")
#%%

#Katz
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
k = gt.centrality.closeness(ug)

a=[]
for p in range(0, 62):
    a.append(k[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])
#grafo da metrica
graph_draw(ug, vertex_fill_color=k, vorder=k, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/dolphins_katz.png")
#%%



# Politic books US
#%%
ug = Graph(directed=False)
#ug = GraphView(ug, vfilt=lambda v: v.out_degree() > 4)
ug = load_graph("/home/filipebraida/workspace/politics.gml")
#%%

#tree = min_spanning_tree(ug)
#u = GraphView(ug, efilt=tree)
graph_draw(ug, output="composed-filter.png")
#%%
#%%
ug.degree_property_map("out")
#%%
# Numero de vertices e arestas
#%%
'''
itrV = 0
itrA = 0
for v in ug.vertices():
    itrV +=1
for e in ug.edges():
    itrA +=1
'''
print ug
#%%

# Grau minimo e maximo
#%%
minimum = graph_tool.incident_edges_op(ug, "out", "max", ug.edge_index)
print minimum.a
#%%

#%%
#Grau  medio
res = graph_tool.stats.vertex_average(ug, "total")
print res
#%%

#Maior componente conexa
#%%
l = graph_tool.topology.label_largest_component(ug)

u = graph_tool.topology.GraphView(ug, vfilt=l)
print u.num_vertices()
#%%

#Diametro
#%%
print graph_tool.topology.pseudo_diameter(ug)[0]
#%%

#Coeficiente de clusterizacao global
#%%
gC, std = graph_tool.clustering.global_clustering(ug)
print gC
print std
#%%

#Coeficiente de clusterizacao local
#%%
gt = graph_tool
lC = gt.clustering.local_clustering(ug)
print lC
print std
#%%

#distancia
#%%
print average(gt.topology.shortest_distance(ug)[1])
#%%

#Page Rank
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
pr = gt.centrality.pagerank(ug)

a=[]
for p in range(0, 62):
    a.append(pr[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])

#grafo da metrica    
graph_draw(ug, vertex_fill_color=pr, vorder=pr, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/politics_pagerank.png")
#%%

#Betweeness
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
vb, eb = gt.centrality.betweenness(ug)

a=[]
for p in range(0, 62):
    a.append(vb[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])
    
#grafo da metrica
graph_draw(ug, vertex_fill_color=vb, vorder=vb, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/politics_betweeness.png")
#%%

#Closeness
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
cl = gt.centrality.closeness(ug)

a=[]
for p in range(0, 62):
    a.append(cl[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])
    
#grafo da metrica
graph_draw(ug, vertex_fill_color=cl, vorder=cl, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/politics_closeness.png")
#%%

#Katz
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
k = gt.centrality.closeness(ug)

a=[]
for p in range(0, 62):
    a.append(k[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])
    
#grafo da metrica
graph_draw(ug, vertex_fill_color=k, vorder=k, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/politics_katz.png")
#%%




# Political blogs US
#%%
ug = Graph(directed=True)
#ug = GraphView(ug, vfilt=lambda v: v.out_degree() > 4)
ug = load_graph("/home/filipebraida/workspace/politicalBlogs.gml")
#%%

#tree = min_spanning_tree(ug)
#u = GraphView(ug, efilt=tree)
graph_draw(ug, output="composed-filter.png")
#%%
#%%
ug.degree_property_map("out")
#%%
# Numero de vertices e arestas
#%%
'''
itrV = 0
itrA = 0
for v in ug.vertices():
    itrV +=1
for e in ug.edges():
    itrA +=1
'''
print ug
#%%

# Grau minimo e maximo
#%%
minimum = graph_tool.incident_edges_op(ug, "in", "max", ug.edge_index)
print minimum.a
#%%

#%%
#Grau  medio
res = graph_tool.stats.vertex_average(ug, "total")
print res
#%%

#Maior componente conexa
#%%
l = graph_tool.topology.label_largest_component(ug)

u = graph_tool.topology.GraphView(ug, vfilt=l)
print u.num_vertices()
#%%

#Diametro
#%%
print graph_tool.topology.pseudo_diameter(ug)[0]
#%%

#Coeficiente de clusterizacao global
#%%
gC, std = graph_tool.clustering.global_clustering(ug)
print gC
print std
#%%

#Coeficiente de clusterizacao local
#%%
gt = graph_tool
lC = gt.clustering.local_clustering(ug)
print lC
print std
#%%

#distancia
#%%
print average(gt.topology.shortest_distance(ug, directed=True)[0])
#%%

#Page Rank
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
pr = gt.centrality.pagerank(ug)

a=[]
for p in range(0, 62):
    a.append(pr[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])
    
#grafo da metrica
graph_draw(ug, vertex_fill_color=pr, vorder=pr, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/politicalBlogs_pagerank.png")
#%%

#Betweeness
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
vb, eb = gt.centrality.betweenness(ug)

a=[]
for p in range(0, 62):
    a.append(vb[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])
    
#grafo da metrica
graph_draw(ug, vertex_fill_color=vb, vorder=vb, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/politicalBlogs_betweeness.png")
#%%

#Closeness
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
cl = gt.centrality.closeness(ug)

a=[]
for p in range(0, 62):
    a.append(cl[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])
#grafo da metrica
#graph_draw(ug, vertex_fill_color=cl, vorder=cl, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/politicalBlogs_closeness.png")
#%%

#Katz
#%%
gt = graph_tool
ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
k = gt.centrality.closeness(ug)

a=[]
for p in range(0, 62):
    a.append(k[p])

#10 mais centrais
a = sorted(a, reverse=True)
for i in range(0, 10):
    print(a[i])
    
#10 menos centrais
a = sorted(a, reverse=False)
for i in range(0, 10):
    print(a[i])
#grafo da metrica
#graph_draw(ug, vertex_fill_color=k, vorder=k, vcmap=matplotlib.cm.gist_heat, output="/home/filipebraida/workspace/politicalBlogs_katz.png")
#%%