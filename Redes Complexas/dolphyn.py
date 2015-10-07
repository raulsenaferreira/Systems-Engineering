from graph_tool.all import *


#%%
ug = Graph(directed=False)

ug = GraphView(ug, vfilt=lambda v: v.out_degree() > 4)
ug = load_graph("dolphins/dolphins.gml")
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
minimum = graph_tool.ver(ug, "out", "min", ug.edge_index)
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
