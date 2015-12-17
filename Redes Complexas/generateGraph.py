from __future__ import division, absolute_import, print_function
from pylab import *  # for plotting
from numpy.random import *  # for random sampling
from graph_tool.all import *
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
import snap
from pprint import pprint as pp
import re
import psycopg2
import sys
import re 

def calculateDistributions(ug):
    ccdf_patch = mpatches.Patch(color='green', label='CCDF')
    d = ug.degree_property_map("out")
    in_hist = vertex_hist(ug, d)

    y = in_hist[0]
    err = sqrt(in_hist[0])
    err[err >= y] = y[err >= y] - 1e-2

    a = sorted(in_hist[0][1:], reverse=True)
    b = sorted(in_hist[1][1:-1], reverse=False)

    figure(figsize=(6,4))
    errorbar(b, a, fmt="o")
    gca().set_yscale("log")
    gca().set_xscale("log")
    gca().set_ylim(0, 1e4)
    gca().set_xlim(0, 1e4)
    subplots_adjust(left=0.2, bottom=0.2)
    xlabel("$k_{out}$")
    ylabel("$NP(k_{out})$")
    tight_layout()
    savefig("graph_degree_distribution.png")
    values, base = np.histogram(a, bins=max(b))
    cumulative = np.cumsum(values)

    plt.legend(handles=[ccdf_patch])
    plt.plot(base[:-1], len(a)-cumulative, c='green')
    #plt.show()
    savefig("graph_degree_distribution_and_ccdf.png")

def calculateMetrics(ug):
    print('Iniciando calculo das metricas...')
    gt = graph_tool
    lim = gt.topology.GraphView(ug).num_vertices()
    ug.degree_property_map("out")

    # Grau minimo e maximo
    minimum = graph_tool.incident_edges_op(ug, "out", "min", ug.edge_index)
    maximum = graph_tool.incident_edges_op(ug, "out", "min", ug.edge_index)
    print('Grau minino: {0}   Grau maximo: {1}'.format(min(minimum.a), max(maximum.a)))
    
    #Grau  medio
    res = graph_tool.stats.vertex_average(ug, "total")
    print ('Grau medio: {0}'.format(res))

    #tamanho da maior componente conexa
    l = gt.topology.label_largest_component(ug)
    u = gt.topology.GraphView(ug, vfilt=l)
    print ('Tamanho da maior CC: {0}'.format(u.num_vertices()))

    #diametro
    print ('Diametro: {0}'.format(gt.topology.pseudo_diameter(ug)[0]))

    #Coeficiente de clusterizacao global
    gC, std = gt.clustering.global_clustering(ug)
    print ('Coef. Clusterizacao Global: {0}   Desvio padrao: {1}'.format(gC, std))

    #Coeficiente de clusterizacao local
    lC = gt.clustering.local_clustering(ug)
    print ('Coef. Clusterizacao local: {0}'.format(lC))

    #distancia media
    dist = average(gt.topology.shortest_distance(ug)[1])
    print('Distancia media: {0}'.format(dist))
    
    #Page Rank
    print('Iniciando Page Rank...')
    ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
    pr = gt.centrality.pagerank(ug)
    a=[]

    for p in range(0, lim):
        a.append(pr[p])

    #10 mais centrais
    print('Os dez vertices mais centrais: ')
    a = sorted(a, reverse=True)
    for i in range(0, 10):
        print(a[i])
        
    #10 menos centrais
    print('Os dez vertices menos centrais: ')
    a = sorted(a, reverse=False)
    for i in range(0, 10):
        print(a[i])

    print('Fim Page Rank, salvando figura do grafo...')
    #grafo da metrica
    graph_draw(ug, vertex_fill_color=pr, vorder=pr, vcmap=matplotlib.cm.gist_heat, output="pagerank.png")
    

    #Betweeness
    print('Iniciando Betweeness...')
    ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
    vb, eb = gt.centrality.betweenness(ug)

    a=[]
    for p in range(0, lim):
        a.append(vb[p])

    #10 mais centrais
    print('Os dez mais centrais: ')
    a = sorted(a, reverse=True)
    for i in range(0, 10):
        print(a[i])
        
    #10 menos centrais
    print('Os dez menos centrais: ')
    a = sorted(a, reverse=False)
    for i in range(0, 10):
        print(a[i])

    print('Fim Betweeness, salvando figura do grafo...')
    #grafo da metrica
    graph_draw(ug, vertex_fill_color=vb, vorder=vb, vcmap=matplotlib.cm.gist_heat, output="betweeness.png")
    

    #Closeness
    print('Iniciando Closeness...')
    ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
    cl = gt.centrality.closeness(ug)

    a=[]
    for p in range(0, lim):
        a.append(cl[p])

    #10 mais centrais
    print('Os dez mais centrais: ')
    a = sorted(a, reverse=True)
    for i in range(0, 10):
        print(a[i])
        
    #10 menos centrais
    print('Os dez menos centrais: ')
    a = sorted(a, reverse=False)
    for i in range(0, 10):
        print(a[i])
    
    print('Fim Closeness, salvando figura do grafo...')
    #grafo da metrica
    graph_draw(ug, vertex_fill_color=cl, vorder=cl, vcmap=matplotlib.cm.gist_heat, output="closeness.png")
    

    #Katz
    print('Iniciando Katz...')
    ug = gt.GraphView(ug, vfilt=gt.topology.label_largest_component(ug))
    k = gt.centrality.closeness(ug)

    a=[]
    for p in range(0, lim):
        a.append(k[p])

    #10 mais centrais
    print('Os dez mais centrais: ')
    a = sorted(a, reverse=True)
    for i in range(0, 10):
        print(a[i])
        
    #10 menos centrais
    print('Os dez menos centrais: ')
    a = sorted(a, reverse=False)
    for i in range(0, 10):
        print(a[i])

    print('Fim Katz, salvando figura do grafo...')
    #grafo da metrica
    graph_draw(ug, vertex_fill_color=k, vorder=k, vcmap=matplotlib.cm.gist_heat, output="katz.png")
    
    print('Fim do calculo das metricas!')


# Campos do banco
# ID, nome, imagem, sexo, olhos, cor_da_pele, cabelo, peso_aproximado, altura_aproximada, tipo_fisico, transtorno_mental, idade, data_nascimento, dias_desaparecido 
# , data_desaparecimento, bairro_desaparecimento, cidade_desaparecimento, uf_desaparecimento, marca_caracteristica, status, informacoes, boletim_ocorrencia, fonte

#grafo modelado pela similaridade dos vertices para os atributos: faixa etaria, sexo, cor da pele. 
#Neste grafo, para satisfazer a total similaridade entre dois vertices, os atributo desses vertices devem ter os seguintes valores:
#faixa etaria < 29 anos, sexo masculino e cor de pele parda ou negra
#perfis de pessoas assassinadas pelo trafico ou por policiais
def genGraph_Murderer():
    ug = Graph()
    ug.set_directed(False)

    #edge colors
    alpha=0.10
    edge_color = ug.new_edge_property('vector<double>')
    ug.edge_properties['edge_color']=edge_color
    #vertex colors
    vertex_color = ug.new_vertex_property('vector<double>')
    ug.vertex_properties['vertex_color'] = vertex_color
    #ug.vertex_properties['uf'] = plot_uf
    conn_string = "host='jabot.cos.ufrj.br' dbname='epinions' user='jabot' password='jabot'"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    
    query = "SELECT ID, idade, sexo, uf_desaparecimento, cor_da_pele FROM registro where idade not like '';"
    cursor.execute(query)
    rows = cursor.fetchall()
    
    ug.add_vertex(len(rows))
    
    n = re.compile('n', re.IGNORECASE)
    m = re.compile('m', re.IGNORECASE)
    p = re.compile('p', re.IGNORECASE)
    b = re.compile('b', re.IGNORECASE)
    f = re.compile('f', re.IGNORECASE)

    for i in range(0, len(rows)):
        u = rows[i]
        for j in range(i+1, len(rows)):
            v = rows[j]
            if int(u[1]) < 29 and int(v[1]) < 29:
                if bool(m.match(u[2].lower())) and bool(m.match(v[2].lower())): #masculino
                    #if (u[3] == 'RJ' or u[3] == 'SP') and (v[3] == 'RJ' or v[3] == 'SP'):   
                    if (bool(n.match(u[4].lower())) or bool(m.match(u[4].lower())) or bool(p.match(u[4].lower())) ) and (bool(n.match(v[4].lower())) or bool(m.match(v[4].lower())) or bool(p.match(v[4].lower())) ): #preto, pardo, moreno e mulato
                        ug.add_edge(u[0], v[0])
                        vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                        vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                        edge_color[ug.edge(u[0], v[0])] = (1.0/255.0, 1.0/255.0, 1.0/255.0, 100) #preto
                        #print("OK!!!")
                    else:
                        ug.add_edge(u[0], v[0])
                        vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                        vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                        edge_color[ug.edge(u[0], v[0])] = (224.0/255.0, 83.0/255.0, 127.0/255.0, alpha) #laranja
                    '''
                    else:
                        ug.add_edge(u[0], v[0])
                        vertex_color[ug.vertex(u[0])] = (171.0/255.0, 171.0/255.0, 171.0/255.0, 100)#cinza
                        vertex_color[ug.vertex(v[0])] = (171.0/255.0, 171.0/255.0, 171.0/255.0, 100)#cinza
                        edge_color[ug.edge(u[0], v[0])] = (224.0/255.0, 83.0/255.0, 127.0/255.0, alpha) #laranja
                    '''
                else:
                    ug.add_edge(u[0], v[0])
                    vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                    vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                    edge_color[ug.edge(u[0], v[0])] = (224.0/255.0, 83.0/255.0, 127.0/255.0, alpha) #laranja
            elif int(u[1]) >= 29 and int(v[1]) >= 29:
                ug.add_edge(u[0], v[0])
                vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                edge_color[ug.edge(u[0], v[0])] = (224.0/255.0, 83.0/255.0, 127.0/255.0, alpha) #laranja

            else:
                if bool(m.match(u[2].lower())) and bool(m.match(v[2].lower())): #masculino
                    ug.add_edge(u[0], v[0])
                    vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                    vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                    edge_color[ug.edge(u[0], v[0])] = (224.0/255.0, 83.0/255.0, 127.0/255.0, alpha) #laranja
                else:
                    if ( bool(n.match(u[4].lower())) or bool(m.match(u[4].lower())) or bool(p.match(u[4].lower())) ) and (bool(n.match(v[4].lower())) or bool(m.match(v[4].lower())) or bool(p.match(v[4].lower())) ): #preto, pardo, moreno e mulato
                        ug.add_edge(u[0], v[0])
                        vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                        vertex_color[ug.vertex(v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, 100)#lilas
                        edge_color[ug.edge(u[0], v[0])] = (224.0/255.0, 83.0/255.0, 127.0/255.0, alpha) #laranja

    ug = graph_tool.GraphView(ug,vfilt=lambda v: (v.out_degree() > 0) )
    ug.purge_vertices()

    #calculateDistributions(ug)
    #calculateMetrics(ug)
    #graph_draw(ug, edge_color = ug.edge_properties['edge_color'], vertex_fill_color=ug.vertex_properties['vertex_color'], vertex_color=ug.vertex_properties['vertex_color'])#,vertex_text=ug.vertex_index, vertex_font_size=10, output="UF_corDaPele.png"
    #detectando B blocos (comunidades) no grafo
    #ug = graph_tool.GraphView(ug, vfilt=graph_tool.topology.label_largest_component(graph_tool.GraphView(ug, directed=False)))
    #state = graph_tool.community.BlockState(ug, B=ug.num_vertices(), deg_corr=True)
    #state = graph_tool.community.multilevel_minimize(state, B=2)
    #graph_draw(ug,  vertex_fill_color=state.get_blocks(), output="community.png")
    #numero de blocos maximos que podem ser construidos a partir do grafo dado
    print(graph_tool.community.get_max_B(N=ug.num_vertices(), E=ug.num_edges()))

# Perfis de pessoas que costumam ser exploradas sexualmente ou vitima de trafico de pessoas
def genGraph_Sexual_Exploitation():
    ug = Graph()
    ug.set_directed(False)

    #edge colors
    alpha=0.15
    edge_color = ug.new_edge_property('vector<double>')
    ug.edge_properties['edge_color']=edge_color

    size = ug.new_edge_property('float')
    ug.edge_properties['size']=size

    #vertex colors
    vertex_color = ug.new_vertex_property('vector<double>')
    ug.vertex_properties['vertex_color'] = vertex_color
    #ug.vertex_properties['uf'] = plot_uf
    conn_string = "host='jabot.cos.ufrj.br' dbname='epinions' user='jabot' password='jabot'"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    
    query = "SELECT ID, sexo, uf_desaparecimento FROM registro where sexo ilike 'F%' and (uf_desaparecimento like 'RJ' or uf_desaparecimento like 'SP');"
    cursor.execute(query)
    rows = cursor.fetchall()
    
    ug.add_vertex(len(rows))
    '''
    vertices = ug.vertices()
    arestas = ug.edges()
    print(len(rows))
    pp(vertices)
    pp(arestas)
    '''
    for i in range(0, len(rows)):
        u = rows[i]
        for j in range(i+1, len(rows)):
            v = rows[j]
            if (u[1].lower() == v[1].lower()) and u[1].lower() == 'negra' or u[1].lower() == "Morena Escura" or u[1].lower() == "Mulato":
                ug.add_edge(u[0], v[0])

                vertex_color[ug.vertex(u[0])] = (25.0/255.0, 25.0/255.0, 112.0/255.0, 100)
                vertex_color[ug.vertex(v[0])] = (25.0/255.0, 25.0/255.0, 112.0/255.0, 100)
                edge_color[ug.edge(u[0], v[0])] = (25.0/255.0, 25.0/255.0, 112.0/255.0, alpha) #azul escuro

            elif u[2].lower() == v[2].lower():
                ug.add_edge(u[0], v[0])

                vertex_color[ug.vertex(u[0])] = (0.0/255.0, 255.0/255.0, 255.0/255.0, 100)
                vertex_color[ug.vertex(v[0])] = (0.0/255.0, 255.0/255.0, 255.0/255.0, 100)
                edge_color[ug.edge(u[0], v[0])] = (0.0/255.0, 255.0/255.0, 255.0/255.0, alpha) #azul claro
    
    ug = graph_tool.GraphView(ug,vfilt=lambda v: (v.out_degree() > 0) )
    ug.purge_vertices()
    
    graph_draw(ug, edge_color = ug.edge_properties['edge_color'], vertex_fill_color=ug.vertex_properties['vertex_color'], vertex_color=ug.vertex_properties['vertex_color'])#,vertex_text=ug.vertex_index, vertex_font_size=10, output="UF_corDaPele.png"


#grafo completo com os principais atributos como propriedade dos vertices
def genGraph():
    a =0
    ug = Graph()
    ug.set_directed(False)

    #edge colors
    alpha=0.15
    edge_color = ug.new_edge_property('vector<double>')
    ug.edge_properties['edge_color']=edge_color
    #edges weights
    edge_weights = ug.new_edge_property('int')
    ug.edge_properties['edge_weights']=edge_weights

    #vertex colors
    vertex_color = ug.new_vertex_property('vector<double>')
    ug.vertex_properties['vertex_color'] = vertex_color
    conn_string = "host='jabot.cos.ufrj.br' dbname='epinions' user='jabot' password='jabot'"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()

    query = "SELECT  ID, idade, uf_desaparecimento FROM registro where sexo ilike 'M%' and idade not like '' and (uf_desaparecimento like 'RJ' or  uf_desaparecimento like 'SP') limit 100"
    cursor.execute(query)
    rows = cursor.fetchall()
    
    tam = len(rows)

    ug.add_vertex(tam)
    
    for i in range(0, tam):
        u = rows[i]
        for j in range(i+1, tam):
            v = rows[j]
            
            if int(u[1]) == int(v[1]) and int(u[1]) < 29:
                ug.add_edge(u[0], v[0])
                vertex_color[ug.vertex(u[0])] = (25.0/255.0, 25.0/255.0, 112.0/255.0, 100)#azul claro
                vertex_color[ug.vertex(v[0])] = (25.0/255.0, 25.0/255.0, 112.0/255.0, 100)#azul claro
                edge_color[ug.edge(u[0], v[0])] = (0.0/255.0, 0.0/255.0, 0.0/255.0, alpha) #preto
                
                edge_weights[ug.edge(u[0], v[0])] = 1
                #print(size[ug.edge(u[0], v[0])])
                #pen_width #largura da aresta (0 a 1.0)
                #text# texto da aresta
            elif int(u[1]) == int(v[1]) and int(u[1]) >= 29:
                ug.add_edge(u[0], v[0])
                vertex_color[ug.vertex(u[0])] = (0.0/255.0, 255.0/255.0, 255.0/255.0, 100)# azul escuro
                vertex_color[ug.vertex(v[0])] = (0.0/255.0, 255.0/255.0, 255.0/255.0, 100)# azul escuro
                edge_color[ug.edge(u[0], v[0])] = (0.0/255.0, 0.0/255.0, 0.0/0.0, alpha) #azul claro
                 
                edge_weights[ug.edge(u[0], v[0])] = 1
                
                #print(size[ug.edge(u[0], v[0])])
            if u[2] == v[2] == 'SP': # !!!! Quando entrar em um dos ifs deve olhar qual a faixa etaria do vertice e pinta-lo com a cor correspondente
                ug.add_edge(u[0], v[0])
                edge_color[ug.edge(u[0], v[0])] = (171.0/255.0, 171.0/255.0, 171.0/255.0, alpha) # cinza
                edge_weights[ug.edge(u[0], v[0])] = edge_weights[ug.edge(u[0], v[0])]+1
            elif u[2] == v[2] == 'RJ':
                ug.add_edge(u[0], v[0])
                edge_weights[ug.edge(u[0], v[0])] = edge_weights[ug.edge(u[0], v[0])]+1
                edge_color[ug.edge(u[0], v[0])] = (195.0/255.0, 144.0/255.0, 212.0/255.0, alpha)#lilas

    ug = graph_tool.GraphView(ug )
    ug.purge_vertices()
    
    graph_draw(ug, edge_color = ug.edge_properties['edge_color'], vertex_fill_color=ug.vertex_properties['vertex_color'], vertex_color=ug.vertex_properties['vertex_color'])#,edge_text = ug.edge_properties['edge_weights'], vertex_text=ug.vertex_index, vertex_font_size=10, output="UF_corDaPele.png"


genGraph_Murderer()
#genGraph_Sexual_Exploitation()
#genGraph()
