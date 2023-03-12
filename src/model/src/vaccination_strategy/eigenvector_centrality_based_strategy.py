import networkx as nx
from operator import itemgetter
from math import sqrt
import dgl

def eigenvector_centrality(G, max_iterations=100, lim=1.0e-6, nstart=None, weight ='weight'):
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise nx.NetworkXException("Not defined for multigraphs.")
  
    if len(G) == 0:
        raise nx.NetworkXException("Empty graph.")
  
    if nstart is None:
        x = dict([(n,1.0/len(G)) for n in G])
    else:
        x = nstart
  
    s = 1.0/sum(x.values())
    for k in x:
        x[k] *= s
    n_nodes = G.number_of_nodes()
  
    for i in range(max_iterations):
        x_last = x
        x = dict.fromkeys(x_last, 0)
  
        for n in x:
            for entry in G[n]:
                x[entry] += x_last[n] * G[n][entry].get(weight, 1)
  
        # normalize vector
        try:
            s = 1.0/sqrt(sum(v**2 for v in x.values()))
  
        except ZeroDivisionError:
            s = 1.0
        for n in x:
            x[n] *= s
  
        # check convergence
        err = sum([abs(x[n]-x_last[n]) for n in x])
        if err < n_nodes*lim:
            return x 

def vaccination_strategy(model):
    contact_data = model.df
    cd_array1 = contact_data['p1'].to_numpy()
    cd_array2 = contact_data['p2'].to_numpy()
    metadata = model.metadata
    g = dgl.graph((cd_array1, cd_array2), num_nodes = len(metadata.index))
    G = nx.gnm_random_graph(len(metadata.index), len(cd_array1), seed=None, directed=False)
    e = nx.eigenvector_centrality(G)
    nodes_to_be_vaccinated = dict(sorted(e.items(), key = itemgetter(1), reverse = True)[:int(0.2*(len(metadata.index)))])
    return nodes_to_be_vaccinated
    
