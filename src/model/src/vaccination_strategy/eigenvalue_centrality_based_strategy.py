import networkx as nx
from operator import itemgetter
import dgl
def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None, weight ='weight'):
    from math import sqrt
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise nx.NetworkXException("Not defined for multigraphs.")
  
    if len(G) == 0:
        raise nx.NetworkXException("Empty graph.")
  
    if nstart is None:
  
        # choose starting vector with entries of 1/len(G)
        x = dict([(n,1.0/len(G)) for n in G])
    else:
        x = nstart
  
    # normalize starting vector
    s = 1.0/sum(x.values())
    for k in x:
        x[k] *= s
    nnodes = G.number_of_nodes()
  
    # make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
  
        # do the multiplication y^T = x^T A
        for n in x:
            for nbr in G[n]:
                x[nbr] += xlast[n] * G[n][nbr].get(weight, 1)
  
        # normalize vector
        try:
            s = 1.0/sqrt(sum(v**2 for v in x.values()))
  
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in x:
            x[n] *= s
  
        # check convergence
        err = sum([abs(x[n]-xlast[n]) for n in x])
        if err < nnodes*tol:
            return x 

def vaccination_strategy(model):
    contact_data = model.df
    cd_array1 = contact_data['p1'].to_numpy()
    cd_array2 = contact_data['p2'].to_numpy()
    metadata = model.metadata
    #metadata_array = metadata.to_numpy()
    g = dgl.graph((cd_array1, cd_array2), num_nodes = len(metadata.index))
    G = nx.gnm_random_graph(len(metadata.index), len(cd_array1), seed=None, directed=False)
    e = nx.eigenvector_centrality(G)
    nodes_to_be_vaccinated = dict(sorted(e.items(), key = itemgetter(1), reverse = True)[:int(0.2*(len(metadata.index)))])
    return nodes_to_be_vaccinated
    