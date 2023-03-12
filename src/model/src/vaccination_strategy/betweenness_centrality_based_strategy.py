import random
import networkx as nx
from operator import itemgetter
import dgl
# Betweenness centrality is going to be the measure based on which we are going to vaccinate the population
def betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):

    betweenness = dict.fromkeys(G, 0.0)
    if k is None:
        nodes = G
    else:
        random.seed(seed)
        nodes = random.sample(G.nodes(), k)
    for s in nodes:

        # single source shortest path
        if weight is None: 
            S, P, sigma = nx.single_source_shortest_path_basic(G, s)

        # accumulation of the nodes
        if endpoints:
            betweenness = nx.accumulate_endpoints(betweenness, S, P, sigma, s)
        else:
            betweenness = nx.accumulate_basic(betweenness, S, P, sigma, s)

    betweenness = nx.rescale(betweenness, len(G), normalized=normalized,
            directed=G.is_directed(), k=k)
    return betweenness

def vaccination_strategy(model):
    contact_data = model.df
    cd_array1 = contact_data['p1'].to_numpy()
    cd_array2 = contact_data['p2'].to_numpy()
    metadata = model.metadata
    g = dgl.graph((cd_array1, cd_array2), num_nodes = len(metadata.index))
    G = nx.gnm_random_graph(len(metadata.index), len(cd_array1), seed=None, directed=False)
    b = nx.betweenness_centrality(G)
    nodes_to_be_vaccinated = dict(sorted(b.items(), key = itemgetter(1), reverse = True)[:int(0.2*(len(metadata.index)))])
    return nodes_to_be_vaccinated
