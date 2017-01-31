import numpy as np
import networkx as nx
import random
from collections import defaultdict

class Graph():
  def __init__(self, nx_G, is_directed, p, q):
    self.G = nx_G
    self.is_directed = is_directed
    self.p = p
    self.q = q
    self.sampled_edges = defaultdict(set)

  def node2vec_walk(self, walk_length, start_node):
    '''
    Simulate a random walk starting from start node.
    '''
    G = self.G
    alias_nodes = self.alias_nodes
    alias_edges = self.alias_edges
    sampled_edges = self.sampled_edges

    walk = [start_node]
    while len(walk) < walk_length:
      cur = walk[-1]
      cur_nbrs = sorted(G.neighbors(cur))
      if len(cur_nbrs) > 0:
        if len(walk) == 1:
          next = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
          walk.append(next)
        else:
          prev = walk[-2]
          next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                     alias_edges[(prev, cur)][1])]
          walk.append(next)
      else:
        break
    
    return walk

  def simulate_walks(self, num_walks, walk_length, start_nodes):
    '''
    Repeatedly simulate random walks from each node.
    '''
    walks = []
    # print 'Walk iteration:'
    walk_cnt = 0
    for walk_iter in range(num_walks):
      # print str(walk_iter + 1), '/', str(num_walks)
      for node in start_nodes:
        walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        walk_cnt += 1
        if walk_cnt % 5000 == 0:
          print("Current walks: ", walk_cnt)
        if walk_cnt >= num_walks: break
      if walk_cnt >= num_walks: break

    return walks
  
  

  def get_alias_edge(self, src, dst):
    '''
    Get the alias edge setup lists for a given edge.
    '''
    G = self.G
    p = self.p
    q = self.q

    unnormalized_probs = []
    for dst_nbr in sorted(G.neighbors(dst)):
      if dst_nbr == src:
        unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
      elif G.has_edge(dst_nbr, src):
        unnormalized_probs.append(G[dst][dst_nbr]['weight'])
      else:
        unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)

  def preprocess_transition_probs(self):
    '''
    Preprocessing of transition probabilities for guiding the random walks.
    '''
    G = self.G
    is_directed = self.is_directed

    alias_nodes = {}
    node_cnt = 0
    for node in G.nodes():
      # Sampling probabilities of neighbors are in proportion to neighbor weights.
      unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
      norm_const = sum(unnormalized_probs)
      normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
      # A preprocess used to efficiently sample from multinomial distribution.
      alias_nodes[node] = alias_setup(normalized_probs)
      node_cnt += 1
      if node_cnt % 10000 == 0: print("Processed transitions for nodes: ", node_cnt)

    alias_edges = {}
    triads = {}
    
    edge_cnt = 0
    num_edges = len(G.edges())
    if is_directed:
      for edge in G.edges():
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        edge_cnt += 1
        if edge_cnt % 10000 == 0: print("Processed transitions for edges: "+str(edge_cnt)+"/"+str(num_edges))
    else:
      for edge in G.edges():
        alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

    self.alias_nodes = alias_nodes
    self.alias_edges = alias_edges

    return


def alias_setup(probs):
  '''
  Compute utility lists for non-uniform sampling from discrete distributions.
  Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
  for details
  '''
  K = len(probs)
  q = np.zeros(K)
  J = np.zeros(K, dtype=np.int)

  smaller = []
  larger = []
  for kk, prob in enumerate(probs):
    q[kk] = K * prob
    if q[kk] < 1.0:
      smaller.append(kk)
    else:
      larger.append(kk)

  while len(smaller) > 0 and len(larger) > 0:
    small = smaller.pop()
    large = larger.pop()

    J[small] = large
    q[large] = q[large] + q[small] - 1.0
    if q[large] < 1.0:
      smaller.append(large)
    else:
      larger.append(large)

  return J, q


def alias_draw(J, q):
  '''
  Draw sample from a non-uniform discrete distribution using alias sampling.
  '''
  K = len(J)

  kk = int(np.floor(np.random.rand() * K))
  if np.random.rand() < q[kk]:
    return kk
  else:
    return J[kk]
