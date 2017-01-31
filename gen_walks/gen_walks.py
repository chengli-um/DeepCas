"""
Generate random walk paths for each cascade graph and pre-train node embeddings.

Adapted from node2vec [1].

[1] node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016

"""

import numpy as np
import os
from optparse import OptionParser
import sys
import networkx as nx
import node2vec
import time
from gensim.models import Word2Vec

# parse commandline arguments
op = OptionParser()
op.add_option("--data_root", dest="data_root", type="string", default="data/",
              help="data root.")
op.add_option("--dataset", dest="dataset", type="string", default="test-net",
              help="data set.")
op.add_option("--walks_per_graph", dest="walks_per_graph", type=int, default=200,
              help="number of walks per graph.")
op.add_option("--walk_length", dest="walk_length", type=int, default=10,
              help="length of each walk.")
op.add_option("--trans_type", dest="trans_type_str", type="string", default="edge",
              help="Type of function for transition probability: edge, deg, and DEG.")
# node2vec params.
op.add_option("--p", dest="p", type=float, default=1.0,
              help="Return hyperparameter in node2vec.")
op.add_option("--q", dest="q", type=float, default=1.0,
              help="Inout hyperparameter in node2vec.")
# word2vec params.
op.add_option('--dimensions', dest="dimensions", type=int, default=50,
              help='Number of dimensions of embedding.')
op.add_option('--window_size', dest="window_size", type=int, default=10,
              help='Context size for optimization. Default is 10.')
op.add_option('--iter', dest="iter", default=5, type=int, help='Number of epochs in SGD')
op.add_option('--workers', dest="workers", type=int, default=8, help='Number of parallel workers.')

(opts, args) = op.parse_args()
if len(args) > 0:
  op.error("this script takes no arguments.")
  sys.exit(1)

if opts.trans_type_str == "edge":
  opts.trans_type = 0
elif opts.trans_type_str == "deg":
  opts.trans_type = 1
else:
  assert opts.trans_type_str == "DEG", "%s: unseen transition type." % opts.trans_type_str
  opts.trans_type = 2

opts.data_root = os.path.expanduser(opts.data_root)
data_path = os.path.join(opts.data_root, opts.dataset)
# Format: node_id \t\t null|[target_id:freq \t ]
global_graph_file = os.path.join(data_path, "global_graph.txt")
# Format: graph_id \t [author_id ] \t org_date \t num_nodes \t [source:target:weight ] \t [label ]
cascade_file_prefix = os.path.join(data_path, "cascade_")
# Output file. Format: graph_id \t walk1 - [node ] \t walk2 ..
graph_walk_prefix = os.path.join(data_path, "random_walks_")
# Output file, which contains pretrained embeddings by word2vec.
embed_prefix = os.path.join(data_path, "node_vec_")
sets = ["train", "val", "test"]

node_to_degree = dict()
edge_to_weight = dict()
pseudo_count = 0.01

def get_global_info():
  rfile = open(global_graph_file, 'r')
  for line in rfile:
    line = line.rstrip('\r\n')
    parts = line.split("\t\t")
    source = long(parts[0])
    if parts[1] != "null":
      node_freq_strs = parts[1].split("\t")
      for node_freq_str in node_freq_strs:
        node_freq = node_freq_str.split(":")
        weight = int(node_freq[1])
        target = long(node_freq[0])
        if opts.trans_type == 0:
          edge_to_weight[(source, target)] = weight
      degree = len(node_freq_strs)
    else:
      degree = 0
    node_to_degree[source] = degree
  rfile.close()
  return

def get_global_degree(node):
  return node_to_degree.get(node, 0)

def get_edge_weight(source, target):
  return edge_to_weight.get((source, target), 0)


def parse_graph(graph_string):
  parts = graph_string.split("\t")
  edge_strs = parts[4].split(" ")

  node_to_edges = dict()
  for edge_str in edge_strs:
    edge_parts = edge_str.split(":")
    source = long(edge_parts[0])
    target = long(edge_parts[1])

    if not source in node_to_edges:
      neighbors = list()
      node_to_edges[source] = neighbors
    else:
      neighbors = node_to_edges[source]
    neighbors.append((target, get_global_degree(target)))

  nx_G = nx.DiGraph()
  for source, nbr_weights in node_to_edges.iteritems():
    for nbr_weight in nbr_weights:
      target = nbr_weight[0]

      if opts.trans_type == 0:
        edge_weight = get_edge_weight(source, target) + pseudo_count
        weight = edge_weight
      elif opts.trans_type == 1:
        target_nbrs = node_to_edges.get(target, None)
        local_degree = 0 if target_nbrs is None else len(target_nbrs)
        local_degree += pseudo_count
        weight = local_degree
      else:
        global_degree = nbr_weight[1] + pseudo_count
        weight = global_degree

      nx_G.add_edge(source, target, weight=weight)
  # List of the starting nodes.
  roots = list()
  # List of the starting nodes excluding nodes without outgoing neighbors.
  roots_noleaf = list()

  str_list = list()
  str_list.append(parts[0])

  probs = list()
  probs_noleaf = list()
  weight_sum_noleaf = 0.0
  weight_sum = 0.0

  # Obtain sampling probabilities of roots.
  for node, weight in nx_G.out_degree_iter(weight="weight"):
    org_weight = weight
    if weight == 0: weight += pseudo_count
    weight_sum += weight
    if org_weight > 0:
      weight_sum_noleaf += weight

  for node, weight in nx_G.out_degree_iter(weight="weight"):
    org_weight = weight
    if weight == 0: weight += pseudo_count
    roots.append(node)
    prob = weight / weight_sum
    probs.append(prob)
    if org_weight > 0:
      roots_noleaf.append(node)
      prob = weight / weight_sum_noleaf
      probs_noleaf.append(prob)

  sample_total = opts.walks_per_graph
  first_time = True
  G = node2vec.Graph(nx_G, True, opts.p, opts.q)
  G.preprocess_transition_probs()

  while True:
    if first_time:
      first_time = False
      node_list = roots
      prob_list = probs
    else:
      node_list = roots_noleaf
      prob_list = probs_noleaf
    n_sample = min(len(node_list), sample_total)
    if n_sample <= 0: break
    sample_total -= n_sample

    sampled_nodes = np.random.choice(node_list, n_sample, replace=False, p=prob_list)
    walks = G.simulate_walks(len(sampled_nodes), opts.walk_length, sampled_nodes)
    for walk in walks:
      str_list.append(' '.join(str(k) for k in walk))
  return '\t'.join(str_list)

def file_len(fname):
  lines = 0
  for line in open(fname):
    lines += 1
  return lines

def read_graphs(which_set):
  graph_cnt = 0
  graph_file = cascade_file_prefix + which_set + ".txt"
  graph_walk_file = graph_walk_prefix + which_set + ".txt"
  num_graphs = file_len(graph_file)
  write_file = open(graph_walk_file, 'w')
  rfile = open(graph_file, 'r')
  start_time = time.time()
  for line in rfile:
    line = line.rstrip('\r\n')
    walk_string = parse_graph(line)
    write_file.write(walk_string + "\n")
    graph_cnt += 1
    if graph_cnt % 1000 == 0: print("Processed graphs in %s set: %d/%d"%(which_set, graph_cnt, num_graphs))
  print("--- %.2f seconds per graphs in %s set ---" % ((time.time() - start_time)/graph_cnt, which_set))
  rfile.close()
  write_file.close()

def read_walks_set(which_set, walks):
  graph_walk_file = graph_walk_prefix + which_set + ".txt"
  rfile = open(graph_walk_file, 'r')
  for line in rfile:
    line = line.rstrip('\r\n')
    walk_strings = line.split('\t')
    for i,walk_str in enumerate(walk_strings):
      if(i == 0): continue
      walks.append(walk_str.split(" "))
  rfile.close()

def learn_embeddings(walks, embeding_size):
  embed_file = embed_prefix + str(embeding_size) + ".txt"
  # Learn embeddings by optimizing the Skipgram objective using SGD.
  model = Word2Vec(walks, size=embeding_size, window=opts.window_size, min_count=0, sg=1, workers=opts.workers,
                   iter=opts.iter)
  model.save_word2vec_format(embed_file)

if __name__ == "__main__":
  get_global_info()
  for which_set in sets:
    read_graphs(which_set)

  print("Train word2vec with dimension " + str(opts.dimensions))
  walks = list()
  read_walks_set(sets[0], walks)

  learn_embeddings(walks, opts.dimensions)