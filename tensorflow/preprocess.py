import numpy as np
import six.moves.cPickle as pickle

DATA_PATH = '../data/test-net/'

LABEL_NUM = 1

graphs_train = {}
with open(DATA_PATH + 'random_walks_train.txt', 'r') as f:
    for line in f:
        walks = line.strip().split('\t')
        graphs_train[walks[0]] = []
        for i in range(1, len(walks)):
            graphs_train[walks[0]].append([int(x) for x in walks[i].split()])

graphs_val = {}
with open(DATA_PATH + 'random_walks_val.txt', 'r') as f:
    for line in f:
        walks = line.strip().split('\t')
        graphs_val[walks[0]] = []
        for i in range(1, len(walks)):
            graphs_val[walks[0]].append([int(x) for x in walks[i].split()])

graphs_test = {}
with open(DATA_PATH + 'random_walks_test.txt', 'r') as f:
    for line in f:
        walks = line.strip().split('\t')
        graphs_test[walks[0]] = []
        for i in range(1, len(walks)):
            graphs_test[walks[0]].append([int(x) for x in walks[i].split()])
            
labels_train = {}
sizes_train = {}
with open(DATA_PATH + 'cascade_train.txt', 'r') as f:
    for line in f:
        profile = line.split('\t')
        labels_train[profile[0]] = profile[-1]
        sizes_train[profile[0]] = int(profile[3])

labels_val = {}
sizes_val = {}
with open(DATA_PATH + 'cascade_val.txt', 'r') as f:
    for line in f:
        profile = line.split('\t')
        labels_val[profile[0]] = profile[-1]
        sizes_val[profile[0]] = int(profile[3])
        
labels_test = {}
sizes_test = {}
with open(DATA_PATH + 'cascade_test.txt', 'r') as f:
    for line in f:
        profile = line.split('\t')
        labels_test[profile[0]] = profile[-1]
        sizes_test[profile[0]] = int(profile[3])
        
        
class IndexDict:
    def __init__(self, original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        for i in original_ids:
            new = self.original_to_new.get(i,cnt)
            if new == cnt:
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)
    def new(self, original):
        if type(original) is int:
            return self.original_to_new[original]
        else:
            if type(original[0]) is int:
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]
    def original(self, new):
        if type(new) is int:
            return self.new_to_original[new]
        else:
            if type(new[0]) is int:
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]
    def length(self):
        return len(self.new_to_original)
    
    
original_ids = set()
for graph in graphs_train.keys():
    for walk in graphs_train[graph]:
        for i in set(walk):
            original_ids.add(i)
for graph in graphs_val.keys():
    for walk in graphs_val[graph]:
        for i in set(walk):
            original_ids.add(i)
for graph in graphs_test.keys():
    for walk in graphs_test[graph]:
        for i in set(walk):
            original_ids.add(i)

original_ids.add(-1)
index = IndexDict(original_ids)


x_data = []
y_data = []
sz_data = []
for key,graph in graphs_train.items():
    label = labels_train[key].split()
    y = int(label[LABEL_NUM])
    temp = []
    for walk in graph:
        if len(walk) < 10:
            for i in range(10 - len(walk)):
                walk.append(-1)
        temp.append(index.new(walk))
    x_data.append(temp)
    y_data.append(np.log(y+1.0)/np.log(2.0))
    sz_data.append(sizes_train[key])
    
pickle.dump((x_data, y_data, sz_data, index.length()), open('data/data_train.pkl','w'))

x_data = []
y_data = []
sz_data = []
for key,graph in graphs_val.items():
    label = labels_val[key].split()
    y = int(label[LABEL_NUM])
    temp = []
    for walk in graph:
        if len(walk) < 10:
            for i in range(10 - len(walk)):
                walk.append(-1)
        temp.append(index.new(walk))
    x_data.append(temp)
    y_data.append(np.log(y+1.0)/np.log(2.0))
    sz_data.append(sizes_val[key])
    
pickle.dump((x_data, y_data, sz_data, index.length()), open('data/data_val.pkl','w'))

x_data = []
y_data = []
sz_data = []
for key,graph in graphs_test.items():
    label = labels_test[key].split()
    y = int(label[LABEL_NUM])
    temp = []
    for walk in graph:
        if len(walk) < 10:
            for i in range(10 - len(walk)):
                walk.append(-1)
        temp.append(index.new(walk))
    x_data.append(temp)
    y_data.append(np.log(y+1.0)/np.log(2.0))
    sz_data.append(sizes_test[key])
    
pickle.dump((x_data, y_data, sz_data, index.length()), open('data/data_test.pkl','w'))


np.random.seed(13)
with open(DATA_PATH+'node_vec_50.txt', 'r') as f:
    line = f.readline()
    temp = line.strip().split()
    num_nodes = int(temp[0])
    num_dims = int(temp[1])
    node_vec = np.random.normal(size=(index.length(), num_dims))
    for i in range(num_nodes):
        line = f.readline()
        temp = line.strip().split()
        node_id = int(temp[0])
        if not node_id in original_ids:
            continue
        node_vec[index.new(node_id), :] = np.array([float(temp[j]) for j in range(1, len(temp))])
        
        
pickle.dump(node_vec, open('data/node_vec.pkl','w'))