# DeepCas
This repository provides a reference implementation of [*DeepCas* as described in the paper](https://arxiv.org/abs/1611.05373):
	
	DeepCas: an End-to-end Predictor of Information Cascades
	Cheng Li, Jiaqi Ma, Xiaoxiao Guo and Qiaozhu Mei
	World wide web (WWW), 2017

The *DeepCas* algorithm learns the representation of cascade graphs in an end-to-end manner for cascade prediction.

## Torch implementation
### Prerequisites
The following packages are required to install:

```{r, engine='bash', count_lines}
luarocks install cephes
luarocks install optim
luarocks install cudnn
luarocks install dp
luarocks install dpnn
luarocks install tds
luarocks install rnn
```

### Basic Usage

#### Example
To run *DeepCas* on a test data set, execute the following command:<br/>
```{r, engine='bash', count_lines}
cd DeepcasTorch
python gen_walks/gen_walks.py --dataset test-net
cd torch
th main/run.lua --dataset test-net
```
#### Options
You can check out the other options available to use with *DeepCas* using:<br/>
```{r, engine='bash', count_lines}
python gen_walks/gen_walks.py --help
th main/run.lua --help
```
#### Input
global_graph.txt lists each node's neighbors in the global network:

	node_id \t\t (null|neighbor_id:weight \t neighbor_id:weight...)

"\t" means tab, and "null" is used when a node has no neighbors.

cascade_(train|val|test).txt list cascades, one cascade per line:

	cascade_id \t starter_id... \t constant_field \t num_nodes \t source:target:weight... \t label...

"starter_id" are nodes who start the cascade, "num_nodes" counts the number of nodes in the cascade.
Since we can predict cascade growth at different timepoints, there could be multiple labels. 

## Tensorflow Implementation
### Prerequisites
Tensorflow 0.12.1

### Basic Usage
To run *DeepCas* tensorflow version on a test data set, execute the following command:<br/>
```{r, engine='bash', count_lines}
cd DeepCas
python gen_walks/gen_walks.py --dataset test-net
cd tensorflow
python preprocess.py
python run.py
```

### Citing
If you find *DeepCas* useful for your research, please consider citing the following paper:

	@inproceedings{DeepCas-www2017,
	author = {Li, Cheng and Ma, Jiaqi and Guo, Xiaoxiao and Mei, Qiaozhu},
	 title = {DeepCas: an End-to-end Predictor of Information Cascades},
	 booktitle = {Proceedings of the 26th international conference on World wide web},
	 year = {2017}
	}

## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <lichengz@umich.edu>.
