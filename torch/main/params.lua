-- Set up the configuration and hyper-parameters for DeepCas.
require 'dp'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Cascade prediction using DeepCas.')
cmd:text('Example:')
cmd:text('$> th main/run.lua --batchSize 128 --useCuda')
cmd:text('Options:')
cmd:option('--useCuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--dataRoot', '../data/')
cmd:option('--dataset', 'test-net', 'data set')
cmd:option('--progress', 1, 'print progress bar.')
cmd:option('--savePredictions', 0, 'save predictions.')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--epochBatches', 100, 'number of batches per epoch')
cmd:option('--nEpochs', 50)
cmd:option('--maxTries', 10, 'maximum number of epochs to try to find a better local minima for early-stopping.')
cmd:option('--ithLabel', 1, 'use i-th label in the label array, indicating the cascade growth at time i.')

-- Learning rate.
cmd:option('--lrDecayStep', 3)
cmd:option('--initLR', 0.01)
cmd:option('--embLR', 1e-5, "learning rate of embeddings.")
cmd:option('--initStd', 0.1, "std for parameter initilization.")
cmd:option('--maxGradNorm', 100)
cmd:option('--optimizer', "Adam", "SGD, Adam")
cmd:option('--coefL1', 5e-06, "L1 Regularization")
cmd:option('--weightDecay', 5e-09, "L2 Regularization")

-- Structure.
cmd:option('--inputSize', 50, "Dim of embedding vector mi.")
cmd:option('--maxMemorySize', 100, 'number of memory cells. 100 enought for |allData|.')
cmd:option("--n_cellnodes", 10, "Number of max nodes per cell.")
cmd:option('--addNonlin', 1, "add non-linearity to internal states")
cmd:option('--initEmbed', 1, "initialize learnable embedding with pre-trained ones.")
cmd:option('--dropoutProb', 0, 'if > 0, apply dropout on embeddings.')

-- LSTM
cmd:option('--activation', 'Tanh', 'transfer function like ReLU, Tanh, Sigmoid.')
cmd:option('--numBinsAtten', 5, 'number of bins for sequences in attention.')
cmd:option('--initAttenWeights', 1, 'init attention weights.')
cmd:option('--localSeqAtten1', 0, 'initialized attention.')
cmd:option('--localSeqAtten2', 1, 'initialized attention.')
cmd:option('--globallocalSeqAtten', 0, 'learn the same attention across graphs.')
cmd:option('--localWordAtten1', 2, 'initialized attention.')
cmd:option('--localWordAtten2', 2, 'initialized attention.')
cmd:option('--samelocalWordAtten', 1, 'initialized the same attention across graphs.')
cmd:option('--globallocalWordAtten', 1, 'learn the same attention across graphs.')
cmd:option('--attenType', 2, 'Attention for nodes in a sequence. 1: geometric; 2: multinomial')

cmd:text()
local params = cmd:parse(arg or {})

if params.samelocalWordAtten == 1 then
   params.localWordAtten2 = params.localWordAtten1
end

params.globallocalSeqAtten = params.globallocalSeqAtten == 1
params.globallocalWordAtten = params.globallocalWordAtten == 1
params.savePredictions = params.savePredictions == 1
params.initEmbed = params.initEmbed == 1
params.progress = params.progress == 1
params.addNonlin = params.addNonlin == 1
params.initAttenWeights = params.initAttenWeights == 1
params.graphAttention = params.graphAttention == 1

-- Data paths.
params.dataPath = paths.concat(params.dataRoot, params.dataset)
params.logDir = paths.concat(params.dataPath, "logs")
dp.mkdir(params.logDir)

-- Random walks for each cascade, one cascade per line.
-- Format: graph_id \t walk1 - [node ] \t walk2 ..
params.graphWalkPrefix = paths.concat(params.dataPath, "random_walks_")
-- Pretrained node embeddings using node2vec.
params.nodeVecFile = paths.concat(params.dataPath, "node_vec_"..params.inputSize..".txt")
-- Intermediate files.
params.memoryPrefix = paths.concat(params.dataPath, "memnet_")
-- Cascade file. Format: graph_id \t [author_id ] \t org_date \t num_nodes \t [source:target:weight ] \t [label ]
params.cascadePrefix = paths.concat(params.dataPath, "cascade_")

params.saveModelPrefix = paths.concat(params.logDir, "bigru-attention-"..params.ithLabel..'_')

params.includeQuestion = false

return params
