--[[
 
Reference implementation of DeepCas.

Author: Cheng Li

For more details, refer to the paper:
DeepCas: an End-to-end Predictor of Information Cascades
Cheng Li, Jiaqi Ma, Xiaoxiao Guo and Qiaozhu Mei
World wide web (WWW), 2017

]]--

require 'dp'
require 'optim'
require 'util.netParser'
local tds = require('tds')
require('nn.BiLSTMAtten')
local params = require('main.params')
local optMethod = require('util.optMethod')
local dataUtil = require('util.dataUtil')

local function getConfig(params, netParser)
   local config = {}
   config.inputSize = params.inputSize -- Dim of embedding vector mi.
   config.size = math.min(params.maxMemorySize, netParser.story:size(2))
   config.maxReadSize = config.size
   
   config.dropoutProb = params.dropoutProb
   config.vocSize = #netParser.vocab
   config.batchSize = params.batchSize
   config.addNonlin = params.addNonlin
   config.useCuda = params.useCuda
   config.useDevice = params.useDevice
   config.numTasks = 1
   config.sepEmbed = params.sepEmbed
   config.maxWordsStory = math.min(netParser.story:size(3), params.n_cellnodes)
   config.maxWordsStoryOrg = config.maxWordsStory
   config.maxWordsQuestion = netParser.qstory:size(2)
   config.activation = params.activation
   
   config.numBins = params.numBinsAtten
   config.numGraphSizes = netParser.graphSizes:max()
   config.globallocalWordAtten = params.globallocalWordAtten
   config.attenType = params.attenType
   config.globallocalSeqAtten = params.globallocalSeqAtten
   
   config.embedSize = config.inputSize
   config.projSize = config.embedSize
   
   config.maxWords = math.max(config.maxWordsStory, config.maxWordsQuestion)
   
   return config
end

-- Load data.
print("Loading data...")
local netParser = dp.netParser(params)
local vocab = netParser.vocab
local story, questions, qstory = netParser.story, netParser.questions, netParser.qstory
local valStory, valQuestions, valQstory = netParser.valStory, netParser.valQuestions, netParser.valQstory
local testStory, testQuestions, testQstory = netParser.testStory, netParser.testQuestions, netParser.testQstory
local trainGraphSizes, valGraphSizes, testGraphSizes = netParser.graphSizes, netParser.valGraphSizes, netParser.testGraphSizes

local function printStories(story, id2word)
   for i=1,story:size(1) do
      local str = {}
      local row = story[i]
      for j=1,story:size(2) do
         table.insert(str, id2word[row[j]])
      end
      print(table.concat(str, " "))
   end
end

-- Determine model configuration.
local config = getConfig(params, netParser)

params.maxWords = config.maxWords
params.size = config.size
params.maxReadSize = config.maxReadSize
params.maxWordsQuestion = config.maxWordsQuestion

-- Construct models.
print("Constructing model...")
local biLSTM = nn.BiLSTMAtten(config)
local optimMethod = params.optimizer == "SGD" and optMethod.sgd or optMethod.adam

io.output():setvbuf("line")

local trainRange = torch.range(1, questions:size(1))
local valRange = torch.range(1, valQuestions:size(1))
local testRange = torch.range(1, testQuestions:size(1))

--------------------------------------
-- Start training.

print("Initializing parameters...")
local parameters, gradParameters, paramsEmb, gradParamsEmb = dataUtil.getParametersSepEmb(biLSTM.model)
-- Initialization.
parameters:normal(0, params.initStd)
if params.initEmbed then
   local weight = biLSTM.lookupTbl.weight
   dataUtil.loadNodePreEmbedding(params, weight, netParser.vocab,
      netParser.nodeIdStart, netParser.numNodes)
end
if params.initAttenWeights then
   assert(#biLSTM.seqProbTbls == 1)
   assert(#biLSTM.wordProbTbls == 1)
   
   local function initLinespace(weight, startWeight, endWeight)
      local lastAtten = endWeight
      if weight:size(1) == 1 then
         lastAtten = startWeight
      end
      local lineInit = torch.linspace(startWeight, lastAtten, weight:size(1))
      weight:copy(lineInit)
   end
   
   local function initMultinomial(weight, firstK)
      local fillVal = 5
      for i=1,weight:size(2) do
         if i > firstK then
            fillVal = -5
         end
         weight:select(2,i):fill(fillVal)
      end
   end
   
   local weight = biLSTM.seqProbTbls[1].weight
   initLinespace(weight, params.localSeqAtten1, params.localSeqAtten2)
   
   if params.attenType == 1 then
      local weight = biLSTM.wordProbTbls[1].weight
      initLinespace(weight, params.localWordAtten1, params.localWordAtten2)
   else
      local weight = biLSTM.wordProbTbls[1].weight
      initMultinomial(weight, params.localWordAtten1)
   end
end

local function getEmbConfig(params)
   local embConfig = {
      coefL1 = params.coefL1,
      weightDecay = params.weightDecay,
      lr = params.embLR,
      maxGradNorm = params.maxGradNorm
   }
   return embConfig
end

local constantNode = dataUtil.nodePrefix..dataUtil.constNodeSymbl
local randInd = torch.LongTensor(params.batchSize)
local selectedQuestions = torch.Tensor(params.batchSize, questions:size(2))
local trueYs = torch.Tensor(params.batchSize)
local memoryData = torch.Tensor(config.batchSize, config.maxReadSize, config.maxWordsStory)
local graphSizes = torch.Tensor(config.batchSize)

-- tryTimes: used to decide early stop.
local minMSE, minMSETest, tryTimes, maxEpoch = 1000, 1000, 0, 0

params.lr = params.initLR
local embConfig = getEmbConfig(params)
local function sampleTrainBatchData(params)
   randInd:random(1, trainRange:size(1))
   -- Answers.
   selectedQuestions:index(questions, 1, randInd)
   trueYs:copy(selectedQuestions:select(2,selectedQuestions:size(2)))
   
   memoryData:fill(vocab[dataUtil.nilSymbl])
   
   -- Run one batch.
   for b = 1, params.batchSize do
      local d = story[{selectedQuestions[b][1], {1, params.maxReadSize}, {1, config.maxWordsStoryOrg}}]
      local n_sentences = d:size(1)
      
      memoryData[{ b, { 1, n_sentences }, { 1, d:size(2) } }]:copy(d)
   
      graphSizes[b] = trainGraphSizes[selectedQuestions[b][1]]
   end
end

local trainPredictions
if params.savePredictions then
   local numBatches = math.min(params.epochBatches, torch.floor(trainRange:size(1) / params.batchSize))
   trainPredictions = torch.Tensor(numBatches * params.batchSize, 2)
end

local weightBuff, lookupWeight

local function runTrainBatch(params, costs, times, ithBatch)
   local input = {graphSizes, {graphSizes, memoryData}}
   
   local out = biLSTM.model:forward(input)
   local target = biLSTM.target_module:forward(trueYs)
   costs.totalCost = costs.totalCost + biLSTM.loss:forward(out, target)
   
   if params.savePredictions then
      local predBatch = trainPredictions:narrow(1,(ithBatch-1)*params.batchSize+1,params.batchSize)
      predBatch:select(2, 1):copy(out)
      predBatch:select(2, 2):copy(target)
   end
   
   gradParameters:zero()
   gradParamsEmb:zero()
   
   local grad = biLSTM.loss:backward(out, target)
   biLSTM.model:backward(input, grad)
   
   optMethod.regularize(parameters, gradParameters, params)
   optimMethod(parameters, gradParameters, params)
   
   optMethod.regularize(paramsEmb, gradParamsEmb, embConfig)
   optimMethod(paramsEmb, gradParamsEmb, embConfig)
end


local function runTrainEpoch(epoch, params)
   local costs = {totalCost = 0, costSenti = 0, totalNum = 0}
   local times = {fill_data = 0, forward = 0, backward = 0, optimize = 0}
   
   biLSTM.model:training()
   
   -- Run one epoch.
   local numBatches = math.min(params.epochBatches, torch.floor(trainRange:size(1) / params.batchSize))
   for b = 1, numBatches do
      costs.totalNum = costs.totalNum + params.batchSize
      
      local time = sys.clock()
      sampleTrainBatchData(params)
      times.fill_data = times.fill_data + sys.clock() - time
      
      runTrainBatch(params, costs, times, b)
      if params.progress then
         xlua.progress(b*params.batchSize, numBatches*params.batchSize)
      end
   end
   if epoch % 5 == 0 then
      collectgarbage()
   end
   
   local trainCost = costs.totalCost / costs.totalNum
   
   print('# ' .. epoch .. ', train: MSE=' .. trainCost)
end


local function sampleValBatchData(k, params, testQuestions, testStory, testGraphSizes)
   randInd:range(1 + (k - 1) * params.batchSize, k * params.batchSize)
   -- Answers.
   selectedQuestions:index(testQuestions, 1, randInd)
   trueYs:copy(selectedQuestions:select(2,selectedQuestions:size(2)))
   
   memoryData:fill(vocab[dataUtil.nilSymbl])
   
   for b = 1, params.batchSize do
      local d = testStory[{selectedQuestions[b][1], {1, params.maxReadSize}, {1, config.maxWordsStoryOrg}}]
      local n_sentences = d:size(1)
      
      memoryData[{ b, { 1, n_sentences }, { 1, d:size(2) } }]:copy(d)
   
      graphSizes[b] = testGraphSizes[selectedQuestions[b][1]]
   end
end

local nbatchTest = torch.floor(testRange:size(1) / params.batchSize)
local nbatchVal = torch.floor(valRange:size(1) / params.batchSize)
local nTest = nbatchTest*params.batchSize
local predictions = torch.Tensor(nTest)
local expId = os.date("%Y_%m_%d_%H_%M_%S")
predictions = torch.cat({predictions, testQuestions:select(2,testQuestions:size(2)):narrow(1, 1, nbatchTest*params.batchSize)}, 2)
local nNodeBatch = torch.floor(#vocab / params.batchSize)
local nNodes = nNodeBatch*params.batchSize
local lastHidden, lastHiddenNodes


local function runValEpoch(epoch, params, testQuestions, testStory, testGraphSizes, nbatchTest, isTest)
   local totalValCost = 0
   local totalValNum = 0
   
   biLSTM.model:evaluate()
   for k = 1, nbatchTest do
      sampleValBatchData(k, params, testQuestions, testStory, testGraphSizes)
      
      local input = {graphSizes, {graphSizes, memoryData}}
      local out = biLSTM.model:forward(input)      
      local target = biLSTM.target_module:forward(trueYs)
      totalValCost = totalValCost + biLSTM.loss:forward(out, target)
      if isTest and params.savePredictions then
         local predRecord = predictions:narrow(1, (k-1)*params.batchSize+1, params.batchSize)
         predRecord:select(2, 1):copy(out)
      end
      totalValNum = totalValNum + params.batchSize
   end
   
   local valCost = totalValCost / totalValNum
   return valCost
end

for ep = 1, params.nEpochs do
   if ep % params.lrDecayStep == 0 then
      params.lr = params.lr * 0.5
   end
   
   runTrainEpoch(ep, params)
   local valCost = runValEpoch(ep, params, valQuestions, valStory, valGraphSizes, nbatchVal, false)
   local testCost = runValEpoch(ep, params, testQuestions, testStory, testGraphSizes, nbatchTest, true)
   
   tryTimes = tryTimes + 1
   if valCost < minMSE then
      minMSE = valCost
      minMSETest = testCost
      tryTimes = 0
      maxEpoch = ep
      if params.savePredictions then
         torch.save(params.saveModelPrefix..'best_pred_'..expId..'.t7', {testCost, predictions, trainPredictions, valCost})
      end
   end
   print(params.dataset .. " " .. params.ithLabel)
   print('Val MSE\t' .. valCost ..'\tepoch\t' .. ep ..'\ttest MSE\t' .. testCost)
   print('-----------------------------\n')
   print('Min val MSE\t' .. minMSE ..'\tat epoch\t' .. maxEpoch ..'\tmin test MSE\t' .. minMSETest)
   if tryTimes >= params.maxTries then
      break
   end
end
print("====================")
