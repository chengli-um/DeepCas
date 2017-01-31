require 'dp'
require 'rnn'
require 'nn.SeqBGRU'
require 'nn.GeoAttention'

-- Apply attention on sequence selection and words selection in one sequence.

local BiLSTMAtten = torch.class('nn.BiLSTMAtten')

-- Input is {graph_sizes {graph_sizes, seqs}}, seqs: batch x numSentence x numWords
-- graph_sizes: batch
local function buildModel(config)
   local hiddenSize = config.inputSize
   local dropoutProb = config.dropoutProb or 0.25
   local numSentence = config.size
   -- Every binSize elements share the same attention.
   local numBins = config.numBins
   local binSize = math.floor(numSentence / numBins)
   assert(numBins*binSize == numSentence)
   local numGraphSizes = config.numGraphSizes
   local batchSize = config.batchSize
   local numWords = config.maxWordsStory
   local vocSize = config.vocSize
   local activation = config.activation
   local addNonlin = config.addNonlin
   local globallocalWordAtten = config.globallocalWordAtten
   local globallocalSeqAtten = config.globallocalSeqAtten
   local attenType = config.attenType
   local batchFirst = true

   local mdl = nn.Sequential()
   local sizeLstmP = nn.ParallelTable()
   mdl:add(sizeLstmP)
   
   local attWords = nn.Sequential()
   local lstmS = nn.Sequential()
   sizeLstmP:add(attWords)
   sizeLstmP:add(lstmS)

   local lstmP = nn.ParallelTable()
   lstmS:add(lstmP)
   local attSeq = nn.Sequential()
   local lstm = nn.Sequential()
   lstmP:add(attSeq)
   lstmP:add(lstm)
   
   -- input is batch x numSentence x numWords
   lstm:add(nn.Convert())
   -- (batch * numSentence) x numWords
   lstm:add(nn.View(-1, numWords))
   local lookupTbl = nn.LookupTable(vocSize, hiddenSize)
   
   -- (batch * numSentence) x numWords x hiddenSize
   lstm:add(lookupTbl)
   if dropoutProb > 0 then
      lstm:add(nn.Dropout(dropoutProb))
   end
   
   -- (batch * numSentence) x numWords x 2hiddenSize
   lstm:add(nn.SeqBGRU(hiddenSize, hiddenSize, batchFirst, nn.JoinTable(2,2)))
   if addNonlin then
      lstm:add(nn[activation]())
   end

   local outSize = 2*hiddenSize
   -- batch x numSentence x (numWords*2hiddenSize)
   lstm:add(nn.View(-1, numSentence, (numWords*outSize)))
   -------------------------------
   -- Attention for sequences.
   -- batch x 1
   attSeq:add(nn.Convert())
   local seqProbTbl
   if globallocalSeqAtten then
      attSeq:add(nn.MulConstant(0, false))
      attSeq:add(nn.AddConstant(1, true))
      seqProbTbl = nn.LookupTable(1, 1)
   else
      seqProbTbl = nn.LookupTable(numGraphSizes, 1)
   end
   attSeq:add(seqProbTbl)
   attSeq:add(nn.Sigmoid())
   -- batch x numBins
   attSeq:add(nn.GeoAttention(batchSize, numBins))
   -- batch*numBins
   attSeq:add(nn.View(-1))
   -- (batch*numBins) x binSize
   attSeq:add(nn.Replicate(binSize,2))
   -- batch x (numBins*binSize) == batch x numSentence
   attSeq:add(nn.Reshape(batchSize,numBins*binSize))
   -- batch x numSentence x 1
   attSeq:add(nn.View(-1,numSentence,1))
   -- batch x 1 x (numWords*outSize)
   lstmS:add(nn.MM(true, false))
   -- batch x numWords x outSize
   lstmS:add(nn.View(-1, numWords, outSize))

   -------------------------------
   -- Attention for words(nodes) in a sequence.
   -- batch
   attWords:add(nn.Convert())
   
   local wordProbTbl, attenTbl
   if attenType == 1 then
      if globallocalWordAtten then
         attWords:add(nn.MulConstant(0, false))
         attWords:add(nn.AddConstant(1, true))
         wordProbTbl = nn.LookupTable(1, 1)
      else
         wordProbTbl = nn.LookupTable(numGraphSizes, 1)
      end
      
      attWords:add(wordProbTbl)
      attWords:add(nn.Sigmoid())
      -- batch x numWords
      attenTbl = nn.GeoAttention(batchSize, numWords)
      attWords:add(attenTbl)
   else
      if globallocalWordAtten then
         attWords:add(nn.MulConstant(0, false))
         attWords:add(nn.AddConstant(1, true))
         -- batch x numWords
         wordProbTbl = nn.LookupTable(1, numWords)
      else
         -- batch x numWords
         wordProbTbl = nn.LookupTable(numGraphSizes, numWords)
      end

      attWords:add(wordProbTbl)
      -- batch x numWords
      attenTbl = nn.SoftMax()
      attWords:add(attenTbl)
   end
   
   -- batch x numWords x 1
   attWords:add(nn.View(-1,numWords,1))
   -- batch x 1 x outSize
   mdl:add(nn.MM(true, false))
   -- batch x outSize
   mdl:add(nn.View(-1, outSize))
   
   return mdl, lookupTbl, seqProbTbl, wordProbTbl, outSize, attenTbl
end

-- Input is {graph_sizes {graph_sizes, seqs}}
-- seqs: batch x numSentence x numWords
function BiLSTMAtten:__init(config)
   self.config = config
   local nTasks = config.numTasks
   local lstms = nn.Sequential()
   
   local lstm, lookupTbl, seqProbTbl, wordProbTbl, outSizeSum, attenTbl = buildModel(config)
   self.lookupTbl = lookupTbl
   self.lookupTbl.sep = true
   self.seqProbTbls = {seqProbTbl}
   self.wordProbTbls = {wordProbTbl}
   self.attenTbls = {attenTbl }
   self.lstm = lstm
   
   lstms:add(lstm)
   self.outSizeSum = outSizeSum

   local inputSize = outSizeSum
   local pred_c, loss_module
   if nTasks > 1 then
      pred_c = nn.ConcatTable()
      loss_module = nn.ParallelCriterion()
   
      for _=1,nTasks do
         local ts = nn.Linear(inputSize,1)
         pred_c:add(ts)

         local loss = nn.MSECriterion()
         loss.sizeAverage = false
         
         loss_module:add(loss)
      end
   else
      pred_c = nn.Linear(inputSize,1)

      local loss = nn.MSECriterion()
      loss.sizeAverage = false
      
      loss_module = loss
   end
   
   lstms:add(pred_c)
   self.model = lstms

   local target_module = nn.Convert()

   self.target_module = target_module

   self.loss = loss_module
   self:setMode()
end

function BiLSTMAtten:restoreWeights()
   self.lookupTbl.weight:copy(self.config.fixedEmbedding)
end

function BiLSTMAtten:setMode()
   --------------------------------------
   -- Cuda
   if self.config.useCuda then
      require 'cutorch'
      require 'cunn'
      require 'cudnn'

      cutorch.setDevice(self.config.useDevice)

      self.model:cuda()
      self.target_module:cuda()
      self.loss:cuda()
      cudnn.convert(self.model, cudnn)
   end
end
