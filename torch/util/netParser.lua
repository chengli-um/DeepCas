--
-- Parser for data of memory format.
-- Transform training data from text to matrix representation
--

require 'dp'
require 'sys'
require 'cephes'
local stringx = require('pl.stringx')
local file = require('pl.file')
local dataUtil = require('util.dataUtil')

local netParser = torch.class('dp.netParser')

function netParser:__init(params)
   self.params = params
   local ithLabel = params.ithLabel
   local logLabel = params.logLabel or true
   
   self.vocab, self.id2word, self.nodeIdStart, self.numNodes = dataUtil.getDict(params)
   local storySet, questionsSet, qstorySet = self:parseData(params, params.graphWalkPrefix, params.memoryPrefix, true)
   self.story = storySet[1]
   self.valStory = storySet[2]
   self.testStory = storySet[3]

   local indices = torch.LongTensor{1,2,ithLabel+2}
   self.questions = questionsSet[1]:index(2, indices)
   self.valQuestions = questionsSet[2]:index(2, indices)
   self.testQuestions = questionsSet[3]:index(2, indices)

   self.qstory = qstorySet[1]
   self.valQstory = qstorySet[2]
   self.testQstory = qstorySet[3]
   
   if logLabel then
      self:logLabels(self.questions)
      self:logLabels(self.valQuestions)
      self:logLabels(self.testQuestions)
   end
   
   local graphSizeSet, maxSize = dataUtil.getGraphSizes(params)
   self.graphSizes = graphSizeSet[1]
   self.valGraphSizes = graphSizeSet[2]
   self.testGraphSizes = graphSizeSet[3]
   assert(self.graphSizes:max() >= self.testGraphSizes:max())
   collectgarbage()
end


function netParser:logLabels(questions)
   local lastColumn = questions:select(2,questions:size(2))
   local logged = cephes.log2(lastColumn+1)
   lastColumn:copy(logged)
end

-- return the dimensionality for the data
local function analyzeData(fname, vocab, includeQuestion, isTrain, checkVocab)
   -- Story index.
   local storyInd = 0
   local sentenceInd = 0
   -- Max number of words in a sentence of a story.
   local maxWordsStory = 0
   local maxWordsQuestion = 0
   local maxSentences = 0
   local numLabels = -1
   
   local questionInd = 0
   local lineInd = 0

   local fh,err = io.open(fname)
   if err then error("Error reading file: " .. fname); end

   while true do
      local line = fh:read()
      if line == nil then break end

      lineInd = lineInd + 1

      local parts = stringx.split(line, '\t')
      -- New story.
      if parts[1] == "1" then
         storyInd = storyInd + 1
         sentenceInd = 0
      end
      -- question or not
      local isQuestion = false
      if parts[2] == "s" then
         isQuestion = false
         sentenceInd = sentenceInd + 1
      else
         isQuestion = true
         questionInd = questionInd + 1
   
         if includeQuestion then
            sentenceInd = sentenceInd + 1
         end
      end

      local words = stringx.split(parts[3])
      -- Determine the maximum number of words in a sentence.
      if not isQuestion then
         if maxWordsStory < #words then
            maxWordsStory = #words
         end
      end

      if isTrain and checkVocab then
         for k = 1, #words do
            local w = dataUtil.nodePrefix..words[k]
            assert(vocab[w], "Unseen word: " .. w .. ', in: '..line)
         end
      end

      if isQuestion then
         if(numLabels < 0) then
            local labels = stringx.split(parts[4], " ")
            numLabels = #labels
         end
   
         if maxWordsQuestion < #words then
            maxWordsQuestion = #words
         end
      end

      if maxSentences < sentenceInd then
         maxSentences = sentenceInd
      end
   end

   fh:close()
   
   return storyInd, maxSentences, maxWordsStory, maxWordsQuestion, questionInd, numLabels
end

-- processing data
local function parseOneFile(fname, vocab, includeQuestion, numStory, maxSentence, maxWordsStory,
   maxWordsQuestion, numQuestion, numLabels, isTrain, checkVocab)
   assert(numStory == numQuestion, "Num story: "..numStory..", num questions: "..numQuestion)
   
   -- story(i,j,k) stores the id of k-th word of j-th sentence of i-th story.
   local story = torch.Tensor(numStory, maxSentence, maxWordsStory):fill(vocab[dataUtil.nilSymbl])

   -- story index.
   local storyInd = 0
   local sentenceInd = 0
   -- i-th row of questions stores meta info for i-th question.
   -- row: 1: storyInd; 2: sentenceInd of the question in the story; 3-end: labels
   local questions = torch.Tensor(numQuestion, 2+numLabels):fill(vocab[dataUtil.nilSymbl])
   local questionInd = 0
   
   -- qstory(i,j) stores the id of j-th word of i-th question sentence.
   local qstory = torch.Tensor(numQuestion, maxWordsQuestion):fill(vocab[dataUtil.nilSymbl])
   local lineInd = 0

   local fh,err = io.open(fname)
   if err then error("Error reading file: " .. fname); end

   while true do
      local line = fh:read()
      if line == nil then break end

      lineInd = lineInd + 1

      local parts = stringx.split(line, '\t')
      -- new story
      if parts[1] == "1" then
         storyInd = storyInd + 1
         sentenceInd = 0
      end
      -- question or not
      local isQuestion = false
      if parts[2] == "s" then
         isQuestion = false
         sentenceInd = sentenceInd + 1
      else
         isQuestion = true
         questionInd = questionInd + 1
         questions[questionInd][1] = storyInd
         questions[questionInd][2] = math.min(sentenceInd, maxSentence)
         if includeQuestion then
            sentenceInd = sentenceInd + 1
         end
      end

      local words = stringx.split(parts[3])

      -- compute xi
      local wcnt = 0
      for k = 1, #words do
         local w = dataUtil.nodePrefix..words[k]
         local wid = vocab[w]
         if wid then
            wcnt = wcnt + 1
            if not isQuestion then
               story[storyInd][sentenceInd][wcnt] = wid
            else
               qstory[questionInd][wcnt] = wid
               if includeQuestion then
                  story[storyInd][sentenceInd][wcnt] = wid
               end
            end
         else
            if checkVocab then
               assert(not isTrain)
            end
         end
      end

      if isQuestion then
         local labels = stringx.split(parts[4], " ")
         for i=1,#labels do
            questions[questionInd][2+i] = tonumber(labels[i])
         end
      end
   end

   fh:close()
   return story, questions, qstory
end

function netParser:parseData(params, graphWalkPrefix, memoryPrefix, checkVocab)
   local includeQuestion = params.includeQuestion
   local cache_file = memoryPrefix..'cache.t7'
   if paths.filep(cache_file) then
      return table.unpack(torch.load(cache_file))
   end

   dataUtil.cascade2MemoryFormat(params, graphWalkPrefix, memoryPrefix)
   
   local maxWordsStoryAll, maxWordsQuestionAll, maxSentenceAll = 0, 0, 0
   local numStorySet, numQuestionSet, numLabelsSet = {}, {}, {}
   for i,set in ipairs(dataUtil.sets) do
      local memFile = memoryPrefix..set..".txt"
      local numStory, maxSentence, maxWordsStory, maxWordsQuestion, numQuestion, numLabels = 
         analyzeData(memFile, self.vocab, includeQuestion, i==1, checkVocab)
      maxWordsStoryAll = math.max(maxWordsStoryAll, maxWordsStory)
      maxWordsQuestionAll = math.max(maxWordsQuestionAll, maxWordsQuestion)
      maxSentenceAll = math.max(maxSentenceAll, maxSentence)
      table.insert(numStorySet, numStory)
      table.insert(numQuestionSet, numQuestion)
      table.insert(numLabelsSet, numLabels)
   end

   local storySet, questionsSet, qstorySet = {}, {}, {}
   for i,set in ipairs(dataUtil.sets) do
      local memFile = memoryPrefix..set..".txt"
      local story, questions, qstory = parseOneFile(memFile, self.vocab, includeQuestion, numStorySet[i],
         maxSentenceAll, maxWordsStoryAll, maxWordsQuestionAll, numQuestionSet[i], numLabelsSet[i], i==1, checkVocab)
      table.insert(storySet, story)
      table.insert(questionsSet, questions)
      table.insert(qstorySet, qstory)
   end
   
   torch.save(cache_file, {storySet, questionsSet, qstorySet})
   return storySet, questionsSet, qstorySet
end

