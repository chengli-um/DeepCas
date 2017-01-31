--
-- Util functions for data.
--
require 'dp'
require 'optim'
require 'cephes'
local stringx = require('pl.stringx')
local file = require('pl.file')
local tds = require('tds')
local dataUtil = {}
----------------------------------------
-- constants.
dataUtil.startSymbl = "<s>"
dataUtil.endSymbl = "</s>"
-- Dummy node used to fill up short sentence.
dataUtil.nilSymbl = "<nil>"
dataUtil.nilWordSymbl = "<nilWord>"
dataUtil.sentiSymbl = '<senti>'
dataUtil.mentionSymbl = '<@>'
dataUtil.unknownSymbl = '<unk>'
dataUtil.constNodeSymbl = '<node>'
dataUtil.nodePrefix = 'NODE-'


dataUtil.sets = {"train", "val", "test"}

----------------------------------------

local function getGraphSizesForFile(filename, isTrain, maxSize)
   local fh,err = io.open(filename)
   if err then error("Error reading file: " .. filename); end
   
   if isTrain then
      maxSize = 0
   end

   local graphSizes = {}
   -- Cascade format: graph_id \t [author_id ] \t org_date \t num_nodes \t [source:target:weight ] \t [label ] \t text
   while true do
      local line = fh:read()
      if (line == nil) then break end
   
      local fields = stringx.split(line, '\t')
      local graphSize = tonumber(fields[4])
      if isTrain then
         if maxSize < graphSize then
            maxSize = graphSize
         end
      else
         if graphSize > maxSize then
            graphSize = maxSize
         end
      end
      table.insert(graphSizes, graphSize)
   end
   fh:close()
   local gSize = torch.Tensor(graphSizes)
   cephes.log2(gSize, gSize+1)
   gSize:floor()
   return maxSize, gSize
end

function dataUtil.getGraphSizes(params)
   local sizeFile = paths.concat(params.dataPath, "graph_sizes.t7")
   
   if dp.is_file(sizeFile) then
      return table.unpack(torch.load(sizeFile))
   end
   
   local maxSize = 0
   local gSize
   local graphSizeSet = {}
   for i, which_set in ipairs(dataUtil.sets) do
      maxSize, gSize = getGraphSizesForFile(params.cascadePrefix..which_set..".txt", i==1, maxSize)
      table.insert(graphSizeSet, gSize)
   end

   torch.save(sizeFile, {graphSizeSet, maxSize})
   return graphSizeSet, maxSize
end

local function addToDict(word, word2id, id2word, removeStopwords)
   if (word == nil or word == '') then
      return nil
   end
   
   local id = word2id[word]
   if not id then
      id = #word2id + 1
      word2id[word] = id
      id2word[id] = word
   end
   return id
end


local function addNodesAsWords(filename, isWalkFile, word2id, id2word)
   local fh,err = io.open(filename)
   if err then error("Error reading file: " .. filename); end

   if isWalkFile then
      -- Graphwalk format: graph_id \t walk1 - [node ] \t walk2 ..
      while true do
         local line = fh:read()
         if line == nil then break end
         local fields = stringx.split(line, '\t')
   
         for i = 2, #fields do
            local nodes = stringx.split(fields[i], " ");
   
            for t = 1,#nodes do
               addToDict(dataUtil.nodePrefix..nodes[t], word2id, id2word, false)
            end
         end
      end
   else
      -- Cascade format: graph_id \t [author_id ] \t org_date \t num_nodes \t [source:target:weight ] \t [label ] \t text
      while true do
         local line = fh:read()
         if (line == nil) then break end
   
         local fields = stringx.split(line, '\t')
         local rootArr = stringx.split(fields[2], ' ')
         for t = 1,#rootArr do
            addToDict(dataUtil.nodePrefix..rootArr[t], word2id, id2word, false)
         end
         
         local edgeArr = stringx.split(fields[5], ' ')
         for t = 1,#edgeArr do
            local nodes = stringx.split(edgeArr[t], ':')
            if #nodes > 2 then
               addToDict(dataUtil.nodePrefix..nodes[1], word2id, id2word, false)
               addToDict(dataUtil.nodePrefix..nodes[2], word2id, id2word, false)
            end
         end
      end
   end
   fh:close()
end


-- Dict for nodes.
function dataUtil.getDict(params)
   local dictFile = paths.concat(params.dataPath, "dict.t7")
   
   if dp.is_file(dictFile) then
      return table.unpack(torch.load(dictFile))
   end
   
   local word2id = tds.hash()
   local id2word = tds.Vec()
   
   local nodeIdStart = #word2id + 1
   addToDict(dataUtil.nilSymbl, word2id, id2word, false)
   addToDict(dataUtil.nodePrefix..dataUtil.constNodeSymbl, word2id, id2word, false)
   --   addNodesAsWords(params.graphWalkPrefix.."train.txt", true, word2id, id2word)
   addNodesAsWords(params.cascadePrefix.."train.txt", false, word2id, id2word)

   local numNodes = #word2id - nodeIdStart + 1
   print("Number of nodes: " .. numNodes)
   
   torch.save(dictFile, {word2id, id2word, nodeIdStart, numNodes})
   return word2id, id2word, nodeIdStart, numNodes
end



function dataUtil.loadNodePreEmbedding(params, orgEmbedMat, word2id, nodeIdStart, numNodes)
   local orgNodeMat = orgEmbedMat:narrow(1,nodeIdStart,numNodes)
   local embedFile = params.nodeVecFile .. ".t7"
   if dp.is_file(embedFile) then
      local preEmbedMat = torch.load(embedFile)
      assert(preEmbedMat:size(1) == orgNodeMat:size(1), "Pre-trained embedding has different node numbers.")
      if(preEmbedMat:size(2) > orgNodeMat:size(2)) then
         preEmbedMat = preEmbedMat:narrow(2,1,orgNodeMat:size(2))
      elseif(preEmbedMat:size(2) < orgNodeMat:size(2)) then
         orgNodeMat = orgNodeMat:narrow(2,1,preEmbedMat:size(2))
      end
      orgNodeMat:copy(preEmbedMat)
      return
   end

   local embedMat = torch.Tensor(orgNodeMat:size()):copy(orgNodeMat)
   
   local fh,err = io.open(params.nodeVecFile)
   if err then error("Error reading file: " .. params.nodeVecFile); end

   local foundWords = 0
   local initSize
   while true do
      local line = fh:read()
      if line == nil then break end
      local vec = stringx.split(line)
      if #vec > 5 then
         if not initSize then
            initSize = math.min(#vec-1, embedMat:size(2))
         end
         
         local word = dataUtil.nodePrefix..vec[1]
         local id = word2id[word]
      
         if id then
            id = id - (nodeIdStart - 1)
            assert(id > 0 and id <= numNodes)
            foundWords = foundWords + 1
         
            local word_vec = embedMat[id]
            for t = 2,initSize do
               word_vec[t-1] = tonumber(vec[t])
            end
         end
      end
   end
   fh:close()
   
   print("Found node num in pre-trained embedding: "..foundWords..", not found: "..(numNodes-foundWords))
   
   torch.save(embedFile, embedMat)
   orgNodeMat:copy(embedMat)
end

-- Graphwalk format: graph_id \t walk1 - [node ] \t walk2 ..
-- Cascade format: graph_id \t [author_id ] \t org_date \t num_nodes \t [source:target:weight ] \t [label ] \t text
local function cascade2MemFile(graphWalkPrefix, cascadePrefix, memoryPrefix, which_set)
   local memFile = memoryPrefix..which_set..".txt"
--   if dp.is_file(memFile) then
--      return
--   end
   
   local graphWalkFile = graphWalkPrefix..which_set..".txt"
   local cascadeFile = cascadePrefix..which_set..".txt"
   
   local fhWalk,err = io.open(graphWalkFile)
   if err then error("Error reading file: " .. graphWalkFile); end

   local fhCascade,err = io.open(cascadeFile)
   if err then error("Error reading file: " .. cascadeFile); end
   local file = io.open(memFile, "w")
   
   while true do
      local lineWalk = fhWalk:read()
      local lineCascade = fhCascade:read()
      if (lineWalk == nil or lineCascade == nil) then break end

      local fields = stringx.split(lineCascade, '\t')
      local roots = fields[2]
      local rootArr = stringx.split(roots, ' ')
      local labels = fields[6]
      local fields = stringx.split(lineWalk, '\t')
      -- "s" stands for "story".
      local s_idx = 1
      for t = 2,#fields do
         file:write(s_idx.."\ts\t"..fields[t]..'\n')
         s_idx = s_idx + 1
      end

      -- Write question. "q" stands for "question".
      file:write(s_idx.."\tq\t"..roots..'\t'..labels..'\n')
   end

   fhWalk:close()
   fhCascade:close()
   
   file:close()
end

function dataUtil.cascade2MemoryFormat(params, graphWalkPrefix, memoryPrefix)
   for _,set in ipairs(dataUtil.sets) do
      cascade2MemFile(graphWalkPrefix, params.cascadePrefix, memoryPrefix, set)
   end
end


function dataUtil.shallowcopyTable(orig)
   local copy = {}
   for orig_key, orig_value in pairs(orig) do
      copy[orig_key] = orig_value
   end
   return copy
end


-- Return a list of flattened modules.
function dataUtil.listModules(model)
   local moduleList = {}
   for _,module in ipairs(model:listModules()) do
      if module.weight or module.bias or module.name then
         table.insert(moduleList, module)
      end
   end
   return moduleList
end


-- Separate the params of embedding from other model parameters.
-- lookupTbl.sep should be set to true if want to be separated.
-- Input: a table of modules.
-- Return: paramsNoEmb, gradParamsNoEmb, paramsEmb, gradParamsEmb.
function dataUtil.getParametersSepEmb(model)
   local function moduleParams(module)
      if module.weight then
         return {module.weight, module.bias}, {module.gradWeight, module.gradBias}
      end
   end
   
   local function tinsert(to, from)
      if not from then
         return
      end
      
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   
   local function flattenParams(parameters, gradParameters)
      local p, g = nn.Module.flatten(parameters), nn.Module.flatten(gradParameters)
      assert(p:nElement() == g:nElement(),
         'check that you are sharing parameters and gradParameters')
      if parameters then
         for i=1,#parameters do
            assert(parameters[i]:storageOffset() == gradParameters[i]:storageOffset(),
               'misaligned parameter at ' .. tostring(i))
         end
      end
      return p, g
   end
   
   local moduleList = dataUtil.listModules(model)
   
   local w = {}
   local gw = {}
   local w_emb = {}
   local gw_emb = {}
   
   for _,module in ipairs(moduleList) do
      local mw,mgw = moduleParams(module)
      if module.sep and torch.typename(module) == "nn.LookupTable" then
         tinsert(w_emb, mw)
         tinsert(gw_emb,mgw)
      else
         tinsert(w, mw)
         tinsert(gw,mgw)
      end
   end
   local paramsNoEmb, gradParamsNoEmb = flattenParams(w, gw)
   local paramsEmb, gradParamsEmb = flattenParams(w_emb, gw_emb)
   
   return paramsNoEmb, gradParamsNoEmb, paramsEmb, gradParamsEmb
end


return dataUtil
