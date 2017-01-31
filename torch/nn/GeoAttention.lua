-- Given input batch x 1 (a vector of probs), output an attention tensor of size batch x len. 
-- k-th cell a_k = (1-p)^(k-1) * p.

local GeoAttention, parent = torch.class('nn.GeoAttention', 'nn.Module')

function GeoAttention:__init(batch, len)
   parent.__init(self)
   assert(len > 0, "Length must be larger than 0.")
   self.batch = batch
   self.len = len
   self.output = torch.Tensor(batch, len)
   self.probBuff = torch.Tensor(batch, 1)
   self.probBuffk = torch.Tensor(batch, 1)
   self.onesVec = torch.ones(batch, 1)
   self.gradInput = torch.Tensor(batch, 1)
   -- maxIter to compute gradients.
   self.maxIter = math.min(len, 8)
end

-- input is batch x 1.
function GeoAttention:updateOutput(input)
   local output = self.output
   local onesVec = self.onesVec
   local probBuff = self.probBuff
   
   local lastCol = output:select(2,1)
   lastCol:copy(input)
   if self.len < 2 then
      return self.output
   end
   
   -- 1-p
   probBuff:add(onesVec, -1, input)
   
   for i=2,self.len do
      local curCol = output:select(2,i)
      curCol:cmul(lastCol, probBuff)
      lastCol = curCol
   end

   return self.output
end

-- Compute gradient of input.
function GeoAttention:updateGradInput(input, gradOutput)
   local gradInput = self.gradInput
   local onesVec = self.onesVec
   local probBuff = self.probBuff
   local probBuffk = self.probBuffk

   gradInput:copy(gradOutput:select(2,1))
   if self.len < 2 then
      return self.gradInput
   end

   -- 1-2p
   probBuffk:add(onesVec, -2, input)
   gradInput:addcmul(probBuffk, gradOutput:select(2,2))
   
   probBuff:fill(1)
   for i=3,self.maxIter do
      -- (1-p)^(k-2)
      probBuff:cmul(1-input)
      -- 1-p*k
      probBuffk:add(onesVec, -i, input)
      -- (1-p*k) * (1-p)^(k-2)
      probBuffk:cmul(probBuff)
      gradInput:addcmul(probBuffk, gradOutput:select(2,i))
   end
   
   return self.gradInput
end

-- Compute gradient of weights.
function GeoAttention:accGradParameters(input, gradOutput, scale)
end

-- we do not need to accumulate parameters when sharing
GeoAttention.sharedAccUpdateGradParameters = GeoAttention.accUpdateGradParameters
