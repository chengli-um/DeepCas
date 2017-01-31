
-- Optimization methods

require 'dp'
local optMethod = {}

--[[ An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
- 'config.learningRate'      : learning rate
- 'config.beta1'             : first moment coefficient
- 'config.beta2'             : second moment coefficient
- 'config.epsilon'           : for numerical stability
- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified
]]
function optMethod.adam(parameters, gradParams, params)
   local x = parameters
   local dfdx = gradParams
   
   -- (0) get/update state
   local config = params or {}
   local state = config
   local lr = config.lr or 0.001
   
   local beta1 = config.beta1 or 0.9
   local beta2 = config.beta2 or 0.999
   local epsilon = config.epsilon or 1e-8
   
   -- (1) evaluate df/dx
   -- Initialization
   state.t = state.t or 0
   -- Exponential moving average of gradient values
   state.m = state.m or x.new(dfdx:size()):zero()
   -- Exponential moving average of squared gradient values
   state.v = state.v or x.new(dfdx:size()):zero()
   -- A tmp tensor to hold the sqrt(v) + epsilon
   state.denom = state.denom or x.new(dfdx:size()):zero()
   -- Current gradient
   state.grad = state.grad or x.new(dfdx:size()):zero()
   
   state.t = state.t + 1
   
   -- Decay the first and second moment running average coefficient
   state.m:mul(beta1):add(1-beta1, dfdx)
   state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
   
   state.denom:copy(state.v):sqrt():add(epsilon)
   
   local biasCorrection1 = 1 - beta1^state.t
   local biasCorrection2 = 1 - beta2^state.t
   local stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
   -- (2) update parameters. += -stepSize * lrVec * (state.m / state.denom)
--   x:addcdiv(-stepSize, state.m, state.denom)
   -- Clip gradients.
   state.grad:cdiv(state.m, state.denom)
   -- L2 norm: math.sqrt(x:pow(2):sum())
   local gn = state.grad:norm()
   if (gn > params.maxGradNorm) then
      state.grad:mul(params.maxGradNorm / gn)
   end

   x:add(-stepSize, state.grad)
end

function optMethod.sgd(parameters, gradParams, params)
   parameters:add(gradParams:mul(-params.lr))
end

function optMethod.regularize(parameters, gradParams, config)
   -- L1 Regularization
   if config.coefL1 ~= 0 then
      gradParams:add(torch.sign(parameters):mul(config.coefL1))
   end
   
   -- L2 Regularization
   if config.weightDecay ~= 0 then
      gradParams:add(config.weightDecay, parameters)
   end
end

return optMethod
