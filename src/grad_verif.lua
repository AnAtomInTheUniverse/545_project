#!/usr/bin/lua
--[[	
		Check Gradient
--]]
require 'torch'
require 'optim'
require 'nn'
paths.dofile( '../models/modules.lua')
-- local params
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local batch = 1
local filt = 2
local img_dim = 4

local input = torch.Tensor(batch, filt, img_dim, img_dim):fill(1)
local grad_output = torch.Tensor(batch, filt*4, img_dim, img_dim):zero()
grad_output:fill(math.random(0,255))
print(grad_output[{{1},{1},{},{}}])
local module = nn.CycSlice(input, grad_output)
module:updateGradInput(input, grad_output)

-- test backprop with Jacobian
local err = jac.testJacobian(module, input)
print('==> error: ' .. err)
if err < precision then
	print(' --> module OK')
else
	print('--> ERROR too large, incorrect implementation!')
end

--input = torch.tensor(1,1,4,4):fill(1) 
--err, corr, est = optim.checkgrad(CycSlice.updateGradInput, input)
--print(string.format('Error: %f\n Correct: %f \n Estimate: %d\n', err, corr, est))
