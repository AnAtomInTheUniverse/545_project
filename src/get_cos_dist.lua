--[[
--	Gets the cosine similarity between specified
--	weight layers of two models
--]]
require 'torch'
require 'optim'
require 'nn'

cmd = torch.CmdLine()
cmd:option('-m1','base', 'First model')
cmd:option('-m2', 'base', 'Second model')
cmd:option('-l', 1, 'Layer to compute distance for')
args = cmd:parse(arg)

model1 = torch.load('../models/' .. args.m1 .. '.model')
model2 = torch.load('../models/' .. args.m2 .. '.model')
layer = args.l

weights1, gradWeights1 = model1:parameters()
weights2, gradWeights2 = model2:parameters()

cos = nn.CosineDistance()
dist = cos:forward({weights1[layer], weights2[layer]})
print('Cosine Distance for Layer '.. layer)
print(dist) 


