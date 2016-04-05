--[[
-- Computes the accuracy of classifications
-- predicted by the output.lua script using 
-- models and data from ../models/ and ../data/
-- ]]
require 'torch'

cmd = torch.CmdLine()
cmd:option('-p', 'predicted label file')
cmd:option('-t', 'true label file')
args = cmd:parse(arg)

pred_temp = torch.load(args.p)
true_temp = torch.load(args.t)

pred = torch.DoubleTensor(pred_temp:size()):copy(pred_temp)
true_labels = torch.DoubleTensor(true_temp:size()):copy(true_temp)

acc = torch.sum(torch.eq(pred, true_labels))/50000
print("Accuracy: " .. acc)
