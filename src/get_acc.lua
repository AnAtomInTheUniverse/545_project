--[[
-- Computes the accuracy of classifications
-- predicted by the output.lua script using 
-- models and data from ../models/ and ../data/
-- ]]
require 'torch'
require 'optim'

classes = {'0','1','2','3','4','5','6','7','8','9'}

cmd = torch.CmdLine()
cmd:option('-p', 'predicted label file')
cmd:option('-t', 'true label file')
args = cmd:parse(arg)

pred_temp = torch.load(args.p)
true_temp = torch.load(args.t)

pred = torch.DoubleTensor(pred_temp:size()):copy(pred_temp)
true_labels = torch.DoubleTensor(true_temp:size()):copy(true_temp)
confusion = optim.ConfusionMatrix(classes)

for i = 1,pred:size(1) do
	local temp = torch.zeros(10)
	temp[pred[i][1]+1] = 1
	confusion:add(temp,true_labels[i][1]+1)
end
print(confusion)
acc = torch.sum(torch.eq(pred, true_labels))/50000
print("Accuracy: " .. acc)
