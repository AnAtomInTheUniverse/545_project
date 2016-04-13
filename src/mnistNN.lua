--Train model for MNIST digit recognition
require 'nn'
require 'math'
require 'optim'
require 'torch'
classes = {'1','2','3','4','5','6','7','8','9','10'}


--Load Data
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
local ftrain_data = '../data/mnist_' .. opt.data .. '_train_data.th7'
local ftrain_labels = '../data/mnist_' .. opt.data.. '_train_labels.th7'


trainData_temp = torch.load(ftrain_data);
trainData = trainData_temp:clone()
imdim = math.sqrt(trainData:size(2))
trainData:resize(trainData:size(1),1,imdim,imdim);
trainLabels = torch.load(ftrain_labels) + 1;


--Normalize Data
mean = trainData:mean()
std = trainData:std()
trainData:add(-mean);
trainData:mul(1/std);

--SGD Characteristics
sgdState = {
	learningRate = opt.LR,
	momentum = opt.momentum,
	learningRateDecay = opt.LRdecay
}

if opt.method == 'nag' then
	sgdState.momentum = 0.5
end

--Create training and validation sets
valData = trainData[{{opt.numTrain+1,opt.numTrain + opt.numVal},{},{},{}}]
valLabels = trainLabels[{{opt.numTrain+1,opt.numTrain + opt.numVal},{}}]
trainData = trainData[{{1,opt.numTrain},{},{},{}}]
trainLabels = trainLabels[{{1,opt.numTrain},{}}]

--ind = torch.randperm(opt.numTrain + opt.numVal)
--valData = trainData[{{ind[]},{},{},{}}]


--Define Model
paths.dofile('../models/'.. opt.model .. '.lua')
--[[model = nn.Sequential()
model:add( nn.SpatialConvolutionMM(1,20,5,5) )
model:add( nn.ReLU() )
model:add( nn.SpatialMaxPooling(2,2,2,2))
model:add( nn.SpatialConvolution(20,40,5,5) )
model:add( nn.ReLU() )
model:add( nn.SpatialMaxPooling(2,2,2,2))
model:add( nn.Reshape(40*4*4) )
model:add( nn.Linear(40*4*4,200) )
model:add( nn.Tanh() )
model:add( nn.Linear(200,10) )--]]

parameters,gradParameters = model:getParameters()

--Create criterion and add additional layer
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

--Create Confusion Matrix
confusion = optim.ConfusionMatrix(classes)

for i = 1,opt.nEpochs do
	print('Epoch ' .. i)
	
------Training----------
	index = torch.randperm(opt.nIters*opt.batchSize)
	for j = 1,opt.nIters do
		local inputs, targets
		for k = 1,opt.batchSize do
			if k == 1 then 
				inputs = trainData[{{index[(j-1)*opt.batchSize + k]},{},{},{}}]
				targets = trainLabels[index[(j-1)*opt.batchSize + k]]
			else 
				inputs = torch.cat(inputs,trainData[{{index[(j-1)*opt.batchSize + k]},{},{},{}}],1);
				targets = torch.cat(targets, trainLabels[index[(j-1)*opt.batchSize + k]],1)
			end
		end


		local feval = function(x)
			-- just in case:
			collectgarbage()

			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

			-- evaluate function for complete mini batch
			local outputs = model:forward(inputs)
			local f = criterion:forward(outputs, targets)

			-- estimate df/dW
			local df_do = criterion:backward(outputs, targets)
			model:backward(inputs, df_do)

			-- penalties (L1 and L2):
			if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
			-- locals:
				local norm,sign= torch.norm,torch.sign

				-- Loss:
				f = f + opt.coefL1 * norm(parameters,1)
				f = f + opt.coefL2 * norm(parameters,2)^2/2

				-- Gradients:
				gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
			end

			 for l = 1,opt.batchSize do
          confusion:add(outputs[l], targets[l])
       end

			--if math.fmod(j,10) == 0 then print('Loss: ' .. f) end
			return f,gradParameters
		end

		if opt.method == 'nag' then
			optim.nag(feval,parameters,sgdState)
		else
			optim.sgd(feval,parameters,sgdState)
		end


	end
	print('Train Confusion')
	print(confusion)
	confusion:zero()
	
--------Validation---------
	for j = 1,opt.numVal/opt.batchSize do
		local inputs, targets
		for k = 1,opt.batchSize do
			if k == 1 then 
				inputs = valData[{{(j-1)*opt.batchSize + k},{},{},{}}]
				targets = valLabels[(j-1)*opt.batchSize + k]
			else 
				inputs = torch.cat(inputs,valData[{{(j-1)*opt.batchSize + k},{},{},{}}],1);
				targets = torch.cat(targets, valLabels[(j-1)*opt.batchSize + k],1)
			end
		end
		local preds = model:forward(inputs)
		for l = 1,opt.batchSize do
         		confusion:add(preds[l], targets[l])
    		end
	end
	print('Test Confusion')
	print(confusion)
	confusion:zero()


	if math.fmod(i,5) == 0 then torch.save(opt.save,model) end
end


torch.save(opt.save,model)



