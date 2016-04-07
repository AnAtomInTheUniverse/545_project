require 'image'

--------------------------------------------------------------

local CycSlice = torch.class('nn.CycSlice', 'nn.Module')

function CycSlice:updateOutput(input)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local oldBatch=input:size(1)
	self.output = torch.Tensor(oldBatch*4,input:size(2),input:size(3),input:size(4))

	for i = 1,oldBatch do
		self.output[{i,{},{},{}}] = input[{i,{},{},{}}]
		self.output[{oldBatch + i,{},{},{}}] = torch.bmm(input[{i,{},{},{}}]:transpose(2,3),rot)
		self.output[{2*oldBatch + i,{},{},{}}] = torch.bmm(rot,torch.bmm(input[{i,{},{},{}}],rot))
		self.output[{3*oldBatch + i,{},{},{}}] = torch.bmm(rot,input[{i,{},{},{}}]:transpose(2,3))
	end
	return self.output
end

----------------------------------------------------------------

local MeanCycPool = torch.class('nn.MeanCycPool', 'nn.Module')

function MeanCycPool:updateOutput(input)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local newBatch = input:size(1)/4
	self.output = torch.zeros(newBatch,input:size(2),input:size(3),input:size(4))
	for i = 1,newBatch do
		self.output[{i,{},{},{}}] = ( input[{i,{},{},{}}]	
				+ torch.bmm(rot,input[{newBatch + i,{},{},{}}]:transpose(2,3)) 
				+ torch.bmm(rot,torch.bmm(input[{newBatch*2 + i,{},{},{}}],rot)) 
				+ torch.bmm(input[{newBatch*3 + i,{},{},{}}]:transpose(2,3),rot))/4
	end
	--self.output:copy(output)
	return self.output
end

function MeanCycPool:updateGradInput(input,gradOutput)
	local rot = torch.eye(gradOutput:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(gradOutput:size(2),1,1)	
	--self.gradInput:copy(input)
	local oldBatch = gradOutput:size(1)
	self.gradInput = torch.zeros(input:size())
	for i = 1,oldBatch do 
		self.gradInput[{i,{},{},{}}] = gradOutput[{i,{},{},{}}]/4
		self.gradInput[{oldBatch + i,{},{},{}}] = torch.bmm(gradOutput[{i,{},{},{}}]:transpose(2,3),rot)/4
		self.gradInput[{2*oldBatch + i,{},{},{}}] = torch.bmm(rot,torch.bmm(gradOutput[{i,{},{},{}}],rot))/4
		self.gradInput[{3*oldBatch + i,{},{},{}}] = torch.bmm(rot,gradOutput[{i,{},{},{}}]:transpose(2,3))/4
	end
	return self.gradInput
end








