require 'image'

--------------------------------------------------------------

local CycSlice = torch.class('nn.CycSlice', 'nn.Module')

function CycSlice:updateOutput(input)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local batch=input:size(1)
	self.output = torch.Tensor(batch*4,input:size(2),input:size(3),input:size(4))

	for i = 1,batch do
		self.output[{i,{},{},{}}] = input[{i,{},{},{}}]
		self.output[{batch + i,{},{},{}}] = torch.bmm(input[{i,{},{},{}}]:transpose(2,3),rot) --perform 90 rotation
		self.output[{2*batch + i,{},{},{}}] = torch.bmm(rot,torch.bmm(input[{i,{},{},{}}],rot)) --perform 180 rotation
		self.output[{3*batch + i,{},{},{}}] = torch.bmm(rot,input[{i,{},{},{}}]:transpose(2,3)) -- perform 270 rotation
	end
	return self.output
end

function CycSlice:updateGradInput(input,gradOutput)
	local rot = torch.eye(gradOutput:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(gradOutput:size(2),1,1)
	local batch = input:size(1)
	self.gradInput = torch.zeros(input:size())
	for i = 1,inBatch do
		self.gradInput[{i,{},{},{}}}] = 
				self.gradOutput[{i,{},{},{}}]
				+ torch.bmm(rot,self.gradOutput[{batch + i,{},{},{}}]:transpose(2,3)) --perform 270 rotation
				+ torch.bmm(rot,torch.bmm(self.gradOutput[{2*batch + i,{},{},{}}],rot)) -- perform 180 rotation
				+ torch.bmm(self.gradOutput[{3*batch + i,{},{},{}}]:transpose(2,3),rot) -- perform 90 rotation
	end
	return self.gradInput
end
----------------------------------------------------------------

local MeanCycPool = torch.class('nn.MeanCycPool', 'nn.Module')

function MeanCycPool:updateOutput(input)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local batch = input:size(1)/4
	self.output = torch.zeros(newBatch,input:size(2),input:size(3),input:size(4))
	for i = 1,batch do
		self.output[{i,{},{},{}}] = ( input[{i,{},{},{}}]	
				+ torch.bmm(rot,input[{batch + i,{},{},{}}]:transpose(2,3)) --perform 270 rotation
				+ torch.bmm(rot,torch.bmm(input[{batch*2 + i,{},{},{}}],rot)) --perform 180 rotation
				+ torch.bmm(input[{batch*3 + i,{},{},{}}]:transpose(2,3),rot))/4 --perform 90 rotation
	end
	--self.output:copy(output)
	return self.output
end

function MeanCycPool:updateGradInput(input,gradOutput)
	local rot = torch.eye(gradOutput:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(gradOutput:size(2),1,1)	
	--self.gradInput:copy(input)
	local batch = gradOutput:size(1)
	self.gradInput = torch.zeros(input:size())
	for i = 1,batch do 
		self.gradInput[{i,{},{},{}}] = gradOutput[{i,{},{},{}}]/4
		self.gradInput[{batch + i,{},{},{}}] = torch.bmm(gradOutput[{i,{},{},{}}]:transpose(2,3),rot)/4 --perform 90 rotation
		self.gradInput[{2*batch + i,{},{},{}}] = torch.bmm(rot,torch.bmm(gradOutput[{i,{},{},{}}],rot))/4 -- perform 180 rotation
		self.gradInput[{3*batch + i,{},{},{}}] = torch.bmm(rot,gradOutput[{i,{},{},{}}]:transpose(2,3))/4 --perform 270 rotation
	end
	return self.gradInput
end








