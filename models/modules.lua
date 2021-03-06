require 'image'
--Modules based on http://arxiv.org/pdf/1602.02660.pdf
--All rotations are clockwise
--------------------------------------------------------------

local CycSlice, Parent = torch.class('nn.CycSlice', 'nn.Module')
--Perform Cyclical Slicing with 4 slices 

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
		self.output[{3*batch + i,{},{},{}}] = torch.bmm(rot,input[{i,{},{},{}}]:transpose(2,3)) --perform 270 rotation
	end 
	return self.output
end

function CycSlice:updateGradInput(input,gradOutput)
	local rot = torch.eye(gradOutput:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(gradOutput:size(2),1,1)
	local batch = input:size(1)
	self.gradInput = torch.zeros(input:size())
	for i = 1, batch do
		self.gradInput[{i,{},{},{}}] = 
				gradOutput[{i,{},{},{}}]
				+ torch.bmm(rot,gradOutput[{batch + i,{},{},{}}]:transpose(2,3)) --perform 270 rotation
				+ torch.bmm(rot,torch.bmm(gradOutput[{2*batch + i,{},{},{}}],rot)) -- perform 180 rotation
				+ torch.bmm(gradOutput[{3*batch + i,{},{},{}}]:transpose(2,3),rot) -- perform 90 rotation
	end
	return self.gradInput
end
----------------------------------------------------------------

local MeanCycPool = torch.class('nn.MeanCycPool', 'nn.Module')
--Perform Mean Pooling for 4 slices with rotation back to original orientation

function MeanCycPool:updateOutput(input)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local batch = input:size(1)/4
	self.output = torch.zeros(batch,input:size(2),input:size(3),input:size(4))
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

----------------------------------------------------------------

local MeanCycPoolB = torch.class('nn.MeanCycPool2', 'nn.Module')
--Perform Mean Pooling on 4 slices without reorientation 

function MeanCycPoolB:updateOutput(input)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local batch = input:size(1)/4
	self.output = torch.zeros(batch,input:size(2),input:size(3),input:size(4))
	for i = 1,batch do
		self.output[{i,{},{},{}}] = (input[{i,{},{},{}}]	
				+ input[{batch + i,{},{},{}}] 
				+ input[{batch*2 + i,{},{},{}}] --perform 180 rotation
				+ input[{batch*3 + i,{},{},{}}])/4 --perform 90 rotation
	end
	--self.output:copy(output)
	return self.output
end

function MeanCycPoolB:updateGradInput(input,gradOutput)
	local rot = torch.eye(gradOutput:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(gradOutput:size(2),1,1)	
	--self.gradInput:copy(input)
	local batch = gradOutput:size(1)
	self.gradInput = torch.zeros(input:size())
	for i = 1,batch do 
		self.gradInput[{i,{},{},{}}] = gradOutput[{i,{},{},{}}]/4
		self.gradInput[{batch + i,{},{},{}}] = gradOutput[{i,{},{},{}}]/4 --perform 90 rotation
		self.gradInput[{2*batch + i,{},{},{}}] = gradOutput[{i,{},{},{}}]/4 -- perform 180 rotation
		self.gradInput[{3*batch + i,{},{},{}}] = gradOutput[{i,{},{},{}}]/4 --perform 270 rotation
	end
	return self.gradInput
end

---------------------------------------------------------------

local CycRoll, Parent = torch.class('nn.CycRoll', 'nn.Module')
--Increases filter size by 4 for each rotated image by rotating each images'
--rotated counterparts into its orientation and concatenating as filters.
--Built for slices of 4

function CycRoll:updateOutput(input)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local batch=input:size(1)/4
	self.output = torch.Tensor(4*batch,4*input:size(2),input:size(3),input:size(4))

	for i = 1,batch do
		self.output[{i,{},{},{}}] = torch.cat({input[{i,{},{},{}}],  
				torch.bmm(rot,input[{batch + i,{},{},{}}]:transpose(2,3)),
				torch.bmm(rot,torch.bmm(input[{batch*2 + i,{},{},{}}],rot)),
				torch.bmm(input[{batch*3 + i,{},{},{}}]:transpose(2,3),rot)},1) --0
		self.output[{batch + i,{},{},{}}] = torch.cat({input[{batch + i,{},{},{}}], 
				torch.bmm(rot,input[{batch*2+i,{},{},{}}]:transpose(2,3)),				
				torch.bmm(rot,torch.bmm(input[{batch*3 + i,{},{},{}}],rot)),
				torch.bmm(input[{i,{},{},{}}]:transpose(2,3),rot)},1) --90
		self.output[{2*batch + i,{},{},{}}] = torch.cat({input[{2*batch + i,{},{},{}}], 
				torch.bmm(rot,input[{batch*3+i,{},{},{}}]:transpose(2,3)),				
				torch.bmm(rot,torch.bmm(input[{i,{},{},{}}],rot)),
				torch.bmm(input[{batch+i,{},{},{}}]:transpose(2,3),rot)},1) --180
		self.output[{3*batch + i,{},{},{}}] = torch.cat({input[{3*batch + i,{},{},{}}], 
				torch.bmm(rot,input[{i,{},{},{}}]:transpose(2,3)),				
				torch.bmm(rot,torch.bmm(input[{batch + i,{},{},{}}],rot)),
				torch.bmm(input[{2*batch + i,{},{},{}}]:transpose(2,3),rot)},1) --270
	end
	return self.output
end

function CycRoll:updateGradInput(input,gradOutput)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local batch = input:size(1)/4
	local filters = input:size(2)
	self.gradInput = torch.zeros(input:size())
	for i = 1,batch do 
		self.gradInput[{i,{},{},{}}] = (gradOutput[{i,{1,filters},{},{}}]
						+ torch.bmm(rot,gradOutput[{batch+i,{3*filters+1,4*filters},{},{}}]:transpose(2,3))
						+ torch.bmm(rot,torch.bmm(gradOutput[{2*batch+i,{2*filters+1,3*filters},{},{}}],rot))
						+	torch.bmm(gradOutput[{3*batch+i,{filters+1,2*filters},{},{}}]:transpose(2,3),rot))
		self.gradInput[{batch + i,{},{},{}}] = (gradOutput[{batch+i,{1,filters},{},{}}]
						+ torch.bmm(rot,gradOutput[{2*batch+i,{3*filters+1,4*filters},{},{}}]:transpose(2,3))
						+ torch.bmm(rot,torch.bmm(gradOutput[{3*batch+i,{2*filters+1,3*filters},{},{}}],rot))
						+	torch.bmm(gradOutput[{i,{filters+1,2*filters},{},{}}]:transpose(2,3),rot))
		self.gradInput[{2*batch + i,{},{},{}}] = (gradOutput[{2*batch+i,{1,filters},{},{}}]
						+ torch.bmm(rot,gradOutput[{3*batch+i,{3*filters+1,4*filters},{},{}}]:transpose(2,3))
						+ torch.bmm(rot,torch.bmm(gradOutput[{i,{2*filters+1,3*filters},{},{}}],rot))
						+	torch.bmm(gradOutput[{batch+i,{filters+1,2*filters},{},{}}]:transpose(2,3),rot))
		self.gradInput[{3*batch + i,{},{},{}}] = (gradOutput[{3*batch+i,{1,filters},{},{}}]
						+ torch.bmm(rot,gradOutput[{i,{3*filters+1,4*filters},{},{}}]:transpose(2,3))
						+ torch.bmm(rot,torch.bmm(gradOutput[{batch + i,{2*filters+1,3*filters},{},{}}],rot))
						+	torch.bmm(gradOutput[{2*batch+i,{filters+1,2*filters},{},{}}]:transpose(2,3),rot))
	end
	return self.gradInput
end



----------------------------------------------------------------

local CycSlice8, Parent = torch.class('nn.CycSlice8', 'nn.Module')
--Performs Cyclical Slicing with 8 slices.

function CycSlice8:updateOutput(input)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local batch=input:size(1)
	self.output = torch.Tensor(batch*8,input:size(2),input:size(3),input:size(4))

	for i = 1,batch do
		self.output[{i,{},{},{}}] = input[{i,{},{},{}}]
		self.output[{batch + i,{},{},{}}] = torch.bmm(input[{i,{},{},{}}]:transpose(2,3),rot) --perform 90 rotation
		self.output[{2*batch + i,{},{},{}}] = torch.bmm(rot,torch.bmm(input[{i,{},{},{}}],rot)) --perform 180 rotation
		self.output[{3*batch + i,{},{},{}}] = torch.bmm(rot,input[{i,{},{},{}}]:transpose(2,3)) -- perform 270 rotation

		self.output[{4*batch + i,{},{},{}}] = image.rotate(input[{i,{},{},{}}],-math.pi/4,'bilinear') --perform 45 rotation
		self.output[{5*batch + i,{},{},{}}] = torch.bmm(self.output[{4*batch + i,{},{},{}}]:transpose(2,3),rot) --perform 90 rotation on 45
		self.output[{6*batch + i,{},{},{}}] = torch.bmm(rot,torch.bmm(self.output[{4*batch + i,{},{},{}}],rot)) --perform 180 rotation on 45
		self.output[{7*batch + i,{},{},{}}] = torch.bmm(rot,self.output[{4*batch + i,{},{},{}}]:transpose(2,3)) -- perform 270 rotation on 45
	end
	return self.output
end

function CycSlice8:updateGradInput(input,gradOutput)
	local rot = torch.eye(gradOutput:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(gradOutput:size(2),1,1)
	local batch = input:size(1)
	self.gradInput = torch.zeros(input:size())
	for i = 1, batch do
		self.gradInput[{i,{},{},{}}] = 
				gradOutput[{i,{},{},{}}]
				+ torch.bmm(rot,(gradOutput[{batch + i,{},{},{}}] + image.rotate(gradOutput[{5*batch + i,{},{},{}}],math.pi/4,'bilinear')):transpose(2,3)) --perform 270 rotation
				+ torch.bmm(rot,torch.bmm(gradOutput[{2*batch + i,{},{},{}}] + image.rotate(gradOutput[{6*batch + i,{},{},{}}],math.pi/4,'bilinear'),rot)) -- perform 180 rotation
				+ torch.bmm((gradOutput[{3*batch + i,{},{},{}}] + image.rotate(gradOutput[{7*batch + i,{},{},{}}],math.pi/4,'bilinear')):transpose(2,3),rot) -- perform 90 rotation
				+ image.rotate(gradOutput[{4*batch + i,{},{},{}}],math.pi/4,'bilinear') --perform -45 degree rotation

	end
	return self.gradInput
end
----------------------------------------------------------------

local MeanCycPool8 = torch.class('nn.MeanCycPool8', 'nn.Module')
--Performs Mean Pooling on 8 slices

function MeanCycPool8:updateOutput(input)
	local rot = torch.eye(input:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(input:size(2),1,1)
	local batch = input:size(1)/8
	self.output = torch.zeros(batch,input:size(2),input:size(3),input:size(4))
	for i = 1,batch do
		self.output[{i,{},{},{}}] = ( input[{i,{},{},{}}] + image.rotate(input[{batch*4 + i,{},{},{}}],math.pi/4,'bilinear')
				+ torch.bmm(rot,(input[{batch + i,{},{},{}}]+image.rotate(input[{batch*5 + i,{},{},{}}],math.pi/4,'bilinear')):transpose(2,3)) --perform 270 rotation
				+ torch.bmm(rot,torch.bmm(input[{batch*2 + i,{},{},{}}]+image.rotate(input[{batch*6 + i,{},{},{}}],math.pi/4,'bilinear'),rot)) --perform 180 rotation
				+ torch.bmm((input[{batch*3 + i,{},{},{}}]+image.rotate(input[{batch*7 + i,{},{},{}}],math.pi/4,'bilinear')):transpose(2,3),rot))/8 --perform 90 rotation
	end
	--self.output:copy(output)
	return self.output
end

function MeanCycPool8:updateGradInput(input,gradOutput)
	local rot = torch.eye(gradOutput:size(4))
	rot = image.hflip(rot)
	rot = rot:repeatTensor(gradOutput:size(2),1,1)	
	--self.gradInput:copy(input)
	local batch = gradOutput:size(1)
	self.gradInput = torch.zeros(input:size())
	for i = 1,batch do 
		self.gradInput[{i,{},{},{}}] = gradOutput[{i,{},{},{}}]/8
		self.gradInput[{batch + i,{},{},{}}] = torch.bmm(gradOutput[{i,{},{},{}}]:transpose(2,3),rot)/8 --perform 90 rotation
		self.gradInput[{2*batch + i,{},{},{}}] = torch.bmm(rot,torch.bmm(gradOutput[{i,{},{},{}}],rot))/8 -- perform 180 rotation
		self.gradInput[{3*batch + i,{},{},{}}] = torch.bmm(rot,gradOutput[{i,{},{},{}}]:transpose(2,3))/8 --perform 270 rotation

		self.gradInput[{4*batch + i,{},{},{}}] = image.rotate(gradOutput[{i,{},{},{}}],-math.pi/4,'bilinear')/8 -- perform 45 degree rotation
		self.gradInput[{5*batch + i,{},{},{}}] = torch.bmm(self.gradInput[{4*batch + i,{},{},{}}]:transpose(2,3),rot)/8 --perform 90 rotation
		self.gradInput[{6*batch + i,{},{},{}}] = torch.bmm(rot,torch.bmm(self.gradInput[{4*batch + i,{},{},{}}],rot))/8 -- perform 180 rotation
		self.gradInput[{7*batch + i,{},{},{}}] = torch.bmm(rot,self.gradInput[{4*batch + i,{},{},{}}]:transpose(2,3))/8 --perform 270 rotation
	end
	return self.gradInput
end




