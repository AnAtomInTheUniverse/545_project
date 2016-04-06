--Calculate output of test data from already trained model
require 'nn'
require 'math'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-m', 'Model file name')
cmd:option('-t', 'Test data file name')
cmd:option('-o', 'Output file name')

args = cmd:parse(arg)

fwrite = args.o

batchSize = 200

model = torch.load(args.m)
data_temp = torch.load(args.t);
data = data_temp:clone()
imdim = math.sqrt(data:size(2))
data:resize(data:size(1),1,imdim,imdim);
mean = data:mean()
std = data:std()
data:add(-mean);
data:mul(1/std);

num = data:size(1)


for i = 1,num/batchSize do
	max,temp = model:forward(data[{{(i-1)*batchSize+1,i*batchSize},{}}]):max(2)
	
	if i == 1 then preds = temp - 1 
	else preds = torch.cat(preds,temp - 1,1) end
end

torch.save(fwrite, preds)
--[[
csv = io.open(write,'w')
csv:write("id,category\n")
for i = 1,preds:size(1) do
	csv:write(string.format("%i,%i\n",i,preds[i][1]))
end
csv:close()
]]--

