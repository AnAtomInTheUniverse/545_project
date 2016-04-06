#!/usr/bin/lua

local con_data = torch.load('../data/mnist_con_train_data.th7')
local con_labels = torch.load('../data/mnist_con_train_labels.th7')
local rot_data = torch.load('../data/mnist_rot_train_data.th7')
local rot_labels = torch.load('../data/mnist_rot_train_labels.th7')

local ROWS, COLS = con_data:size(1)*2, con_data:size(2)
combined_data = torch.Tensor(ROWS, COLS)
combined_labels = torch.Tensor(ROWS,1)	

local j = 1
for i=1,24000 do
	if i%2 ~= 0 then
		combined_data[{{i},{}}] = con_data[{{j},{}}]:clone()
		combined_labels[{{i},{}}] = con_labels[{{j},{}}]:clone()
 	else
		combined_data[{{i},{}}] = rot_data[{{j},{}}]:clone()
		combined_labels[{{i},{}}] = rot_labels[{{j},{}}]:clone()
		j = j+1
  end
end
--combined_labels = torch.cat(con_labels, rot_labels, 1)
--combined_data = torch.cat(con_data, rot_data, 1)

-- Serialize tensor
local outputFilePath = '../data/comb_train_data.th7'
torch.save(outputFilePath, combined_data)
local outputFilePath2 = '../data/comb_train_labels.th7'
torch.save(outputFilePath2, combined_labels)
