function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end


local filePath = 'mnist_train.amat'

-- Count number of rows and columns in file
local i = 0
for line in io.lines(filePath) do
  if i == 0 then
    COLS = #line:split(' ')
  end
  i = i + 1
end

local ROWS = i  -- Minus 1 because of header

-- Read data from CSV to tensor
local csvFile = io.open(filePath, 'r')
--local header = csvFile:read()

data = torch.Tensor(ROWS, COLS)

i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  l = line:split(' ')
  for key, val in ipairs(l) do
    data[i][key] = val
  end
end

csvFile:close()

labels = data[{{},{-1}}]:clone()
data2 = data[{{},{1,-2}}]:clone()

-- Serialize tensor
local outputFilePath = 'mnist_train_data.th7'
torch.save(outputFilePath, data2)
local outputFilePath2 = 'mnist_train_labels.th7'
torch.save(outputFilePath2, labels)
--[[
-- Deserialize tensor object
local restored_data = torch.load(outputFilePath)

-- Make test
print(data:size())
print(restored_data:size())--]]
