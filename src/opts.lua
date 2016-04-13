local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
		cmd:text()
    cmd:option('-model',  'base', 'Network model')
		cmd:option('-data', 'can', 'Training dataset')
		cmd:option('-batchSize', 100, 'Batch Size')
		cmd:option('-nEpochs', 60, 'Number of epochs')
		cmd:option('-numTrain', 10000, 'Number of training samples')
		cmd:option('-numVal', 2000, 'Number of validation samples')
		cmd:option('-coefL1', 0, 'L1 normalization coefficient')
		cmd:option('-coefL2', 0, 'L2 normalization coefficient')
		cmd:option('-LR', 0.03, 'SGD learning rate')
		cmd:option('-momentum', 0, 'SGD momentum')
		cmd:option('-LRdecay', 0, 'SGD learning rate decay')
		cmd:option('-method','sgd','Method for optimization (nag or sgd)')
		local opt = cmd:parse(arg or {})
		opt.save = '../models/' .. opt.model .. '_' .. opt.data .. '.model'
		opt.nIters = opt.numTrain/opt.batchSize
		return opt
end

return M
