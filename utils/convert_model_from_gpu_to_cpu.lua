require('torch')
require('cutorch')
require('nn')
require('cudnn')

model = torch.load(arg[1])
cpu_model = cudnn.convert(model, nn)
cpu_model:float()
torch.save(arg[2], cpu_model)
