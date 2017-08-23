require('torch')
require('cutorch')
require('nn')
require('image')
require('optim')
require('json')

opts = dofile('opts.lua')
opt = opts.parse(arg)
print(opt)
json.save(paths.concat(opt.result_path, 'opts.json'), opt)

torch.manualSeed(opt.manual_seed)

if not opt.no_cuda then
  require('cunn')
  require('cudnn')
  cudnn.fastest = true
  cudnn.benchmark = true
  cudnn.verbose = false

  cutorch.setDevice(opt.gpu_id)
  cutorch.manualSeed(opt.manual_seed)
end

utils = dofile('utils.lua')
dofile('mean.lua')
dofile('model.lua')
dofile('dataset.lua')
dofile('data_threads.lua')

if not opt.no_train then
  optimizer = optim['sgd']
  optim_state = {
    learningRate = opt.learning_rate,
    weightDecay = opt.weight_decay,
    momentum = opt.momentum,
    learningRateDecay = 0,
  }
  if opt.freeze_params ~= 0 then
    lrs, lrs_model = utils.get_learning_rates_for_freezing_layers(model, opt.freeze_params)
    optim_state.learningRates = lrs
  end

  dofile('train.lua')
end
if not opt.no_test then
  dofile('test.lua')
end

print('run')
for i = opt.begin_epoch, opt.n_epochs do
  epoch = i

  if not opt.no_train then
    if opt.regimes ~= None then
      for _, row in ipairs(opt.regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            optim_state.learningRate = row[3]
            optim_state.weightDecay = row[4]
        end
      end
    end
    train_epoch()
  end
  if not opt.no_test then
    test_epoch()
  end
end

if opt.test_video then
  print('test video')
  dofile('test_video.lua')
  test_video()
end
