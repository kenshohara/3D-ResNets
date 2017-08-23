if opt.begin_epoch == 1 then
  if opt.model == 'c3d' then
    dofile('c3d.lua')
  elseif opt.model == 'resnet' then
    dofile('resnet.lua')
  end
  model = create_model()
  if not opt.no_cuda then
    model = model:cuda()
    cudnn.convert(model, cudnn)
  end
else
  local model_file_path = paths.concat(
      opt.result_path, 'model_' .. (opt.begin_epoch - 1) .. '.t7')
  assert(paths.filep(model_file_path),
         'pretrained model at epoch ' .. (opt.begin_epoch - 1) .. ' does not exist')
  print('pretrained model at epoch ' .. (opt.begin_epoch - 1) .. ' is loaded')
  model = torch.load(model_file_path)
  if not opt.no_cuda then
    model = utils.make_data_parallel(model, opt.gpu_id, opt.n_gpus)
  end
end
print(model)

criterion = nn.ClassNLLCriterion()
if not opt.no_cuda then
  criterion = criterion:cuda()
end
