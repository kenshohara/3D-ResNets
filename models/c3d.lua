function create_model()
  local features
  local classifier
  if paths.filep(opt.premodel_path) then
    print('load pretrained model')

    features = torch.load(opt.premodel_path)

    if opt.bgr_premodel then
      local first_layer = features:get(1)
      local w = first_layer.weight:clone()
      first_layer.weight[{{}, 1, {}, {}, {}}]:copy(w[{{}, 3, {}, {}, {}}])
      first_layer.weight[{{}, 3, {}, {}, {}}]:copy(w[{{}, 1, {}, {}, {}}])
    end

    for i = 1, #features do
      local weight = features:get(i).weight
      local gradWeight = features:get(i).gradWeight
      if weight and gradWeight and gradWeight:size():size() == 0 then -- if gradWeight is empty
        features:get(i).gradWeight = weight:clone()
        features:get(i).gradBias = features:get(i).bias:clone()
      end
    end

    utils.remove_softmax_and_last_fc_layers(features)

    if not opt.no_cuda then
      features = utils.make_data_parallel(features, opt.gpu_id, opt.n_gpus)
    end

    classifier = nn.Sequential():add(nn.Linear(4096, opt.n_classes))
  else
    print('make new model')

    features = nn.Sequential()
    classifier = nn.Sequential()
    local layers = {64, 'p1', 128, 'p', 256, 256, 'p', 512, 512, 'p', 512, 512}
    do
      local inputChannels = 3
      for _, value in pairs(layers) do
        if value == 'p1' then
          features:add(nn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2):ceil())
        elseif value == 'p' then
          features:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2):ceil())
        else
          local outputChannels = value
          features:add(nn.VolumetricConvolution(inputChannels, outputChannels,
                                           3, 3, 3, 1, 1, 1, 1, 1, 1))
          if opt.batch_norm then
            features:add(nn.VolumetricBatchNormalization(outputChannels))
          end
          features:add(nn.ReLU(true))
          inputChannels = outputChannels
        end
      end
    end
    features:get(1).gradInput = nil

    if not opt.no_cuda then
      features = utils.make_data_parallel(features, opt.gpu_id, opt.n_gpus)
    end

    if not opt.global_pooling then
      classifier:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2):ceil())
      classifier:add(nn.View(512 * 1 * 4 * 4))
      classifier:add(nn.Linear(512 * 1 * 4 * 4, 4096))
      if opt.batch_norm then
        classifier:add(nn.BatchNormalization(4096))
      end
      classifier:add(nn.ReLU(true))
      classifier:add(nn.Dropout(opt.dropout_ratio))
      classifier:add(nn.Linear(4096, 4096))
      if opt.batch_norm then
        classifier:add(nn.BatchNormalization(4096))
      end
      classifier:add(nn.ReLU(true))
      classifier:add(nn.Dropout(opt.dropout_ratio))
      classifier:add(nn.Linear(4096, opt.n_classes))
    else
      classifier:add(nn.VolumetricAveragePooling(2, 7, 7, 1, 1, 1))
      classifier:add(nn.View(512))
      classifier:add(nn.Linear(512, opt.n_classes))
    end
  end
  classifier:add(nn.LogSoftMax())
  return nn.Sequential():add(features):add(classifier)
end
