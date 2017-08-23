local Convolution = nn.VolumetricConvolution
local Avg = nn.VolumetricAveragePooling
local ReLU = nn.ReLU
local Max = nn.VolumetricMaxPooling
local VBatchNorm = nn.VolumetricBatchNormalization

function create_model()
  local depth = opt.resnet_depth
  local shortcutType = opt.shortcut_type or 'B'
  local iChannels
  local model

  -- The shortcut layer is either identity or 1x1 convolution
  local function shortcut(nInputPlane, nOutputPlane, stride)
    local useConv = shortcutType == 'C' or
        (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
    if useConv then
      -- 1x1 convolution
      return nn.Sequential()
          :add(Convolution(nInputPlane, nOutputPlane, 1, 1, 1, stride, stride, stride))
          :add(VBatchNorm(nOutputPlane))
    elseif nInputPlane ~= nOutputPlane then
      -- Strided, zero-padded identity shortcut
      return nn.Sequential()
          :add(Avg(1, 1, 1, stride, stride, stride))
          :add(nn.Concat(2)
              :add(nn.Identity())
              :add(nn.MulConstant(0)))
    else
      return nn.Identity()
    end
  end

  -- The basic residual layer block for 18 and 34 layer network, and the
  -- CIFAR networks
  local function basicblock(n, stride)
    local nInputPlane = iChannels
    iChannels = n

    local s = nn.Sequential()
    s:add(Convolution(nInputPlane, n, 3, 3, 3, stride, stride, stride, 1, 1, 1))
    s:add(VBatchNorm(n))
    s:add(ReLU(true))
    s:add(Convolution(n, n, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    s:add(VBatchNorm(n))

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
  end

  -- The bottleneck residual layer for 50, 101, and 152 layer networks
  local function bottleneck(n, stride)
    local nInputPlane = iChannels
    iChannels = n * 4

    local s = nn.Sequential()
    s:add(Convolution(nInputPlane, n, 1, 1, 1, 1, 1, 1, 0, 0, 0))
    s:add(VBatchNorm(n))
    s:add(ReLU(true))
    s:add(Convolution(n, n, 3, 3, 3, stride, stride, stride, 1, 1, 1))
    s:add(VBatchNorm(n))
    s:add(ReLU(true))
    s:add(Convolution(n, n * 4 , 1, 1, 1, 1, 1, 1, 0, 0, 0))
    s:add(VBatchNorm(n * 4))

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
  end

  -- Creates count residual blocks with specified number of features
  local function layer(block, features, count, stride)
    local s = nn.Sequential()
    for i = 1, count do
      s:add(block(features, i == 1 and stride or 1))
    end
    return s
  end

  -- Configurations for ResNet:
  --  num. residual blocks, num features, residual block function
  local cfg = {
    [10]  = {{1, 1, 1, 1}, 512, basicblock},
    [18]  = {{2, 2, 2, 2}, 512, basicblock},
    [34]  = {{3, 4, 6, 3}, 512, basicblock},
    [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
    [101] = {{3, 4, 23, 3}, 2048, bottleneck},
    [152] = {{3, 8, 36, 3}, 2048, bottleneck},
  }

  assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
  local def, nFeatures, block = table.unpack(cfg[depth])
  iChannels = 64

  if paths.filep(opt.premodel_path) then
    print('load pretrained model')

    model = torch.load(opt.premodel_path)
    if #model == 1 then -- remove outer nn.sequential
      model = model:get(1)
    end

    if opt.bgr_premodel then
      local first_layer = model:get(1)
      local w = first_layer.weight:clone()
      first_layer.weight[{{}, 1, {}, {}, {}}]:copy(w[{{}, 3, {}, {}, {}}])
      first_layer.weight[{{}, 3, {}, {}, {}}]:copy(w[{{}, 1, {}, {}, {}}])
    end
    for i = 1, #model do
      local weight = model:get(i).weight
      local gradWeight = model:get(i).gradWeight
      if weight and gradWeight and gradWeight:size():size() == 0 then -- if gradWeight is empty
        model:get(i).gradWeight = weight:clone()
        model:get(i).gradBias = model:get(i).bias:clone()
      end
    end

    utils.remove_softmax_and_last_fc_layers(model)

    model:add(nn.Linear(nFeatures, opt.n_classes))
    model:add(nn.LogSoftMax())
  else
    model = nn.Sequential()

    model:add(Convolution(3, 64, 7, 7, 7, 1, 2, 2, 3, 3, 3))
    model:add(VBatchNorm(64))
    model:add(ReLU(true))
    model:add(Max(3, 3, 3, 2, 2, 2, 1, 1, 1))
    model:add(layer(block, 64, def[1]))
    model:add(layer(block, 128, def[2], 2))
    model:add(layer(block, 256, def[3], 2))
    model:add(layer(block, 512, def[4], 2))
    model:add(Avg(1, 4, 4, 1, 1, 1))
    model:add(nn.View(nFeatures):setNumInputDims(4))
    model:add(nn.Linear(nFeatures, opt.n_classes))
    model:add(nn.LogSoftMax())

    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
        local n = v.kW*v.kH*v.nOutputPlane
        v.weight:normal(0,math.sqrt(2/n))
        if cudnn.version >= 4000 then
          v.bias = nil
          v.gradBias = nil
        else
          v.bias:zero()
        end
      end
    end
    local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
        v.weight:fill(1)
        v.bias:zero()
      end
    end

    if opt.weight_init then
      print('init weight')
      ConvInit('nn.VolumetricConvolution')
      BNInit('nn.VolumetricBatchNormalization')
    end
    for k, v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
    end
    model:type(opt.tensorType)

    if opt.cudnn == 'deterministic' then
      model:apply(function(m)
          if m.setMode then m:setMode(1,1,1) end
          end)
    end

    model:get(1).gradInput = nil
  end

  if not opt.no_cuda then
    dpt_net = nn.Sequential()
    dpt_net:add(utils.make_data_parallel(model, opt.gpu_id, opt.n_gpus))
    return dpt_net
  else
    return model
  end
end
