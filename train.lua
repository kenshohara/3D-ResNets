train_logger = optim.Logger(paths.concat(opt.result_path, 'train.log'))
train_batch_logger = optim.Logger(paths.concat(opt.result_path, 'train_batch.log'))
local batch_number
local epoch_size
local loss
local acc

function train_epoch()
  model:training()
  print('==> training epoch # ' .. epoch)

  local n_samples = #training_data
  local shuffled_indices = torch.randperm(n_samples)

  if not opt.no_cuda then
    cutorch.synchronize()
  end

  batch_number = 0
  loss = 0
  acc = 0
  epoch_size = math.floor(n_samples / opt.batch_size)
  local timer = torch.Timer()
  for i = 1, n_samples - opt.batch_size, opt.batch_size do
    task_queue:addjob(
        function()
          collectgarbage()

          local inputs = torch.Tensor(opt.batch_size, 3, opt.sample_duration,
                                      opt.sample_size, opt.sample_size)
          local targets = torch.Tensor(opt.batch_size)

          for j = i, (i + opt.batch_size - 1) do
            local index = shuffled_indices[j]
            local video_directory_path = training_data[index].video
            local begin_t = training_data[index].segment[1]
            local end_t = training_data[index].segment[2]

            local sample = data_loader.load_random_sample(
                video_directory_path, begin_t, end_t,
                opt.sample_size, opt.sample_duration, opt.scales, opt.no_hflip, opt.crop)
            local target = training_data[index].label

            inputs[j - i + 1] = sample
            targets[j - i + 1] = target
          end

          collectgarbage()

          return inputs, targets
        end,
        train_batch
    )
  end

  task_queue:synchronize()
  if not opt.no_cuda then
    cutorch.synchronize()
  end

  loss = loss / epoch_size
  acc = acc / n_samples

  train_logger:add{
    ['epoch'] = epoch,
    ['loss']  = loss,
    ['acc'] = acc,
    ['lr'] = optim_state.learningRate
  }
  print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
            .. 'Loss: %.2f\t'
            .. 'Acc: %.3f\t\n',
        epoch, timer:time().real, loss, acc))

  collectgarbage()
  model:clearState()

  if epoch % opt.checkpoint == 0 then
    torch.save(paths.concat(opt.result_path, 'model_' .. epoch .. '.t7'),
               utils.extract_model_from_data_parallel_table(model))
  end
end

local inputs
local targets
if not opt.no_cuda then
  inputs = torch.CudaTensor()
  targets = torch.CudaTensor()
end

local batch_timer = torch.Timer()
local data_timer = torch.Timer()

local parameters, grad_parameters = model:getParameters()

function train_batch(inputs_cpu, targets_cpu)
  if not opt.no_cuda then
    cutorch.synchronize()
  end
  collectgarbage()

  local data_loading_time = data_timer:time().real

  if not opt.no_cuda then
    inputs:resize(inputs_cpu:size()):copy(inputs_cpu)
    targets:resize(targets_cpu:size()):copy(targets_cpu)
  else
    inputs = inputs_cpu
    targets = targets_cpu
  end

  local loss_batch
  local outputs
  local feval = function(x)
    model:zeroGradParameters()
    outputs = model:forward(inputs)
    loss_batch = criterion:forward(outputs, targets)
    local grad_outputs = criterion:backward(outputs, targets)
    model:backward(inputs, grad_outputs)
    return loss_batch, grad_parameters
  end
  optimizer(feval, parameters, optim_state)

  if not opt.no_cuda then
    cutorch.synchronize()
  end
  batch_number = batch_number + 1
  loss = loss + loss_batch
  local acc_batch = 0
  do
    outputs = outputs:view(inputs_cpu:size(1), -1)
    local _, scores_sorted = outputs:float():sort(2, true)
    for i = 1, scores_sorted:size(1) do
       if scores_sorted[i][1] == targets_cpu[i] then
        acc = acc + 1
        acc_batch = acc_batch + 1
      end
    end
    acc_batch = acc_batch / inputs_cpu:size(1)
  end

  train_batch_logger:add{
    ['epoch'] = epoch,
    ['batch'] = batch_number,
    ['iter'] = (epoch - 1) * epoch_size + batch_number,
    ['loss']  = loss_batch,
    ['acc'] = acc_batch,
    ['lr'] = optim_state.learningRate
  }

  print(('Epoch: Training [%d][%d/%d]\tTime %.3f Loss %.4f Acc %.3f DataLoadingTime %.2f'):format(
      epoch, batch_number, epoch_size, batch_timer:time().real, loss_batch, acc_batch,
      data_loading_time))
  data_timer:reset()
end
