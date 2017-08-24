val_logger = optim.Logger(paths.concat(opt.result_path, 'val.log'))

local batch_number
local loss
local acc
local epoch_size
local timer = torch.Timer()

local function prepare_val_batch()
  local sample_validation_data = {}
  for i = 1, #validation_data do
    local video_directory_path = validation_data[i].video
    local begin_t = validation_data[i].segment[1]
    local end_t = validation_data[i].segment[2]

    local duration = end_t - begin_t + 1
    local step = math.max(
        opt.sample_duration, math.ceil(duration / opt.n_val_samples))
    for t = begin_t, end_t - opt.sample_duration, step do
      local sample_begin_t = t
      local sample_end_t = t + opt.sample_duration - 1
      local sample_data = utils.clone_table(validation_data[i])
      sample_data.segment = {sample_begin_t, sample_end_t}
      table.insert(sample_validation_data, sample_data)
    end
  end

  return sample_validation_data
end

function val_epoch()
  batch_number = 0
  if not opt.no_cuda then
    cutorch.synchronize()
  end
  timer:reset()

  model:evaluate()
  print('==> validation epoch # ' .. epoch)
  local sample_validation_data = prepare_val_batch()
  local n_samples = #sample_validation_data
  epoch_size = math.ceil(n_samples / opt.batch_size)

  loss = 0
  acc = 0
  for i = 1, n_samples, opt.batch_size do
    local size = math.min(opt.batch_size, n_samples - i + 1)
    task_queue:addjob(
        function()
          collectgarbage()
          local inputs = torch.Tensor(size, 3, opt.sample_duration,
                                      opt.sample_size, opt.sample_size)
          local targets = torch.Tensor(size)

          for j = i, (i + size - 1) do
            local video_directory_path = sample_validation_data[j].video
            local begin_t = sample_validation_data[j].segment[1]
            local end_t = sample_validation_data[j].segment[2]

            local sample = data_loader.load_center_sample(
                video_directory_path, opt.sample_size, begin_t, end_t)
            local target = sample_validation_data[j].label

            inputs[j - i + 1] = sample
            targets[j - i + 1] = target
          end

          collectgarbage()

          return inputs, targets
        end,
        val_batch
    )
  end

  task_queue:synchronize()
  if not opt.no_cuda then
    cutorch.synchronize()
  end

  loss = loss / epoch_size
  acc = acc / n_samples

  val_logger:add{
    ['epoch'] = epoch,
    ['loss'] = loss,
    ['acc'] = acc,
    ['lr'] = optim_state.learningRate
  }
  print(string.format('Epoch: [%d][VALIDATION SUMMARY] Total Time(s): %.2f \t Loss: %.2f \t Acc: %.3f\n',
    epoch, timer:time().real, loss, acc))
end

local inputs
local targets
if not opt.no_cuda then
  inputs = torch.CudaTensor()
  targets = torch.CudaTensor()
end

function val_batch(inputs_cpu, targets_cpu)
  if not opt.no_cuda then
    inputs:resize(inputs_cpu:size()):copy(inputs_cpu)
    targets:resize(targets_cpu:size()):copy(targets_cpu)
  else
    inputs = inputs_cpu
    targets = targets_cpu
  end

  local outputs = model:forward(inputs)
  if outputs:dim() == 1 then
    outputs = outputs:reshape(inputs:size(1), outputs:size(1) / inputs:size(1))
  end
  local loss_batch = criterion:forward(outputs, targets)
  if not opt.no_cuda then
    cutorch.synchronize()
  end
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

  batch_number = batch_number + 1
  print(string.format('Epoch: Validation [%d][%d/%d] \t Loss %.4f \t Acc %.3f', epoch, batch_number, epoch_size, loss_batch, acc_batch))
  collectgarbage()
end
