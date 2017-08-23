local utils = {}

function utils.make_data_parallel(model, first_gpu_id, n_gpus)
  if n_gpus < 2 then
    return model
  end

  assert(n_gpus <= cutorch.getDeviceCount(), 'number of GPUs less than n_gpus specified')
  local gpu_table = torch.range(first_gpu_id, first_gpu_id + n_gpus - 1):totable()
	local fastest, benchmark = cudnn.fastest, cudnn.benchmark
  local dpt = nn.DataParallelTable(1, true):add(model, gpu_table):threads(
    function()
      require 'cudnn'
      cudnn.fastest = fastest
      cudnn.benchmark = benchmark
    end)
  dpt.gradInput = nil
  model = dpt:cuda()

  return model
end

function utils.extract_model_from_data_parallel_table(model)
  if torch.type(model) == 'nn.Sequential' then
    local tmp_model = nn.Sequential()
    for i = 1, #model do
      tmp_model:add(utils.extract_model_from_data_parallel_table(model:get(i)))
    end
    return tmp_model
  elseif torch.type(model) == 'nn.DataParallelTable' then
    return model:get(1)
  else
    return model
  end
end

function utils.remove_softmax_and_last_fc_layers(model)
  if torch.type(model) == 'nn.Sequential' then
    local last_type_name = torch.type(model:get(#model))
    if last_type_name == 'nn.Sequential' then
      utils.remove_softmax_and_last_fc_layers(model:get(#model))
      if #model:get(#model) == 0 then
        model:remove(#model)
      end
    elseif last_type_name == 'nn.LogSoftMax' or last_type_name == 'cudnn.LogSoftMax'
        or last_type_name == 'nn.SoftMax' or last_type_name == 'cudnn.SoftMax' then
      model:remove(#model)
      model:remove(#model)
    end
  end
end

function utils.get_learning_rates_for_freezing_layers(model, n_unfreezed_layers)
  local lrs_model = model:clone()
  local lrs = lrs_model:getParameters()
  lrs:fill(0)

  local count = 0
  utils.unfreeze_layers(lrs_model, n_unfreezed_layers, count)

  return lrs, lrs_model
end

function utils.unfreeze_layers(module, n_unfreezed_layers, count)
  if module.modules == nil then
    local is_target = false
    if module.weight ~= nil then
      module.weight:fill(1.0)
      is_target = true
    end
    if module.bias ~= nil then
      module.bias:fill(1.0)
      is_target = true
    end
    if is_target then
      count = count + 1
    end
  else
    for i = #module.modules, 1, -1 do
      count = utils.unfreeze_layers(module:get(i), n_unfreezed_layers, count)
      if count == n_unfreezed_layers then
        return count
      end
    end
  end

  return count
end

function utils.load_fps_file(fps_file_path)
  assert(paths.filep(fps_file_path), fps_file_path .. ' does not exist')
  local f = io.open(fps_file_path, 'r')
  local fps = f:read('*n')
  f:close()
  return fps
end

function utils.get_fps(video_path)
  require('torch-ffmpeg')
  video = FFmpeg(video_path)
  return video:stats().r_frame_rate
end

function utils.get_duration(video_path)
  require('torch-ffmpeg')
  video = FFmpeg(video_path)
  return video:stats().duration
end

function utils.get_n_frames(video_jpg_path)
  local lfs = require('lfs')
  local n_frames = 0
  for filename in lfs.dir(video_jpg_path) do
    if string.find(filename, 'image') then
      n_frames = n_frames + 1
    end
  end

  return n_frames
end

function utils.get_frame_size(frame_file_path)
  return  image.load(frame_file_path, 3, 'byte'):size()
end

function utils.get_cropping_box(box_width, box_height, image_width, image_height, position)
  if position == 'c' then
    local center_x = math.floor(image_width / 2)
    local center_y = math.floor(image_height / 2)
    local box_half_width = math.floor(box_width / 2)
    local box_half_height = math.floor(box_height / 2)
    return center_x - box_half_width + 1, center_y - box_half_height + 1,
        center_x + box_half_width, center_y + box_half_height
  elseif position == 'tl' then
    return 1, 1, box_width, box_height
  elseif position == 'tr' then
    return image_width - box_width + 1, 1, image_width, box_height
  elseif position == 'bl' then
    return 1, image_height - box_height + 1, box_width, image_height
  elseif position == 'br' then
    return image_width - box_width + 1, image_height - box_height + 1, image_width, image_height
  end

  return 1, 1, 1, 1
end

function utils.clone_table(t)
  if type(t) ~= "table" then
    return t
  end

  local meta = getmetatable(t)
  local target = {}
  for k, v in pairs(t) do
      if type(v) == "table" then
          target[k] = utils.clone_table(v)
      else
          target[k] = v
      end
  end
  setmetatable(target, meta)
  return target
end

function utils.get_n_frames(video_directory_path)
  local files = paths.dir(video_directory_path)
  if #files == 2 then
    return -1
  end

  local indices = {}
  for _, file in pairs(files) do
    if string.match(file, 'image') ~= nil then
      table.insert(indices, tonumber(string.sub(file, 7, 11)))
    end
  end

  table.sort(indices, function(a, b) return a > b end)
  return indices[1]
end

function utils.load_n_frames_file(file_path)
  assert(paths.filep(file_path), file_path .. ' does not exist')
  local f = io.open(file_path, 'r')
  local n_frames = f:read('*n')
  f:close()
  return n_frames
end

function utils.find_existing_frame_files(video_directory_path, begin_t, end_t)
  for t = end_t, begin_t, -1 do
    local file_path = paths.concat(video_directory_path, string.format('image_%05d.jpg', t))
    if paths.filep(file_path) then
      return t
    end
  end

  return begin_t - 1
end

function utils.calculate_overlap(base, other)
  local o_begin = math.max(base[1], other[1])
  local o_end = math.min(base[2], other[2])
  if o_begin >= o_end then
    return 0
  else
    return (o_end - o_begin + 1) /  (base[2] - base[1] + 1)
  end
end

function utils.calculate_iou(base, other)
  -- units of begin and end are sec
  local o_begin = math.max(base[1], other[1])
  local o_end = math.min(base[2], other[2])
  if o_begin >= o_end then
    return 0
  else
    local base_duration = base[2] - base[1]
    local other_duration = other[2] - other[1]
    local intersection = o_end - o_begin
    local union = base_duration + other_duration - intersection
    return intersection / union
  end
end

function utils.save_sample_image(base_name, sample)
  local save_sample = sample:permute(2, 1, 3, 4)
  for k = 1, save_sample:size(1) do
    for ch = 1, 3 do
      save_sample[{{k}, {ch}, {}, {}}]:add(mean[ch]):mul(1.0 / 255)
    end
    local save_path = base_name .. string.format("_%d.jpg", k)
    image.save(save_path, save_sample[k])
  end
end

return utils
