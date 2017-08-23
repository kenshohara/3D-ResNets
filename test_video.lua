local batch_number
local video_outputs
local next_clip_index
local timer = torch.Timer()

local function prepare_test(data)
  local clips = {}
  local n_frames = data.n_frames
  local video_path = data.video
  for t = 1, n_frames - opt.sample_duration, opt.sample_duration do
    local sample_begin_t = t
    local sample_end_t = t + opt.sample_duration - 1
    local sample_data = {}
    sample_data.video = video_path
    sample_data.segment = {sample_begin_t, sample_end_t}
    table.insert(clips, sample_data)
  end

  local video_id = data.video_id

  return clips, video_id
end

function test_video()
  batch_number = 0
  if not opt.no_cuda then
    cutorch.synchronize()
  end
  timer:reset()

  model:evaluate()
  print('==> test video accuracy')

  local output = {
    results = {},
    version = 'VERSION 1.3',
    external_data = {
      used = false,
      details = '...'
    }
  }
  for i = 1, #video_test_data do
    local clips, video_id = prepare_test(video_test_data[i])
    output.results[video_id] = {}
    video_outputs = torch.Tensor(#clips, opt.n_classes):fill(0)
    next_clip_index = 1
    for j = 1, #clips, opt.batch_size do
      task_queue:addjob(
          function()
            collectgarbage()
            local size = math.min(opt.batch_size, #clips - j + 1)
            local inputs = torch.Tensor(size, 3, opt.sample_duration,
                                        opt.sample_size, opt.sample_size)

            local end_k = math.min((j + opt.batch_size - 1), #clips)
            for k = j, end_k do
              local video_directory_path = clips[k].video
              local begin_t = clips[k].segment[1]
              local end_t = clips[k].segment[2]

              local sample = data_loader.load_center_sample(
                  video_directory_path, opt.sample_size, begin_t, end_t)

              inputs[k - j + 1] = sample
            end

            collectgarbage()

            return inputs
          end,
          test_video_batch
      )
    end
    task_queue:synchronize()
    if not opt.no_cuda then
      cutorch.synchronize()
    end

    if #clips ~= 0 then
      mean_outputs = torch.mean(video_outputs, 1)
      local scores_sorted, scores_sorted_loc = mean_outputs:float():sort(2, true)
      for rank = 1, 10 do
        local current_result = {
          score = scores_sorted[1][rank],
          label = class_names_map[scores_sorted_loc[1][rank]]
        }
        table.insert(output.results[video_id], current_result)
      end
    end

    print(string.format('[%d/%d]', i, #video_test_data))
    if i % 500 == 0 then
      local result_json_file_path = paths.concat(opt.result_path, string.format('%s_video.json', opt.test_subset))
      json.save(result_json_file_path, output)
    end
  end

  local result_json_file_path = paths.concat(opt.result_path, string.format('%s_video.json', opt.test_subset))
  local json = require('json')
  print('save json file')
  json.save(result_json_file_path, output)
end

local inputs
if not opt.no_cuda then
  inputs = torch.CudaTensor()
end

function test_video_batch(inputs_cpu)
  local batch_size = inputs_cpu:size(1)
  if batch_size < 10 then
    local new_size = inputs_cpu:size()
    new_size[1] = new_size[1] * 2
    inputs_cpu = inputs_cpu:resize(new_size)
    inputs_cpu[{{batch_size + 1, new_size[1]}, {}, {}, {}}] =
        inputs_cpu[{{1, batch_size}, {}, {}, {}, {}}]
  end

  if not opt.no_cuda then
    inputs:resize(inputs_cpu:size()):copy(inputs_cpu)
  else
    inputs = inputs_cpu
  end

  local outputs = model:forward(inputs)
  if outputs:dim() == 1 then
    outputs = outputs:reshape(inputs:size(1), outputs:size(1) / inputs:size(1))
  end
  if not opt.no_cuda then
    cutorch.synchronize()
  end

  outputs = outputs:float()
  for i = 1, batch_size do
    local index = next_clip_index + i - 1
    video_outputs[index] = outputs[i]
  end
  next_clip_index = next_clip_index + batch_size
end
