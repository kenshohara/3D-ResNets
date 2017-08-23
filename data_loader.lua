local data_loader = {}

local function random_clip(begin_t, end_t, sample_duration)
  local range_end = end_t - (sample_duration - 1)
  if begin_t > range_end then
    begin_t, range_end = range_end, begin_t
  end
  local sample_begin_t = math.ceil(torch.uniform(begin_t, range_end))
  local sample_end_t = sample_begin_t + sample_duration - 1

  return sample_begin_t, sample_end_t
end

local function padding(frames, n_pad)
  for i = 1, n_pad do
    table.insert(frames, frames[i])
  end
end

local function random_crop(scales)
  local scale = scales[torch.random(#scales)]
  local tl_x_ratio = torch.uniform(0.0, 1.0)
  local tl_y_ratio = torch.uniform(0.0, 1.0)
  return tl_x_ratio, tl_y_ratio, scale
end

local function random_corner_crop(scales)
  local scale = scales[torch.random(#scales)]
  local crop_positions = {'c', 'tl', 'tr', 'bl', 'br'}
  local crop_position = crop_positions[torch.random(#crop_positions)]
  return scale, crop_position
end

local function crop(frames, tl_x, tl_y, br_x, br_y)
  local cropped_frames = {}
  for i = 1, #frames do
    table.insert(cropped_frames, frames[i][{{}, {tl_y, br_y}, {tl_x, br_x}}])
  end

  return cropped_frames
end

local function flip(frames)
  local flipped_frames = {}
  for i = 1, #frames do
    table.insert(flipped_frames, image.hflip(frames[i]))
  end

  return flipped_frames
end

local function load_clip(video_directory_path, begin_t, end_t, sample_duration, sample_size)
  local clip = data_loader.load_frames(video_directory_path, begin_t, end_t)
  if #clip == 0 then
    for i = 1, sample_duration do
      table.insert(clip, torch.Tensor(3, sample_size, sample_size):fill(0))
    end
  end

  local n_pad = sample_duration - #clip
  if n_pad ~= 0 then
    padding(clip, n_pad)
  end

  return clip
end

local function decide_random_crop_positions(tl_x_ratio, tl_y_ratio, scale, frame_size)
  local frame_width = frame_size[3]
  local frame_height = frame_size[2]
  local min_length = math.min(frame_width, frame_height)
  local crop_size = min_length * scale

  local tl_x = math.floor((frame_width - crop_size) * tl_x_ratio + 1)
  local tl_y = math.floor((frame_height - crop_size) * tl_y_ratio + 1)
  local br_x = tl_x + crop_size - 1
  local br_y = tl_y + crop_size - 1

  return tl_x, tl_y, br_x, br_y
end

local function decide_corner_crop_positions(crop_position, scale, frame_size)
  local frame_width = frame_size[3]
  local frame_height = frame_size[2]
  local min_length = math.min(frame_width, frame_height)
  local crop_size = math.floor(min_length * scale)

  return utils.get_cropping_box(crop_size, crop_size, frame_width, frame_height, crop_position)
end

local function decide_crop_positions(frame_size, sample_size, scales, crop)
  if crop == 'r' then
    local tl_x_ratio, tl_y_ratio, scale = random_crop(scales)
    return decide_random_crop_positions(tl_x_ratio, tl_y_ratio, scale, frame_size)
  elseif crop == 'c' then
    local scale, crop_position = random_corner_crop(scales)
    return decide_corner_crop_positions(crop_position, scale, frame_size)
  end
end

local function execute_cropping(clip, sample_size, scales, crop)
  local frame_size = clip[1]:size()
  local tl_x, tl_y, br_x, br_y = decide_crop_positions(frame_size, sample_size, scales, crop)

  local sample_frames = crop(clip, tl_x, tl_y, br_x, br_y)
  return sample_frames
end

local function assign_torch_tensor(sample_frames, sample_size, sample_duration)
  local sample = torch.Tensor(sample_duration, 3, sample_size, sample_size)
  if sample_frames[1]:size(2) == sample_size then
    for i = 1, #sample_frames do
      sample[i] = sample_frames[i]
    end
  else
    for i = 1, #sample_frames do
      sample[i] = image.scale(sample_frames[i], sample_size, sample_size)
    end
  end

  return sample
end

local function normalize_and_subtract_mean(sample)
  sample:mul(255)
  for i = 1, 3 do
    sample[{{}, {i}, {}, {}}]:add(-mean[i])
  end
end

function data_loader.load_frames(video_directory_path, begin_t, end_t)
  local frames = {}
  for i = begin_t, end_t do
    local file_path = paths.concat(video_directory_path,
                                   string.format('image_%05d.jpg', i))
    if paths.filep(file_path) then
      table.insert(frames, image.load(file_path, 3, 'float'))
    end
  end

  return frames
end

function data_loader.load_random_sample(
    video_directory_path, begin_t, end_t, sample_size, sample_duration, scales, no_hflip, crop)
  local sample_begin_t, sample_end_t = random_clip(begin_t, end_t, sample_duration)
  sample_end_t = sample_end_t

  local clip = load_clip(video_directory_path, sample_begin_t, sample_end_t, sample_duration, sample_size)
  local sample_frames = execute_cropping(clip, sample_size, scales, crop)

  if (not no_hflip) and torch.random(0, 1) == 1 then
    sample_frames = flip(sample_frames)
  end

  local sample = assign_torch_tensor(sample_frames, sample_size, sample_duration)
  normalize_and_subtract_mean(sample)

  return sample:permute(2, 1, 3, 4)
end

function data_loader.load_center_sample(video_directory_path, sample_size, begin_t, end_t)
  local sample_duration = end_t - begin_t + 1
  local end_t = end_t
  local clip = load_clip(video_directory_path, begin_t, end_t, sample_duration, sample_size)
  local frame_size = clip[1]:size()
  local frame_width = frame_size[3]
  local frame_height = frame_size[2]
  local min_length = math.min(frame_width, frame_height)
  local tl_x, tl_y, br_x, br_y = utils.get_cropping_box(
      min_length, min_length, frame_width, frame_height, 'c')
  local sample_frames = crop(clip, tl_x, tl_y, br_x, br_y)

  local sample = assign_torch_tensor(sample_frames, sample_size, sample_duration)
  normalize_and_subtract_mean(sample)

  return sample:permute(2, 1, 3, 4)
end

return data_loader
