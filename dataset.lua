print('prepare data')
assert(opt.dataset == 'activitynet' or opt.datset == 'kinetics',
   'dataset must be activitynet or kinetics')
if opt.dataset == 'activitynet' then
  dataset_utils = dofile('activitynet_utils.lua')
elseif opt.dataset == 'kinetics' then
  dataset_utils = dofile('kinetics_utils.lua')
end

local data = dataset_utils.load_annotation_data(opt.annotation_path)
local training_video_names, training_annotations,
    validation_video_names, validation_annotations,
    test_video_names =
    dataset_utils.get_data_names_and_annotations(data)
local class_labels_map = dataset_utils.get_class_labels(data)
class_names_map = {}
for name, label in pairs(class_labels_map) do
    class_names_map[label] = name
end

function prepare_data(video_names, annotations)
  local dataset = {}
  for i = 1, #video_names do
    local video_path = paths.concat(opt.video_path, video_names[i])
    if paths.dirp(video_path) then
      local fps_file_path = paths.concat(video_path, 'fps')
      local fps = utils.load_fps_file(fps_file_path)
      for _, one_annotation in pairs(annotations[i]) do
        local begin_t = math.ceil(one_annotation.segment[1] * fps)
        local end_t = math.ceil(one_annotation.segment[2] * fps)
        if begin_t == 0 then
          begin_t = 1
        end
        local sample = {
            video = video_path,
            segment = {begin_t, end_t},
            label = class_labels_map[one_annotation.label],
            fps = fps
        }
        table.insert(dataset, sample)
      end
    end
  end

  return dataset
end

function prepare_data_kinetics(video_names, annotations)
  local dataset = {}
  for i = 1, #video_names do
    local video_path = paths.concat(opt.video_path, video_names[i])
    if paths.dirp(video_path) then
      local n_frames_file_path = paths.concat(video_path, 'n_frames')
      local n_frames = utils.load_n_frames_file(n_frames_file_path)
      if n_frames > 0 then
        n_frames = math.floor(n_frames)
        local begin_t = 1
        local end_t = n_frames
        local sample = {
            video = video_path,
            segment = {begin_t, end_t},
            n_frames = n_frames,
            video_id = string.sub(video_names[i], 1, -15)
        }
        if annotations ~= nil then
          sample.label = class_labels_map[annotations[i].label]
        end
        table.insert(dataset, sample)
      end
    end
  end

  return dataset
end

function prepare_data_per_video(video_names, annotations)
  local dataset = {}
  for i = 1, #video_names do
    local video_path = paths.concat(opt.video_path, video_names[i])
    if paths.dirp(video_path) then
      local n_frames = math.floor(utils.get_n_frames(video_path))
      local fps_file_path = paths.concat(video_path, 'fps')
      local fps = utils.load_fps_file(fps_file_path)
      local video_data = {}
      video_data.video = video_path
      video_data.video_id = string.sub(video_names[i], 3, -1)
      video_data.n_frames = n_frames
      video_data.fps = fps
      video_data.annotations = {}

      if annotations ~= nil then
        for _, one_annotation in pairs(annotations[i]) do
          local begin_t = math.ceil(one_annotation.segment[1] * fps)
          local end_t = math.ceil(one_annotation.segment[2] * fps)
          if begin_t == 0 then
            begin_t = 1
          end
          local sample = {
              segment = {begin_t, end_t},
              label = class_labels_map[one_annotation.label]
          }
          table.insert(video_data.annotations, sample)
        end
      end

      table.insert(dataset, video_data)
    end
  end

  return dataset
end

if not opt.no_train then
  if opt.dataset == 'activitynet' then
    training_data = prepare_data(training_video_names, training_annotations)
  elseif opt.dataset == 'kinetics' then
    training_data = prepare_data_kinetics(training_video_names, training_annotations)
  end
end
if not opt.no_test then
  if opt.dataset == 'activitynet' then
    validation_data = prepare_data(validation_video_names, validation_annotations)
  elseif opt.dataset == 'kinetics' then
    validation_data = prepare_data_kinetics(validation_video_names, validation_annotations)
  end
end

if opt.test_video then
  if opt.dataset == 'activitynet' then
    if opt.test_subset == 'val' then
      video_test_data = prepare_data_per_video(validation_video_names, validation_annotations)
    elseif opt.test_subset == 'test' then
      video_test_data = prepare_data_per_video(test_video_names)
    end
  elseif opt.dataset == 'kinetics' then
    if opt.test_subset == 'val' then
      video_test_data = prepare_data_kinetics(validation_video_names, validation_annotations)
    elseif opt.test_subset == 'test' then
      video_test_data = prepare_data_kinetics(test_video_names)
    end
  end
end
