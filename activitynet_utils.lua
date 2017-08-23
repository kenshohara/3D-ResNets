local activitynet_utils = {}

function activitynet_utils.load_annotation_data(data_file_path)
  return json.load(data_file_path)
end

function activitynet_utils.get_class_labels(data)
  local leaf_names = {}
  for k, v in pairs(data.taxonomy) do
    local is_leaf = true
    for k2, v2 in pairs(data.taxonomy) do
      if v2.parentId == v.nodeId then
        is_leaf = false
        break
      end
    end
    if is_leaf then
      table.insert(leaf_names, v.nodeName)
    end
  end

  local class_labels = {}
  for k, v in pairs(leaf_names) do
    class_labels[v] = k
  end

  return class_labels
end

function activitynet_utils.get_data_names_and_annotations(data, video_directory_path)
  local training_video_names = {}
  local training_annotations = {}
  local validation_video_names = {}
  local validation_annotations = {}
  local test_video_names = {}
  local test_annotations = {}
  for k, v in pairs(data.database) do
    subset = v.subset
    if subset == 'testing' then
      table.insert(test_video_names, string.format('v_%s', k))
      table.insert(test_annotations, v.annotations)
    else
      if subset == 'training' then
        table.insert(training_video_names, string.format('v_%s', k))
        table.insert(training_annotations, v.annotations)
      else
        table.insert(validation_video_names, string.format('v_%s', k))
        table.insert(validation_annotations, v.annotations)
      end
    end
  end

  return training_video_names, training_annotations,
    validation_video_names, validation_annotations,
    test_video_names, test_annotations
end

function activitynet_utils.calculate_fps_stats(data_file_path, video_directory_path)
  require('paths')
  require('utils')

  local fps_lists = {}

  local data = load_annotation_data(data_file_path)
  local n_videos = 0
  for _, _ in pairs(data.database) do
    n_videos = n_videos + 1
  end

  local i = 1
  for k, v in pairs(data.database) do
    xlua.progress(i, n_videos)

    local video_name = string.format('v_%s.mp4', k)
    local video_file_path = paths.concat(video_directory_path, video_name)
    if paths.filep(video_file_path) then
      table.insert(fps_lists, utils.get_fps(video_file_path))
    end

    i = i + 1
  end

  fps_lists = torch.Tensor(fps_lists)

  print(torch.mean(fps_lists), torch.std(fps_lists))
end

return activitynet_utils
