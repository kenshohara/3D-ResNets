local kinetics_utils = {}

function kinetics_utils.load_annotation_data(data_file_path)
  return json.load(data_file_path)
end

function kinetics_utils.get_class_labels(data)
  class_labels_map = {}
  for k, v in pairs(data.labels) do
    class_labels_map[v] = k
  end
  return class_labels_map
end

function kinetics_utils.get_data_names_and_annotations(data, video_directory_path)
  local training_video_names = {}
  local training_annotations = {}
  local validation_video_names = {}
  local validation_annotations = {}
  local test_video_names = {}
  local test_annotations = {}
  for k, v in pairs(data.database) do
    subset = v.subset
    if subset == 'testing' then
      table.insert(test_video_names, string.format('test/%s', k))
      table.insert(test_annotations, v.annotations)
    else
      label = v.annotations.label
      if subset == 'training' then
        table.insert(training_video_names, string.format('%s/%s', label, k))
        table.insert(training_annotations, v.annotations)
      else
        table.insert(validation_video_names, string.format('%s/%s', label, k))
        table.insert(validation_annotations, v.annotations)
      end
    end
  end

  return training_video_names, training_annotations,
    validation_video_names, validation_annotations,
    test_video_names, test_annotations
end

return kinetics_utils
