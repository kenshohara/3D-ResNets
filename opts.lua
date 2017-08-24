local opts = {}
-- parameters and others config
function opts.parse(arg)
  local cmd = torch.CmdLine()
  cmd:option('--root_path',               '', 'Root directory path of data.')
  cmd:option('--video_path',              '', 'Directory path of Videos')
  cmd:option('--annotation_path',         '', 'Annotation file path')
  cmd:option('--premodel_path',           '', 'Pretrained model file path')
  cmd:option('--bgr_premodel',            false, 'If true then the pretrained model is bgr else rgb')
  cmd:option('--result_path',             'results', 'Result directory path')
  cmd:option('--dataset',                 'activitynet', 'Name of the dataset (activitynet | kinetics)')
  cmd:option('--lr_path',                 'File path of learning rate scheduling. If \'\' then lr_path = result_path/lr.lua')
  cmd:option('--n_classes',               200, 'Number of classes')
  cmd:option('--sample_size',             112, 'Height and width of inputs')
  cmd:option('--sample_duration',         16, 'Temporal duration of inputs')
  cmd:option('--initial_scale',           1.0, 'Initial scale for multiscale cropping')
  cmd:option('--n_scales',                5, 'Number of scales for multiscale cropping')
  cmd:option('--scale_step',              0.84089641525, 'Scale step for multiscale cropping')
  cmd:option('--learning_rate',           1e-3, 'Learning rate. If learning rate scheduling by lr_path is enabled, this is ignored.')
  cmd:option('--weight_decay',            0.0, 'Weight decay. If learning rate scheduling by lr_path is enabled, this is ignored.')
  cmd:option('--momentum',                0.9, 'Momentum.')
  cmd:option('--dropout_ratio',           0.5, 'Dropout ratio')
  cmd:option('--batch_size',              40, 'Batch Size')
  cmd:option('--n_epochs',                1000, 'Number of total epochs to run')
  cmd:option('--begin_epoch',             1, 'Training begins at this epoch. Previous trained model at result_path is loaded.')
  cmd:option('--n_val_samples',           3, 'Number of validation samples for each activity')
  cmd:option('--no_train',                false, 'If true training is not performed.')
  cmd:option('--no_test',                 false, 'If true testing is not performed.')
  cmd:option('--test_video',              false, 'If true testing on each video is performed.')
  cmd:option('--no_cuda',                 false, 'If true cuda is not used')
  cmd:option('--gpu_id',                  1, 'ID of GPU to use')
  cmd:option('--n_gpus',                  2, 'Number of GPUs to use')
  cmd:option('--n_threads',               4, 'Number of threads for multi-thread loading')
  cmd:option('--checkpoint',              10, 'Trained model is saved at every this epochs.')
  cmd:option('--manual_seed',             0, 'Manually set random seed')
  cmd:option('--crop',                    'c', 'Spatial cropping method. random is uniform. corner is selection from 4 corners and 1 cetner. (r | c)')
  cmd:option('--no_hflip',                false, 'If true holizontal flipping is not performed.')
  cmd:option('--batch_norm',              false, 'If true batch normalization is added to C3D')
  cmd:option('--global_pooling',          false, 'If true global average pooling is added to C3D')
  cmd:option('--model',                   'c3d', 'Network model (c3d | resnet)')
  cmd:option('--resnet_depth',            18, 'Depth of resnet (10 | 18 | 34 | 50 | 101 | 152)')
  cmd:option('--shortcut_type',           'A', 'Shortcut type of resnet (A | B | C)')
  cmd:option('--weight_init',             false, 'Weight initialization for resnet')
  cmd:option('--test_subset',             'val', 'Subset to use for test_video (val | test)')
  cmd:option('--freeze_params',           0, '0 means all layers are trained. 1 means last layer is trained.')

  local opt = cmd:parse(arg or {})

  if opt.root_path ~= '' then
    opt.video_path = paths.concat(opt.root_path, opt.video_path)
    opt.annotation_path = paths.concat(opt.root_path, opt.annotation_path)
    opt.premodel_path = paths.concat(opt.root_path, opt.premodel_path)
    opt.result_path = paths.concat(opt.root_path, opt.result_path)
  end
  if opt.lr_path == '' then
    opt.lr_path = paths.concat(opt.result_path, 'lr.lua')
  end
  if paths.filep(opt.lr_path) then
    dofile(opt.lr_path)
  end
  opt.scales = {opt.initial_scale}
  for i = 1, opt.n_scales - 1 do
    table.insert(opt.scales, opt.scales[#opt.scales] * opt.scale_step)
  end

  return opt
end

return opts
