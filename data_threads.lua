print('run loader threads')
threads = require('threads')
data_loader = dofile('data_loader.lua')
do
  local options = opt
  local train = training_data
  local val = validation_data
  local video_train = video_training_data
  local video_val = video_validation_data
  local video_test = video_test_data
  task_queue = threads.Threads(
      opt.n_threads,
      function()
        require('torch')
        require('image')
        dofile('mean.lua')
        utils = dofile('utils.lua')
        data_loader = dofile('data_loader.lua')
      end,
      function(thread_id)
        opt = options
        id = thread_id
        local seed = opt.manual_seed + id
        torch.manualSeed(seed)
        training_data = train
        validation_data = val
        video_training_data = video_train
        video_validation_data = video_val
        video_test_data = video_test
      end
  )
end
