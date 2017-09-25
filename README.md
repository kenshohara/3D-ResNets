# 3D ResNets for Action Recognition
This is the torch code for the following paper:

Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,  
"Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition",  
arXiv preprint, arXiv:1708.07632, 2017.

The paper will appear in ICCV 2017 Workshop (Chalearn).  

This code includes only training and testing on the ActivityNet and Kinetics datasets.  
**If you want to classify your videos using our pretrained models,
use [this code](https://github.com/kenshohara/video-classification-3d-cnn).**

**The PyTorch (python) version of this code is available [here](https://github.com/kenshohara/3D-ResNets-PyTorch).**

## Citation
If you use this code or pre-trained models, please cite the following:
```
@article{hara3dresnets
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh}
  title={Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition}
  journal={arXiv preprint}
  volume={arXiv:1708.07632}
  year={2017}
}
```

## Pre-trained models
Pre-trained models are available at [releases](https://github.com/kenshohara/3D-ResNets/releases/tag/1.0).

## Requirements
* [Torch](http://torch.ch/)
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```
* [json package](https://github.com/clementfarabet/lua---json)
```
luarocks install json
```
* FFmpeg, FFprobe
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```
* Python 3

## Preparation
### ActivityNet
* Download datasets using official crawler codes
* Convert from avi to jpg files using ```utils/video_jpg.py```
```
python utils/video_jpg.py avi_video_directory jpg_video_directory
```
* Generate fps files using ```utils/fps.py```
```
python utils/fps.py avi_video_directory jpg_video_directory
```

### Kinetics
* Download datasets using official crawler codes
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```utils/video_jpg_kinetics.py```
```
python utils/video_jpg_kinetics.py avi_video_directory jpg_video_directory
```
* Generate n_frames files using ```utils/n_frames_kinetics.py```
```
python utils/n_frames_kinetics.py jpg_video_directory
```
* Generate annotation file in json format similar to ActivityNet using ```utils/kinetics_json.py```
```
python utils/kinetics_json.py train_csv_path val_csv_path test_csv_path json_path
```

## Running the code
Assume the structure of data directories is the following:
```
~/
  data/
    activitynet_videos/
      jpg/
        .../ (directories of video names)
          ... (jpg files)
    kinetics_videos/
      jpg/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    models/
      resnet.t7
    results/
      model_100.t7
    LR/
      ActivityNet/
        lr.lua
      Kinetics/
        lr.lua
    kinetics.json
    activitynet.json
```

Confirm all options.
```
th main.lua -h
```

Train ResNets-34 on the Kinetics dataset (400 classes) with 4 CPU threads (for data loading) and 2 GPUs.  
Batch size is 128.  
Save models at every 5 epochs.
```
th main.lua --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --lr_path LR/Kinetics/lr.lua --dataset kinetics --model resnet \
--resnet_depth 34 --n_classes 400 --batch_size 128 --n_gpu 2 --n_threads 4 --checkpoint 5
```

Continue Training from epoch 101. (~/data/results/model_100.t7 is loaded.)
```
th main.lua --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --lr_path LR/Kinetics/lr.lua --dataset kinetics --begin_epoch 101 \
--batch_size 128 --n_gpu 2 --n_threads 4 --checkpoint 5
```

Perform recognition for each video of validation set using pretrained model.
This operation outputs top-10 labels for each video.
```
th main.lua --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --premodel_path models/resnet.t7 --dataset kinetics \
--no_train --no_val --test_video --test_subset val --n_gpu 2 --n_threads 4
```
