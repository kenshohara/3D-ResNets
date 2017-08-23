# 3D ResNets for Action Recognition
This is the torch code for the following paper:

Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,  
"Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition".

## Citation
If you use this code or pre-trained models, please cite the following:
```
@article{

}
```

## Pre-trained models

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
* Download datasets
* Convert from avi to jpg files using utils/video_jpg.py
```
python utils/video_jpg.py avi_video_directory jpg_video_directory
```
* Generate fps files using utils/fps.py
```
python utils/fps.py avi_video_directory jpg_video_directory
```

### Kinetics
* Download datasets
* Convert from avi to jpg files using utils/video_jpg_kinetics.py
```
python utils/video_jpg_kinetics.py avi_video_directory jpg_video_directory
```
* Generate n_frames files using utils/n_frames_kinetics.py
```
python utils/n_frames_kinetics.py jpg_video_directory
```
* Generate annotation file in json format similar to ActivityNet using utils/kinetics_json.py
```
python utils/kinetics_json.py train_csv_path val_csv_path test_csv_path json_path
```

## Running the code
```
th main.py
```
