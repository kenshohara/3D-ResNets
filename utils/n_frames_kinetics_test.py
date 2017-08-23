import os
import sys
import subprocess

dir_path = sys.argv[1]

class_name = 'test'
class_path = os.path.join(dir_path, class_name)

for file_name in os.listdir(class_path):
  video_dir_path = os.path.join(class_path, file_name)
  image_indices = []
  for image_file_name in os.listdir(video_dir_path):
    if 'image' not in image_file_name:
      continue
    image_indices.append(int(image_file_name[6:11]))
    
  if len(image_indices) == 0:
    print('no image files', video_dir_path)
    n_frames = 0
  else:
    image_indices.sort(reverse=True)
    n_frames = image_indices[0]
    print(video_dir_path, n_frames)
  with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
    dst_file.write(str(n_frames))
