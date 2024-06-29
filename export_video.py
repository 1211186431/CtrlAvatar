import yaml
import subprocess
import os

# 定义YAML文件的路径
yaml_file_path = '/home/ps/dy/OpenAvatar/config/config.yaml'

# 读取YAML文件
with open(yaml_file_path, 'r') as file:
    config = yaml.safe_load(file)

# 从YAML文件中获取数据
base_path = config['base_path']
subject = config['subject']

# 构建图像和输出路径
img_path = os.path.join(base_path, 'outputs', 'test', subject, 'img_test')
output_path = os.path.join(img_path, f'output_{subject}.mp4')

# 构建ffmpeg命令
ffmpeg_cmd = [
    'ffmpeg',
    '-framerate', '30',
    '-i', os.path.join(img_path, 'test_%d.png'),
    '-c:v', 'libx264',
    '-profile:v', 'high',
    '-crf', '20',
    '-pix_fmt', 'yuv420p',
    output_path
]

# 执行ffmpeg命令
try:
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video has been created at: {output_path}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running ffmpeg: {e}")