import yaml
import subprocess
import os


yaml_file_path = '/home/ps/dy/CtrlAvatar/config/S4DDress.yaml'

with open(yaml_file_path, 'r') as file:
    config = yaml.safe_load(file)

base_path = config['base_path']
subject = config['subject']

img_path = os.path.join(base_path, 'outputs', 'test', subject, 'img_test')
output_path = os.path.join(img_path, f'output_{subject}.mp4')

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

try:
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video has been created at: {output_path}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running ffmpeg: {e}")