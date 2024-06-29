# OpenAvatar
## init
1.运行脚本
```
python mypre_data.py --config /home/ps/dy/OpenAvatar/config/config.yaml
```

2.复制 smplx_model

3.复制 last.ckpt和 meta_info

4.准备 tpose mesh

## train
1.运行
```
python main.py --mode train --config /home/ps/dy/OpenAvatar/config/config.yaml
```

## test
1.修改 config 中 model_name

2.修改 model_mode 为 test

3.修改 pkl_dir 为 xhuman中对应的 SMPLX 文件夹

```
python main.py --mode test --config /home/ps/dy/OpenAvatar/config/config.yaml
```

## export video 
1.切换到有 ffmpeg的环境
```
python export_video.py
```
