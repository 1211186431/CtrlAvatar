# OpenAvatar
## pre data
对于XHumans数据集

0. 修改config文件中 ```geometry_model_path```为XAvatar输出路径

1. 运行脚本
```
python mypre_data.py --config /home/ps/dy/OpenAvatar/config/config.yaml
```

对于CustomHumans数据集
```
python dataset/process_CustomHumans.py  --base_path /home/ps/dy/c_data/CustomHumans  --out_dir_path /home/ps/dy/mycode2/t0628 --subject 00017 --gender male --test_len 2
```

2. 复制 smplx_model


## train
1. 运行
```
python main.py --mode train --config /home/ps/dy/OpenAvatar/config/config.yaml
```

## test
1. 修改 config 中 model_name

2. 修改 model_mode 为 test

3. 修改 pkl_dir 为 xhuman中对应的 SMPLX 文件夹

```
python main.py --mode test --config /home/ps/dy/OpenAvatar/config/config.yaml
```

## export video 
1. 切换到有 ffmpeg的环境
```
python export_video.py
```
