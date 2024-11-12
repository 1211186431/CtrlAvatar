# CtrlAvatar
## Train Geometry
训练几何 [具体流程](geometry/README.md)

## pre data
对于XHumans数据集

1. 修改[base_path](config/SXHumans.yaml)中``base_path``为当前目录;
``data_path``为数据集路径

2. 运行脚本
```
bash script/process_data.sh
```

对于CustomHumans数据集
```
python dataset/process_CustomHumans.py  --base_path /home/ps/dy/c_data/CustomHumans  --out_dir_path /home/ps/dy/mycode2/t0628 --subject 00017 --gender male --test_len 2
```


## train
1. 运行
```
python main.py --mode train --config /home/ps/dy/OpenAvatar/config/config.yaml
```

## test
修改 pkl_dir 为 xhuman中对应的 SMPLX 文件夹
```
bash script/test.sh
```

## eval
1. 保存多视角图片
```
python save_eval_data.py --subject 00016 --data_path /home/ps/dy/CtrlAvatar/outputs/test/00016/mesh_test --method Ctrl --out_dir /home/ps/dy/ctrl
```

2. 计算指标
```
python evaluate.py --subject 00016 --gt_npy /home/ps/dy/eval_aaai25/eval0805/eval_gt/gt_00016.npy --pre_npy /home/ps/dy/eval_aaai25/eval0805/eval_ours/Ours_00016.npy --method Ctrl --out_dir /home/ps/dy
```

## export video 
1. 切换到有 ffmpeg的环境
```
python export_video.py
```
