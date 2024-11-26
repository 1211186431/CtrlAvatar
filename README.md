# CtrlAvatar
## Train Geometry
训练几何 [具体流程](geometry/README.md)


## 数据预处理
准备纹理数据
```
export PYTHONPATH=$PYTHONPATH:/home/ps/dy/CtrlAvatar

python util/texture_process.py --config /home/ps/dy/OpenAvatar/config/S4DDress.yaml --subject 00122_Inner
```

## train
1. 运行
```
python main.py --mode train --config /home/ps/dy/CtrlAvatar/config/S4DDress.yaml --subject 00122_Inner
```

## test
修改 pkl_dir 为 xhuman中对应的 SMPLX 文件夹
```
bash script/test.sh
```

## eval
1. 保存多视角图片
```
python util/save_eval_data.py --subject 00016 --data_path /home/ps/dy/CtrlAvatar/outputs/test/00016/mesh_test --method Ctrl --out_dir /home/ps/dy/ctrl
```

2. 计算指标
```
python evaluate.py --subject 00016 --gt_npy /home/ps/dy/eval_aaai25/eval0805/eval_gt/gt_00016.npy --pre_npy /home/ps/dy/eval_aaai25/eval0805/eval_ours/Ours_00016.npy --method Ctrl --out_dir /home/ps/dy
```

## export video 
1. 切换到有 ffmpeg的环境
```
python util/export_video.py
```
