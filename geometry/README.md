# 训练几何

## 准备数据
1. 下载 [SMPLX](https://smpl-x.is.tue.mpg.de)
2. 下载 [init_model](https://github.com/Skype-line/X-Avatar)
3. 移动到对应位置

## SX-Humans
1. 下载 [X-Humans](https://github.com/Skype-line/X-Avatar)
2. 处理 数据
```
python data_process/preprocess_XHumans.py --data_root=/path/to/XHumans/{Person_ID}
```

3. 处理SX-Humans
```
python data_process/preprocess_SXHumans.py --subject 00016 --xhumans_path /path/to/XHumans --out_path outpath
```

## 4D-Dress
1.下载 [4D-Dress](https://github.com/eth-ait/4d-dress)

2.处理数据

修改 ```DATASET_DIR```为数据集根目录
```
python data_process/preprocess_4DDress.py --subj 00122 --outfit Inner --gender male  --out /datapath
```

3.运行CtrlAvatar
```
python train_delta.py subject=000122_Inner datamodule=Dress_scan_smplx experiments=Dress_occ_smplx_delta datamodule.dataloader.dataset_path=/datapath

```


运行XAvatar
```
python train.py subject=000122_Inner datamodule=Dress_scan_smplx experiments=Dress_occ_smplx_baseline datamodule.dataloader.dataset_path=/datapath

```
<!-- python train.py subject=000122_Inner datamodule=Dress_scan_smplx experiments=Dress_occ_smplx_baseline datamodule.dataloader.dataset_path=/home/ps/dy/dataset/S4d/00122_Inner -->