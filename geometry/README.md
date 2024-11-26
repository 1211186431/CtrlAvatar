# 训练几何

## 准备数据
1. 下载 [SMPLX](https://smpl-x.is.tue.mpg.de)

2. 下载 [init_model](https://github.com/Skype-line/X-Avatar)

3. 移动到对应位置

## SX-Humans
1. 下载 [X-Humans](https://github.com/Skype-line/X-Avatar)

2. 处理数据
修改```pre_SXHumans.sh```中数据集路径和输出路径
```
bash pre_SXHumans.sh
```

3. 训练
```
python train_delta.py subject=00016_delta datamodule=XHumans_scan_smplx experiments=XHumans_occ_smplx_delta datamodule.dataloader.dataset_path=/datapath
```

4. 测试
```
python test_delta.py subject=00016_delta experiments=XHumans_occ_smplx_delta demo.motion_path=/datapath/test
```

## SCustomHumans
1. 下载 [CustomHumans](https://github.com/custom-humans/editable-humans)

2. 处理数据
修改```preprocess_SCustomHumans.py```中数据集路径以及输出路径
```
bash pre_SCHumans.sh
```

3. 训练
```
python train_delta.py subject=00067_delta datamodule=CustomHumans_scan_smplx experiments=CHumans_occ_smplx_delta datamodule.dataloader.dataset_path=/datapath
```

4. 测试
```
python test_delta.py subject=00067_delta experiments=CHumans_occ_smplx_delta demo.motion_path=/datapath/test
```

## 4D-Dress
1. 下载 [4D-Dress](https://github.com/eth-ait/4d-dress)

2. 处理数据

修改 ```preprocess_4DDress.py```中```DATASET_DIR```为数据集根目录，并指定输出路径
```
bash pre_S4DDress.sh
```


3. 训练
```
python train_delta.py subject=00122_Inner datamodule=Dress_scan_smplx experiments=Dress_occ_smplx_delta datamodule.dataloader.dataset_path=/datapath
```

4. 测试
```
python test_delta.py subject=00122_Inner experiments=Dress_occ_smplx_delta demo.motion_path=/datapath/test
```

<!-- 训练XAvatar
```
python train.py subject=00122_Inner datamodule=Dress_scan_smplx experiments=Dress_occ_smplx_baseline datamodule.dataloader.dataset_path=/datapath
```

测试XAvatar
```
python test.py subject=000122_Inner experiments=Dress_occ_smplx_baseline demo.motion_path=/datapath/test
``` -->
<!-- python train.py subject=000122_Inner datamodule=Dress_scan_smplx experiments=Dress_occ_smplx_baseline datamodule.dataloader.dataset_path=/home/ps/dy/dataset/S4d/00122_Inner -->