# 训练几何

## 准备数据
1. 下载 [SMPLX](https://smpl-x.is.tue.mpg.de)
2. 下载 [init_model](https://github.com/Skype-line/X-Avatar)
3. 移动到对应位置

## 处理SX-Humans
1. 下载 [X-Humans](https://github.com/Skype-line/X-Avatar)
2. 处理 数据
```
python data_process/preprocess_XHumans.py --data_root=/path/to/XHumans/{Person_ID}
```
3. 处理SX-Humans
```
python data_process/preprocess_SXHumans.py --subject 00016 --xhumans_path /path/to/XHumans --out_path outpath
```
