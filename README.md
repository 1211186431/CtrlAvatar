# OpenAvatar
## init
1.运行脚本
```
python mypre_data.py --config /home/ps/dy/OpenAvatar/config/config.yaml
```

2.复制smplx_model

3.复制last.ckpt和mate_info

4.准备tpose mesh

## train
1.运行
```
python main.py --mode train --config /home/ps/dy/OpenAvatar/config/config.yaml
```

## 
1.修改config中model_name

2.修改model_mode为test

```
python main.py --mode test --config /home/ps/dy/OpenAvatar/config/config.yaml
```
