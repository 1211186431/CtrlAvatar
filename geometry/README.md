# Geometry Training Guide
This guide explains the steps to complete geometry training tasks, including data preparation, processing, training, testing, and visualization. The code is modified based on [X-Avatar](https://github.com/Skype-line/X-Avatar).

## Data Preparation
1. Download [SMPLX Model(v1.1)](https://smpl-x.is.tue.mpg.de)  
```
mkdir geometry/code/lib/smplx/smplx_model/
mv /path/to/SMPLX_FEMALE.npz geometry/code/lib/smplx/smplx_model/SMPLX_FEMALE.npz
mv /path/to/SMPLX_MALE.npz geometry/code/lib/smplx/smplx_model/SMPLX_MALE.npz
```

2. Download [init_model](https://github.com/Skype-line/X-Avatar) from [X-Avatar](https://github.com/Skype-line/X-Avatar)
```
mv /path/to/init_model geometry/code/init_model
```

## SX-Humans Dataset
1. Download [X-Humans](https://github.com/Skype-line/X-Avatar)  

2. Process the data
- Modify the dataset path and output path in `pre_SXHumans.sh` to match your local environment.

```
cd data_process
bash pre_SXHumans.sh
```

3. Training
```
cd geometry/code

python train_delta.py subject=00016_delta datamodule=XHumans_scan_smplx experiments=XHumans_occ_smplx_delta datamodule.dataloader.dataset_path=/datapath
```

4. Testing
```
python test_delta.py subject=00016_delta experiments=XHumans_occ_smplx_delta demo.motion_path=/datapath/test
```

## SCustomHumans Dataset
1. Download [CustomHumans](https://github.com/custom-humans/editable-humans)

2. Process the data
Modify the dataset path and output path in ```preprocess_SCustomHumans.py```.
```
cd data_process

bash pre_SCHumans.sh
```

3. Training
```
cd geometry/code

python train_delta.py subject=00067_delta datamodule=CustomHumans_scan_smplx experiments=CHumans_occ_smplx_delta datamodule.dataloader.dataset_path=/datapath
```

4. Test
```
python test_delta.py subject=00067_delta experiments=CHumans_occ_smplx_delta demo.motion_path=/datapath/test
```

## 4D-Dress
1. Download [4D-Dress](https://github.com/eth-ait/4d-dress)

2. Process the data

Modify ```DATASET_DIR``` in ```preprocess_4DDress.py``` to set the dataset root directory and specify the output path.
```
cd data_process

bash pre_S4DDress.sh
```

3. Training
```
cd geometry/code

python train_delta.py subject=00122_Inner datamodule=Dress_scan_smplx experiments=Dress_occ_smplx_delta datamodule.dataloader.dataset_path=/datapath
```

4. Testing
```
python test_delta.py subject=00122_Inner experiments=Dress_occ_smplx_delta demo.motion_path=/datapath/test
```

5. Visualizing Results
```
python vis_meshes.py --data_root /home/ps/dy/CtrlAvatar/geometry/outputs/Dress_smplx/00122_Inner/meshes_test --mode_type ply
```