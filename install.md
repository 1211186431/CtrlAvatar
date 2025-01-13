# Install
## Environment setup
1. Clone this repo:
```
git clone https://github.com/1211186431/CtrlAvatar.git
```

2. Install conda,[pytorch3d](https://github.com/facebookresearch/pytorch3d), [kaolin](https://github.com/NVIDIAGameWorks/kaolin) other required packages

```
conda create --name CtrlAvatar python=3.9
conda activate CtrlAvatar

# install pyorch https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# install pytorch3d https://anaconda.org/pytorch3d/pytorch3d/files?page=4
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py39_cu117_pyt1130.tar.bz2
conda install pytorch3d-0.7.4-py39_cu117_pyt1130.tar.bz2

# install kaolin 
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html

# install other
cd CtrlAvatar
pip install -r requirements.txt 
pip install pytorch-lightning==1.3.3
 
# (optional)install nvdiffrast
pip install ninja
pip install git+https://github.com/NVlabs/nvdiffrast
```

3. build code
```
cd geometry

python setup.py install
```

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