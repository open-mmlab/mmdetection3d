# install pytorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# install mim (optional)
pip install openmim
# install mmcv, mmdet, mmengine
pip install -r requirements/mminstall.txt
# install other library
pip install numpy==1.23.5 nuscenes-devkit scipy==1.9.1 setuptools==59.5.0 tensorboardX
# install mmdet3d
python setup.py develop
# install ops
python projects/BEVFusion/setup.py develop
