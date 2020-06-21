# DuDoRNet: Learning a Dual-Domain Recurrent Network for Fast MRI Reconstruction with Deep T1 Prior

Bo Zhou, S. Kevin Zhou

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020

[[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_DuDoRNet_Learning_a_Dual-Domain_Recurrent_Network_for_Fast_MRI_Reconstruction_CVPR_2020_paper.pdf)]

This repository contains the PyTorch implementation of DuDoRNet.

### Citation
If you use this code for your research, please consider citing:

	@inproceedings{zhou2020dudornet,
	  title={DuDoRNet: Learning a Dual-Domain Recurrent Network for Fast MRI Reconstruction with Deep T1 Prior},
	  author={Zhou, Bo and Zhou, S Kevin},
	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	  pages={4273--4282},
	  year={2020}
	}


### Environment and Dependencies
Requirements:
* Python 3.7
* Pytorch 0.4.1
* scipy
* scikit-image
* opencv-python
* tqdm

Our code has been tested with Python 3.7, Pytorch 0.4.1, CUDA 10.0 on Ubuntu 18.04.


### Dataset Setup
    .
    Data
    ├── TRAIN                   # contain training .mat files
    │   ├── T1
    │   │   ├── train_1.mat          
    │   │   ├── train_2.mat 
    │   │   ├── ...         
    │   │   └── train_N.mat 
    │   ├── T2
    │   │   ├── train_1.mat          
    │   │   ├── train_2.mat 
    │   │   ├── ...         
    │   │   └── train_N.mat 
    │   ├── FLAIR
    │   │   ├── train_1.mat          
    │   │   ├── train_2.mat 
    │   │   ├── ...         
    │   │   └── train_N.mat
    │   └── ...
    │
    ├── VALI                    # contain validation .mat files
    │   ├── T1
    │   │   ├── vali_1.mat          
    │   │   ├── vali_2.mat 
    │   │   ├── ...         
    │   │   └── vali_M.mat 
    │   ├── T2
    │   │   ├── vali_1.mat          
    │   │   ├── vali_2.mat 
    │   │   ├── ...         
    │   │   └── vali_M.mat 
    │   ├── FLAIR
    │   │   ├── vali_1.mat          
    │   │   ├── vali_2.mat 
    │   │   ├── ...         
    │   │   └── vali_M.mat
    │   └── ...
    │
    ├── TEST                    # contain test .mat files
    │   ├── T1
    │   │   ├── test_1.mat          
    │   │   ├── test_2.mat 
    │   │   ├── ...         
    │   │   └── test_K.mat 
    │   ├── T2
    │   │   ├── test_1.mat          
    │   │   ├── test_2.mat 
    │   │   ├── ...         
    │   │   └── test_K.mat 
    │   ├── FLAIR
    │   │   ├── test_1.mat          
    │   │   ├── test_2.mat 
    │   │   ├── ...         
    │   │   └── test_K.mat 
    │   └── ...  
    │            
    └── ...

Each .mat should contain a W x W complex value kspace matrix with Variable name: 'kspace_py', where W x W is the kspace size.


### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```