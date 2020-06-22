# DuDoRNet: Learning a Dual-Domain Recurrent Network for Fast MRI Reconstruction with Deep T1 Prior

Bo Zhou, S. Kevin Zhou

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020

[[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_DuDoRNet_Learning_a_Dual-Domain_Recurrent_Network_for_Fast_MRI_Reconstruction_CVPR_2020_paper.pdf)]

This repository contains the PyTorch implementation of DuDoRNet.

### Citation
If you use this code for your research or project, please cite:

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
    ├── TRAIN                   # contain training files
    │   ├── T1
    │   │   ├── kspace
    │   │   │   ├── train_1.mat         
    │   │   │   ├── train_2.mat 
    │   │   │   ├── ...         
    │   │   │   └── train_N.mat 
    │   │   └── ...
    │   │   
    │   ├── T2
    │   │   ├── kspace
    │   │   │   ├── train_1.mat          
    │   │   │   ├── train_2.mat 
    │   │   │   ├── ...         
    │   │   │   └── train_N.mat 
    │   │   └── ...
    │   │   
    │   ├── FLAIR
    │   │   ├── kspace
    │   │   │   ├── train_1.mat          
    │   │   │   ├── train_2.mat 
    │   │   │   ├── ...         
    │   │   │   └── train_N.mat 
    │   │   └── ...
    │   └── ...
    │
    ├── VALI                    # contain validation files
    │   ├── T1
    │   │   ├── kspace
    │   │   │   ├── vali_1.mat          
    │   │   │   ├── vali_2.mat 
    │   │   │   ├── ...         
    │   │   │   └── vali_M.mat 
    │   │   └── ...
    │   │   
    │   ├── T2
    │   │   ├── kspace
    │   │   │   ├── vali_1.mat          
    │   │   │   ├── vali_2.mat 
    │   │   │   ├── ...         
    │   │   │   └── vali_M.mat 
    │   │   └── ...
    │   │   
    │   ├── FLAIR
    │   │   ├── kspace
    │   │   │   ├── vali_1.mat          
    │   │   │   ├── vali_2.mat 
    │   │   │   ├── ...         
    │   │   │   └── vali_M.mat 
    │   │   └── ...
    │   └── ...
    │
    ├── TEST                    # contain test files
    │   ├── T1
    │   │   ├── kspace
    │   │   │   ├── test_1.mat          
    │   │   │   ├── test_2.mat 
    │   │   │   ├── ...         
    │   │   │   └── test_K.mat 
    │   │   └── ...
    │   │   
    │   ├── T2
    │   │   ├── kspace
    │   │   │   ├── test_1.mat          
    │   │   │   ├── test_2.mat 
    │   │   │   ├── ...         
    │   │   │   └── test_K.mat 
    │   │   └── ...
    │   │   
    │   ├── FLAIR
    │   │   ├── kspace
    │   │   │   ├── test_1.mat          
    │   │   │   ├── test_2.mat 
    │   │   │   ├── ...         
    │   │   │   └── test_K.mat 
    │   │   └── ...
    │   └── ...
    │            
    └── ...

Each .mat should contain a W x W complex value matrix with kspace data in it, where W x W is the kspace size. 
Please note the variable name should be set as 'kspace_py'.
Then, please add the data directory './Data/' after --data_root in the code or scripts.

### To Run Our Code
- Train the model
```bash
python train.py --experiment_name 'train_DuDoRN_R4_pT1' --data_root './Data/' --dataset 'Cartesian' --netG 'DRDN' --n_recurrent 4 --use_prior --protocol_ref 'T1' --protocol_tag 'T2'
```
where \
`--experiment_name` provides the experiment name for the current run, and save all the corresponding results under the experiment_name's folder. \
`--data_root`  provides the data folder directory (with structure illustrated above). \
`--n_recurrent` defines number of recurrent blocks in the DuDoRNet. \
`--protocol_tag` defines target modality to be reconstruct, e.g. T2 or FLAIR. \
`--protocol_ref` defines modality to be used as prior, e.g. T1. \
`--use_prior` defines whether to use prior as indicated by protocol_ref. \
Other hyperparameters can be adjusted in the code as well.

- Test the model
```bash
python test.py --experiment_name 'test_DuDoRN_R4_pT1' --accelerations 5 --resume './outputs/train_DuDoRN_R4_pT1/checkpoints/model_259.pt' --data_root './Data/' --dataset 'Cartesian' --netG 'DRDN' --n_recurrent 4 --use_prior --protocol_ref 'T1' --protocol_tag 'T2'
```
where \
`--accelerations` defines the acceleration factor, e.g. 5 for 5 fold accelerations. \
`--resume` defines which checkpoint for testing and evaluation. \
The test will output an eval.mat containing model's input, reconstruction prediction, and ground-truth for evaluation.

Sample training/test scripts are provided under './scripts/' and can be directly executed.


### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```