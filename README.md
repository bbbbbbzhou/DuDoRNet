# DuDoRNet
This repository contains the PyTorch implementation for the 2020 CVPR paper:</br>

[DuDoRnet: Learning a Dual-Domain Recurrent Network for Fast MRI Reconstruction with Deep T1 Prior](https://arxiv.org/pdf/2001.03799.pdf)</br>

[Bo Zhou](https://scholar.google.com/citations?user=94Rsf5wAAAAJ&hl=en), [S. Kevin Zhou](https://scholar.google.com/citations?user=8eNm2GMAAAAJ&hl=en)

### Citation
If you use this code for your research, please consider citing:

	@article{zhou2020dudornet,
	  title={DuDoRNet: Learning a Dual-Domain Recurrent Network for Fast MRI Reconstruction with Deep T1 Prior},
	  author={Zhou, Bo and Zhou, S Kevin},
	  journal={arXiv preprint arXiv:2001.03799},
	  year={2020}
	}


### Environment and Dependencies
Requirements:
* Python2.7+ with Numpy and scikit-image
* [Tensorflow (version 1.0+)](https://www.tensorflow.org/install/)
* [TFLearn](http://tflearn.org/installation/)

Our code has been tested with Python 2.7, **TensorFlow 1.3.0**, TFLearn 0.3.2, CUDA 8.0 on Ubuntu 14.04.


### Dataset Setup
We use [ShapeNet](https://www.shapenet.org/) for model training and evaluation. The official tensorflow implementation provides a subset of ShapeNet for it, you can download it [here](https://drive.google.com/drive/folders/131dH36qXCabym1JjSmEpSQZg4dmZVQid). Extract it and link it to `data_tf` directory as follows. Before that, some meta files [here](https://drive.google.com/file/d/16d9druvCpsjKWsxHmsTD5HSOWiCWtDzo/view?usp=sharing) will help you establish the folder tree, demonstrated as follows.

*P.S. In case more data is needed, another larger data package of ShapeNet is also [available](https://drive.google.com/file/d/1Z8gt4HdPujBNFABYrthhau9VZW10WWYe/view). You can extract it and place it in the `data` directory. But this would take much time and needs about 300GB storage.*

```
datasets/data
├── ellipsoid
│   ├── face1.obj
│   ├── face2.obj
│   ├── face3.obj
│   └── info_ellipsoid.dat
├── pretrained
│   ... (.pth files)
└── shapenet
    ├── data (larger data package, optional)
    │   ├── 02691156
    │   │   └── 3a123ae34379ea6871a70be9f12ce8b0_02.dat
    │   ├── 02828884
    │   └── ...
    ├── data_tf (standard data used in official implementation)
    │   ├── 02691156 (put the folders directly in data_tf)
    │   │   └── 10115655850468db78d106ce0a280f87
    │   ├── 02828884
    │   └── ...
    └── meta
        ...
```


### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```