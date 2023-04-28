# A Pytorch Implementation of Learning continuous piecewise non-linear activation functions for deep neural networks (ICME 2023)

## Requirements
pip install -r requirements.txt

## Introduction

In this paper, we propose a learnable continuous piece-wise nonlinear activation function (or CPN in short), which improves the widely used ReLU from three directions, i.e., finer pieces, non-linear terms and learnable parameterization. CPN is a continuous activation function with multiple pieces and incorporates non-linear terms in every interval. 

* The main results on image classification and super-resolution are as follows:

![image-20230428131758536](image\image-20230428131758536.png)

## Usage

### Classification

For image classification, we take the MobileNetV2_0.25 (0.35) with the width multiplier of 0.25 (0.35) as the baseline mode

* Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
* To use the MTLU, you can refer to the paper:Fast Image Restoration with Multi-bin Trainable Linear Units and the codebase: https://github.com/ShuhangGu/MTLU_ICCV2019.
*  Train the models with the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=23445 --use_env train_mobilenet.py \
--data-set IMNET \
--data-path [path to imagenet dataset]
--dist-eval \
--output [path to output] \
--batch-size 128 \
--model mobilenet \
--lr 0.1
```

* Test the models with the following command:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=23445 --use_env train_mobilenet.py \
--data-set IMNET \
--data-path [path to imagenet dataset]
--dist-eval \
--output [path to output] \
--resume [path to checkpoint]
--batch-size 128 \
--model mobilenet \
--lr 0.1
--eval
```

* checkpoint: We  provide the checkpoints at [classification](https://drive.google.com/drive/folders/1l5MEmNKSrYaYUa35KrOFwAZm0hahipw8?usp=share_link)

### Single-image super resolution

For single-image super resolution, The EDSR baseline single-scale model with 64 feature maps in each layer and 16 basic blocks is taken as our baseline model.

* We mainly refer to the codebase [EDSR-Pytorch](https://github.com/sanghyun-son/EDSR-PyTorch). The aporach to obtaining the DIV2K dataset and benchmark datasets can be found at this repository.
* Train the $\times 2$ model with the following command：

```python
CUDA_VISIBLE_DEVICES=0 python main.py \
--model EDSR \
--scale 2 \
--patch_size 96 \
--save [path to output] \
--reset
```

* Train the $\times 3$ and $\times4$ model from the $\times2$ model:

```python
CUDA_VISIBLE_DEVICES=0 python main.py \
--model EDSR \
--scale [3 or 4] \
--patch_size [144 or 192] \
--save [path to output] \
--reset \
--pre_train [path to pretrained ×2 model]
```

* Test  the models with the following command:

```python
CUDA_VISIBLE_DEVICES=0 python main.py \
--data_test Set5+Set14+B100+Urban100+DIV2K \
--data_range 801-900 \
--scale [2 or 3 or 4] \
--pre_train [path to checkpoint] \
--test_only
```

