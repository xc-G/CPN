# A Pytorch Implementation of Learning continuous piecewise non-linear activation functions for deep neural networks (ICME 2023)

## Requirements
pip install -r requirements.txt

## Usage
* Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
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
