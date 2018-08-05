# Attention Branch Network
Write : [Hiroshi Fukui](https://github.com/Hiroshi-Fukui)

## Abstract
This repository is contained source code that is original Attention Branch Network.
Our Attention Branch Network(ABN) can improve recognition performance and visualize fixation of Deep Convolutional Neural Network on various single image recognition tasks. Structure of ABN is constructed by three modules that are Feature extractor, Attention branch, and Perception branch. The Feature extractor and Perception branch are made by the baseline network. Attention branch is made with the structure of Class Activation Mapping. Detail of our ABN is written at paper ([This link is poster at MIRU2018 in Japanese](https://drive.google.com/file/d/11uMkpMgb1vtcG78cDDwfwC-fowkdrqVU/view?usp=sharing))!!

## Detail
Our source code is based on [https://github.com/bearpaw/pytorch-classification/](https://github.com/bearpaw/pytorch-classification/) with PyTorch. Requirements of PyTorch is as follows, and we published the Docker file. If you need our environment, please build Dockerfile!!
- PyTorch : 0.4.0
- PyTorch vision

Example of play command is as follows, but if you want to learn the detail of the construction of play command, please refer the base code at [https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md](https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md).

- Training
> python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 1,2,3

- Evaluation
> python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 1,2 --evaluate --resume checkpoints/cifar100/resnet-110/model_best.pth.tar


Additionally, we have published the model files that are ResNet family with Attention Branch Network on CIFAR100 and ImageNet2012 Dataset. (Will be available)

### Models on CIFAR100 Dataset
- ResNet110 : http://***
- DenseNet : http://***
- Wide ResNet : http://***
- ResNeXt : http://***

### Models on ImageNet2012 Dataset
- ResNet50 : http://***
- ResNet101 : http://***
- ResNet152 : http://***

### Performance of the ABN on CIFAR100

|  | top-1 error (ABN) | top-1 error ([original](https://github.com/bearpaw/pytorch-classification)) |
|:------------|------------:|------------:|
| ResNet110   |        22.5 |        24.1 |
| DenseNet    |        21.6 |        22.5 |
| Wide ResNet |        18.1 |        18.9 |
| ResNeXt     |        17.7 |        18.3 |


### Performance of the ABN on ImageNet2012

|  | top-1 error (ABN) | top-1 error ([original](https://github.com/bearpaw/pytorch-classification)) |
|:------------|------------:|------------:|
| ResNet50    |        23.1 |        24.1 |
| ResNet101   |        21.8 |        22.5 |
| ResNet152   |        21.4 |        22.2 |




