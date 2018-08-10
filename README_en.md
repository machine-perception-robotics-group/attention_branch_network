# Attention Branch Network
Write : [Hiroshi Fukui](https://github.com/Hiroshi-Fukui)

## Abstract
This repository is contained source code that is original Attention Branch Network.
Our Attention Branch Network(ABN) can improve recognition performance and visualize fixation of Deep Convolutional Neural Network on various single image recognition tasks. Structure of ABN is constructed by three modules that are Feature extractor, Attention branch, and Perception branch. The Feature extractor and Perception branch are made by the baseline network. Attention branch is made with the structure of Class Activation Mapping. Detail of our ABN is written at paper ([This link is slide at MIRU2018 in Japanese](https://www.slideshare.net/greentea1125/miru2018-global-average-poolingattention-branch-network))!!

## Detail
Our source code is based on [https://github.com/bearpaw/pytorch-classification/](https://github.com/bearpaw/pytorch-classification/) with PyTorch. Requirements of PyTorch is as follows, and we published the Docker file. If you need our environment, please build Dockerfile!!
- PyTorch : 0.4.0
- PyTorch vision

Example of play command is as follows, but if you want to learn the detail of the construction of play command, please refer the base code at [https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md](https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md).

- Training
> python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 1,2,3

- Evaluation
> python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 1,2 --evaluate --resume checkpoints/cifar100/resnet-110/model_best.pth.tar


Additionally, we have published the model files that are ResNet family with Attention Branch Network on CIFAR100 and ImageNet2012 Dataset. 

### Models on CIFAR100 Dataset
- ResNet110 : https://www.dropbox.com/sh/6mkk6e9c4qanznz/AACsvn_52Evk9ONjM5yWW8Yra?dl=0
- DenseNet : https://www.dropbox.com/sh/j572oriksv30g4n/AACoH7pQ8sntnZUr0kU2LNbLa?dl=0
- Wide ResNet : https://www.dropbox.com/sh/ou4nbl4mii54p0e/AACwN41B584l9yUzN7CAX_8ja?dl=0
- ResNeXt : https://www.dropbox.com/sh/ds0uk0ileffad7h/AAC-nILHLRzFEy06ohBkVOVha?dl=0

### Models on ImageNet2012 Dataset

- ResNet50 : https://www.dropbox.com/sh/tuo90s1uqmbk1vd/AAAksM9uPT5u-eViAe-PXIqsa?dl=0
- ResNet101 : https://www.dropbox.com/sh/8vzv7ov59xb5wle/AABaE24vo3Kc-VuSKbOUsZiua?dl=0
- ResNet152 : https://www.dropbox.com/sh/senw3akoud9cten/AACXOBuiNCWq6wPjl4EIVmhHa?dl=0


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


### Visualization of Attention map
![overview image](https://github.com/machine-perception-robotics-group/attention_branch_network/blob/master/example.jpeg)


