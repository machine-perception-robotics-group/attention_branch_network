# Attention Branch Network
Writer : [Hiroshi Fukui](https://github.com/Hiroshi-Fukui)

## Abstract
This repository contains the source code of Attention Branch Network for image classification. The attention branch network is designed to extend the top-down visual explanation model by introducing a branch structure with an attention mechanism. ABN improves CNN’s performance and visual explanation at the same time by gnerating attention maps in the forward pass. [This link is the arXiv paper.](https://arxiv.org/abs/1812.10025)

## Citation

@article{fukui2018,  
	author = {Hiroshi Fukui and Tsubasa Hirakawa and Takayoshi Yamashita and Hironobu Fujiyoshi},  
	title = {Attention Branch Network: Learning of Attention Mechanism for Visual Explanation},  
	journal = {arXiv preprint arXiv:1812.10025},  
	year = {2018}  
}  


## Detail
Our source code is based on [https://github.com/bearpaw/pytorch-classification/](https://github.com/bearpaw/pytorch-classification/) with PyTorch. Requirements of PyTorch version is as follows, and we published the [Docker file](https://www.dropbox.com/sh/evn9792hoi75yix/AAC1xMNxKw6Qkus6VCzxrhfVa?dl=0). If you need the Docker, please use our Dockerfile.
- PyTorch : 0.4.0
- PyTorch vision : 0.2.1

Example of run command is as follows, but if you try the other models or ImageNet, please refer the README at [https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md](https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md).

- Training
> python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 0,1

> python imagenet.py -a resnet152 --data ../../../../dataset/imagenet_data/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet152 --gpu-id 0,1,2,3 --test-batch 100

- Evaluation
> python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 0,1 --evaluate --resume checkpoints/cifar100/resnet-110/model_best.pth.tar

> python imagenet.py -a resnet152 --data ../../../../dataset/imagenet_data/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet152 --gpu-id 4,5,6 --test-batch 10 --evaluate --resume checkpoints/imagenet/resnet152/model_best.pth.tar


Additionally, we have published the model files of ABN, which are ResNet family models on CIFAR100 and ImageNet2012 dataset. 

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


### Example of Attention map
![overview image](https://github.com/machine-perception-robotics-group/attention_branch_network/blob/master/example.jpeg)


