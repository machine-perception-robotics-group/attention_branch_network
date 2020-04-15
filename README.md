# Attention Branch Network

Writer : [Hiroshi Fukui](https://github.com/Hiroshi-Fukui)

Maintainer : [Tsubasa Hirakawa](https://github.com/thirakawa)




## Abstract
This repository contains the source code of Attention Branch Network for image classification. The attention branch network is designed to extend the top-down visual explanation model by introducing a branch structure with an attention mechanism. ABN improves CNN’s performance and visual explanation at the same time by gnerating attention maps in the forward pass.

[[CVF open access](http://openaccess.thecvf.com/content_CVPR_2019/html/Fukui_Attention_Branch_Network_Learning_of_Attention_Mechanism_for_Visual_Explanation_CVPR_2019_paper.html)]
[[arXiv paper](https://arxiv.org/abs/1812.10025)]




## Citation
If you find this repository is useful. Please cite the following references.
```
@article{fukui2018cvpr,
    author = {Hiroshi Fukui and Tsubasa Hirakawa and Takayoshi Yamashita and Hironobu Fujiyoshi},
    title = {Attention Branch Network: Learning of Attention Mechanism for Visual Explanation},
    journal = {Computer Vision and Pattern Recognition},
    year = {2019},
    pages = {10705-10714}
}
```
```
@article{fukui2018arxiv,
    author = {Hiroshi Fukui and Tsubasa Hirakawa and Takayoshi Yamashita and Hironobu Fujiyoshi},
    title = {Attention Branch Network: Learning of Attention Mechanism for Visual Explanation},
    journal = {arXiv preprint arXiv:1812.10025},
    year = {2018}
}  
```




## Enviroment
Our source code is based on [https://github.com/bearpaw/pytorch-classification/](https://github.com/bearpaw/pytorch-classification/) implemented with PyTorch. We are grateful for the author!
Requirements of PyTorch version are as follows:
- PyTorch : 0.4.0
- PyTorch vision : 0.2.1

### Docker
We prepared Docker environments for ABN. You quickly start to use Docker and run scripts.
For more details, please see [docker/README.md](https://github.com/machine-perception-robotics-group/attention_branch_network/blob/master/docker/README.md).




## Execution
Example of run command is as follows:

#### Training
```bash
python3 cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 0,1

python3 imagenet.py -a resnet152 --data ../../dataset/imagenet_data/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet152 --gpu-id 4,5,6,7 --test-batch 100
```

#### Evaluation
```bash
python3 cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 0,1 --evaluate --resume checkpoints/cifar100/resnet-110/model_best.pth.tar

python3 imagenet.py -a resnet152 --data ../../../../dataset/imagenet_data/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet152 --gpu-id 4,5,6 --test-batch 10 --evaluate --resume checkpoints/imagenet/resnet152/model_best.pth.tar
```

If you try the other models or ImageNet, please see [TRAINING.md](https://github.com/machine-perception-robotics-group/attention_branch_network/blob/master/TRAINING.md).




## Trained models
We have published the model files of ABN, which are ResNet family models on CIFAR100 and ImageNet2012 dataset. 

### Models on CIFAR100 Dataset
- [ResNet110](https://drive.google.com/open?id=1Wp7_tIXjq24KSI2VaL9V2N8NRlASLETD)
- [DenseNet](https://drive.google.com/open?id=17ILqWvDJzFFZ603CpeoGaYrt6mhUF-B5)
- [Wide ResNet](https://drive.google.com/open?id=1GRDwdtUV2Q2LhDL0NZyzh5b4pj4CEJtv)
- [ResNeXt](https://drive.google.com/open?id=1CIneC_Y1P_sYEgndC8mR-sAJlS-2eJg5)

### Models on ImageNet2012 Dataset
- [ResNet50](https://drive.google.com/open?id=1SRtzbnE-IpB5talp7PLNK1mzMV3UPQNV)
- [ResNet101](https://drive.google.com/open?id=1B5jBHTfskKAgNpsFm9iADn1lskn2UWyk)
- [ResNet152](https://drive.google.com/open?id=1ZFq0ubZitsuOwPhrQopQOqVW5-KXflFr)




## Performances
### CIFAR100
|  | top-1 error (ABN) | top-1 error ([original](https://github.com/bearpaw/pytorch-classification)) |
|:------------|------------:|------------:|
| ResNet110   |        22.5 |        24.1 |
| DenseNet    |        21.6 |        22.5 |
| Wide ResNet |        18.1 |        18.9 |
| ResNeXt     |        17.7 |        18.3 |

### ImageNet2012
|  | top-1 error (ABN) | top-1 error ([original](https://github.com/bearpaw/pytorch-classification)) |
|:------------|------------:|------------:|
| ResNet50    |        23.1 |        24.1 |
| ResNet101   |        21.8 |        22.5 |
| ResNet152   |        21.4 |        22.2 |

### Examples of attention map
![overview image](./example.jpeg)



