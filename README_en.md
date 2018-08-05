Attention Branch Network

Abstract
This repository is contained source code that is original Attention Branch Network.
Our Attention Branch Network(ABN) can improve recognition performance and visualize fixation of Deep Convolutional Neural Network at various single image recognition tasks. Structure of ABN is constructed by three modules that are Feature extractor, Attention branch, and Perception branch. The Feature extractor and Perception branch are made by the baseline network. Attention branch is made with the structure of Class Activation Mapping. Detail of our ABN is written at paper (http://***(will be available))!!


Our source code is based on http://*** with PyTorch. Requirements of PyTorch is as follows, and we published the Docker file and DockerHub. If you need our environment, please build Dockerfile and DockerHub!!
docker pull *****

Example of play command is as follows, but if you want to learn the detail of the construction of play command, please refer the base code at http://****.


Additionally, we have published the model files that are ResNet family with Attention Branch Network on CIFAR100 and ImageNet2012 Dataset. 

Models on CIFAR100 Dataset
ResNet110 : http://***
DenseNet : http://***
Wide ResNet : http://***
ResNeXt : http://***

Models on ImageNet2012 Dataset
ResNet50 : http://***
ResNet101 : http://***
ResNet152 : http://***

Performance of the ABN on CIFAR100 and ImageNet2012

Table




