# Attention Branch Network
Write : [Hiroshi Fukui](https://github.com/Hiroshi-Fukui)

[english ver.](https://github.com/machine-perception-robotics-group/attention_branch_network/blob/master/README_en.md)

## Abstract
このレポジトリは，Attention Branch Networkのオリジナルコードです．Attention Branch Network(ABN)は，Attention機構により画像認識の高精度化と，CNNにおける注視領域の可視化を行うネットワークです．本手法の詳細は，論文([現在はMIRU時のスライドになってます](https://www.slideshare.net/greentea1125/miru2018-global-average-poolingattention-branch-network))を参考にして下さい．


## Detail
本プログラムはCIFAR100, ImageNetでの認識問題を対象にしています．また，ベースにした
プログラムは，[PyTorchのこのコード](https://github.com/bearpaw/pytorch-classification/)をベースに，提案手法を構築しています．使用しているライブラリ等のバージョンは，以下のようになります．また，Dockerファイルを用意しているので，環境構築の際はDockerfileをビルドしてください．
- PyTorch : 0.4.0
- PyTorch vision


実行コマンドの例は，以下のようになります．また，学習時のコマンドはベースのPyTorchと同様なので，詳細な実行コマンドは[ベースコードのreadme](https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md)を参照してください．

- 学習
> python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 1,2,3

- 評価
> python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id 1,2 --evaluate --resume checkpoints/cifar100/resnet-110/model_best.pth.tar

### Models on CIFAR100 Dataset
また，CIFAR100とImageNetで学習したResNet familyの学習済みモデルも公開しています．

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
