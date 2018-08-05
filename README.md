# Attention Branch Network

## Abstract
このレポジトリは，Attention Branch Networkのオリジナルコードです．Attention Branch Network(ABN)は，Attention機構により画像認識の高精度化と，CNNにおける注視領域の可視化を行うネットワークです．本手法の詳細は，論文を参考にして下さい．


## Detail
本プログラムはCIFAR100, ImageNetでの認識問題を対象にしています．また，ベースにした
プログラムは，[PyTorchのこのコード](https://github.com/bearpaw/pytorch-classification/)をベースに，提案手法を構築しています．使用しているライブラリ等のバージョンは，以下のようになります．また，Dockerファイルを用意しているので，環境構築の際はDockerfileをビルドしてください(Dockerfileは近日中に公開します)．
- PyTorch : 0.4.0
- PyTorch vision


実行コマンドの例は，以下のようになります．また，学習時のコマンドはベースのPyTorchと同様なので，詳細な実行コマンドは[ベースコードのreadme](https://github.com/bearpaw/pytorch-classification/blob/master/TRAINING.md)を参照してください．


また，CIFAR100とImageNetで学習したResNet familyの学習済みモデルも公開しています．(こちらも近日中に公開します)

Models on CIFAR100 Dataset
ResNet110 : http://***
DenseNet : http://***
Wide ResNet : http://***
ResNeXt : http://***

Models on ImageNet2012 Dataset
ResNet50 : http://***
ResNet101 : http://***
ResNet152 : http://***

Performance of the ABN on CIFAR100

|  | top-1 error (ABN) | top-1 error ([original](https://github.com/bearpaw/pytorch-classification)) |
|:------------|------------:|------------:|
| ResNet110   |        22.5 |        24.1 |
| DenseNet    |        21.6 |        22.5 |
| Wide ResNet |        18.1 |        18.9 |
| ResNeXt     |        17.7 |        18.3 |

Performance of the ABN on ImageNet2012

|  | top-1 error (ABN) | top-1 error ([original](https://github.com/bearpaw/pytorch-classification)) |
|:------------|------------:|------------:|
| ResNet50    |        23.1 |        24.1 |
| ResNet101   |        21.8 |        22.5 |
| ResNet152   |        21.4 |        22.2 |
