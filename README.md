# Attention Branch Network

Abstract
このレポジトリは，Attention Branch Networkのオリジナルコードです．Attention Branch Network(ABN)は，Attention機構により画像認識の高精度化と，CNNにおける注視領域の可視化を行うネットワークです．本手法の詳細は，論文を参考にして下さい．

本プログラムはCIFAR100, ImageNetでの認識問題を対象にしています．また，ベースにした
プログラムは，PyTorchのこのコードをベースに，提案手法を構築しています．使用しているライブラリ等のバージョンは，以下のようになります．また，Dockerファイルを用意しているので，環境構築の際はDockerfileをビルドしてください．


実行コマンドの例は，以下のようになります．また，学習時のコマンドはベースのPyTorchと同様なので，詳細な実行コマンドはベースコードのreadmeを参照してください．


また，CIFAR100とImageNetで学習したResNet familyの学習済みモデルも公開しています．

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