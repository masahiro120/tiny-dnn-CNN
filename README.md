# Tiny DNNを使用したMNISTサンプルのコンパイルと実行手順

## 1. ビルド用ディレクトリの作成

```bash
cd tiny_dnn
mkdir build
```

## 2. BUILD_EXAMPLESをONにしてコンフィグする

```bash
cd build
cmake .. -DBUILD_EXAMPLES=ON
```

## 3. MNISTサンプルをコンパイルする

```bash
make example_mnist_train
```

## 4. MNISTサンプルを実行する

```bash
cd example
./example_mnist_train --data_path ../../data
```

## オプション引数での実行

引数を付けることでエポック数などをデフォルト値から変更して実行できます。

### 例: エポック数=1, ミニバッチサイズ=32, バックエンドタイプ=インターナルで実行

```bash
./example_mnist_train --data_path ../../data --epochs 1 --minibatch_size 32 --backend_type internal
```
