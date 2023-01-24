### 1. ビルド用ディレクトリの作成
$ cd tiny_dnn
$ mkdir build
 
### 2. BUILD_EXAMPLESをONにしてコンフィグする
$ cd build
$ cmake .. -DBUILD_EXAMPLES=ON
 
### 3. MNISTサンプルをコンパイルする
$ make example_mnist_train
 
### 4. MNISTサンプルを実行する
$ cd example
$ ./example_mnist_train --data_path ../../data
 
## 引数を付けることでエポック数などをデフォルト値から変更して実行できる
# ex). エポック数=1, ミニバッチサイズ=32, バックエンドタイプ=インターナルで実行
$ ./example_mnist_train --data_path ../../data --epochs 1 --minibatch_size 32 --backend_type internal