/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <tuple>
#include <map>
#include <algorithm>
#include <random>

using namespace std;

#include "tiny_dnn/tiny_dnn.h"
// #include "train_labels.cpp"
#include "train_images_gtsrb.cpp"
#include "train_labels_gtsrb.cpp"

// #include "train_images_one.cpp"

;extern std::vector<tiny_dnn::vec_t> train_images_data;
extern std::vector<tiny_dnn::label_t> train_labels_data;
extern std::vector<int> num_list;


static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
                          tiny_dnn::core::backend_t backend_type) {
  using namespace tiny_dnn::layers;

  using tiny_dnn::core::connection_table;
  using padding = tiny_dnn::padding;

  using conv = tiny_dnn::convolutional_layer;
  using fc = tiny_dnn::fully_connected_layer;
  using max_pool = tiny_dnn::max_pooling_layer;
  using batch_norm = tiny_dnn::batch_normalization_layer;
  using dropout = tiny_dnn::dropout_layer;
  using relu = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;
      // DeepThin
    const int n_fmaps1 = 32; // number of feature maps for upper layer
    const int n_fmaps2 = 48; // number of feature maps for lower layer
    const int n_fc = 512;  //number of hidden units in fully-connected layer

    const int input_w = 45;
    const int input_h = 45;
    const int input_c = 1;

    const int num_classes = 43;

    nn << batch_norm(input_w * input_h, input_c)
        << conv(input_w, input_h, 3, 3, input_c, n_fmaps1, tiny_dnn::padding::same, true, 2, 2, 0, 0)  // 3x3 kernel, 2 stride

        << batch_norm(23 * 23, n_fmaps1)
        << relu()
        << conv(23, 23, 3, 3, n_fmaps1, n_fmaps1, tiny_dnn::padding::same)  // 3x3 kernel, 1 stride

        << batch_norm(23 * 23, n_fmaps1)
        << relu()
        << max_pool(23, 23, n_fmaps1, 2, 1, false)
        << conv(22, 22, 3, 3, n_fmaps1, n_fmaps2, tiny_dnn::padding::same, true, 2, 2)  // 3x3 kernel, 2 stride

        << batch_norm(11 * 11, n_fmaps2)
        << relu()
        << conv(11, 11, 3, 3, n_fmaps2, n_fmaps2, tiny_dnn::padding::same)  // 3x3 kernel, 1 stride

        << batch_norm(11 * 11, n_fmaps2)
        << relu()
        << max_pool(11, 11, n_fmaps2, 2, 1, false)
        << fc(10 * 10 * n_fmaps2, n_fc)

        << batch_norm(1 * 1, n_fc)
        << relu()
        << dropout(n_fc, 0.5)
        << fc(n_fc, num_classes)
        << softmax();
}

static void train_lenet(const std::string &data_dir_path,
                        double learning_rate,
                        const int n_train_epochs,
                        const int n_minibatch,
                        tiny_dnn::core::backend_t backend_type) {


  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  // tiny_dnn::adagrad optimizer;
  tiny_dnn::adam optimizer;


  construct_net(nn, backend_type);

  std::cout << "load models..." << std::endl;

  
  // load MNIST dataset
  // extern std::vector<tiny_dnn::vec_t> train_images_data;
  // extern std::vector<tiny_dnn::label_t> train_labels_data;

  std::vector<tiny_dnn::vec_t> train_images;
  std::vector<tiny_dnn::vec_t> test_images;
  std::vector<tiny_dnn::label_t> train_labels;
  std::vector<tiny_dnn::label_t> test_labels;



  std::random_device rd;
  std::mt19937 g(123);
  std::shuffle(num_list.begin(), num_list.end(), g);

  for(int i = 0;i < num_list.size();i++){
    if(i % 5 == 0) {
      test_images.push_back(train_images_data[num_list[i]]);
      test_labels.push_back(train_labels_data[num_list[i]]);
    } else {
      train_images.push_back(train_images_data[num_list[i]]);
      train_labels.push_back(train_labels_data[num_list[i]]);
    }
  }


  std::cout << "start training" << std::endl;

  
  std::cout << "train_images size : " << train_images.size() << std::endl;
  std::cout << "train_labels size : " << train_labels.size() << std::endl;
  std::cout << "test_images size : " << test_images.size() << std::endl;
  std::cout << "test_labels size : " << test_labels.size() << std::endl;


  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  // optimizer.alpha *=
  //   std::min(tiny_dnn::float_t(4),
  //            static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

  optimizer.alpha *=
    static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;
    ++epoch;
    tiny_dnn::result res = nn.test(test_images, test_labels);
    std::cout << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.fit<tiny_dnn::cross_entropy_multiclass>(optimizer, train_images, train_labels, n_minibatch,
                          n_train_epochs, on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);


  // save network model & trained weights
  nn.save("LeNet-model");
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {{
    "internal", "nnpack", "libdnn", "avx", "opencl",
  }};
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<tiny_dnn::core::backend_t>(i);
    }
  }
  return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
            << " --learning_rate 1"
            << " --epochs 30"
            << " --minibatch_size 16"
            << " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
  double learning_rate                   = 1;
  int epochs                             = 30;
  std::string data_path                  = "";
  int minibatch_size                     = 16;
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();


  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }
  if (data_path == "") {
    std::cerr << "Data path not specified." << std::endl;
    usage(argv[0]);
    return -1;
  }
  if (learning_rate <= 0) {
    std::cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << std::endl;
    return -1;
  }
  if (epochs <= 0) {
    std::cerr << "Invalid number of epochs. The number of epochs must be "
                 "greater than 0."
              << std::endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 60000) {
    std::cerr
      << "Invalid minibatch size. The minibatch size must be greater than 0"
         " and less than dataset size (60000)."
      << std::endl;
    return -1;
  }
  std::cout << "Running with the following parameters:" << std::endl
            << "Data path: " << data_path << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try {
    train_lenet(data_path, learning_rate, epochs, minibatch_size, backend_type);
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }

  return 0;
}
