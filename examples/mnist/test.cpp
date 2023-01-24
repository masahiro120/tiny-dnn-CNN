/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <vector>
#include "tiny_dnn/tiny_dnn.h"
#include <list>
using namespace std;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a(1);
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string &imagefilename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   tiny_dnn::vec_t &data) {
  tiny_dnn::image<> img(imagefilename, tiny_dnn::image_type::grayscale);
  tiny_dnn::image<> resized = resize_image(img, w, h);

  // mnist dataset is "white on black", so negate required
  std::transform(
    resized.begin(), resized.end(), std::back_inserter(data),
    [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}


void recognize(const std::string &dictionary, const std::string &src_filename) {
  tiny_dnn::network<tiny_dnn::sequential> nn;

  nn.load(dictionary);

  // convert imagefile to vec_t
  tiny_dnn::vec_t data;
  convert_image(src_filename, -1.0, 1.0, 32, 32, data);

  // recognize
  auto res = nn.predict(data);

  std::vector<std::pair<double, int>> scores;

  // sort & print top-3
  for (int i = 0; i < 10; i++)
    scores.emplace_back(rescale<tiny_dnn::tanh_layer>(res[i]), i);

  sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

  for (int i = 0; i < 5; i++)
    std::cout << scores[i].second << "," << scores[i].first << std::endl;

  // std::cout << typeid(nn).name() << std::endl;

  // save outputs of each layer
  for (size_t i = 0; i < nn.depth(); i++) {
    auto out_img  = nn[i]->output_to_image();
    auto filename = "layer_" + std::to_string(i) + ".png";
    out_img.save(filename);
  }
  // save filter shape of first convolutional layer
  {
    auto weight   = nn.at<tiny_dnn::convolutional_layer>(0).weight_to_image();
    auto filename = "weights.png";
    weight.save(filename);
  }

  // for(int i = 0; i < 12 ;i++){
  //   cout << endl;
  //   cout << "nn[" << i << "] type : " << typeid(nn[i]).name() << endl;
  //   cout << "nn[" << i << "]->weights type : " << typeid(nn[i]->weights()).name() << endl;
  //   cout << "nn[" << i << "]->weights size : " << (nn[i]->weights()).size() << endl;
  //   cout << "nn[" << i << "]->weights[0] type : " << typeid(*(nn[i]->weights())[0]).name() << endl;
  //   cout << "nn[" << i << "]->weights[0][0] type : " << typeid((nn[i]->weights())[0][0]).name() << endl;
  //   cout << "nn[" << i << "]->weights[0][0][0] type : " << typeid((nn[i]->weights())[0][0][0]).name() << endl << endl;
    

  //   auto weights = nn[i]->weights();
  //   std::cout << "I = " << i << " weights Size : " << weights.size() << std::endl;
  //   std::cout << "I = " << i << " weights type : " << typeid(weights).name() << std::endl;
  //   std::cout << "I = " << i << " weights[0] type : " << typeid(weights[0]).name() << std::endl;
  //   std::cout << "I = " << i << " weights[0][0] type : " << typeid(weights[0][0]).name() << std::endl;
  //   std::cout << "I = " << i << " weights[0][0][0] type : " << typeid(weights[0][0][0]).name() << std::endl;
  //   // std::cout << "weights[2][0][0] : " << weights[2][0][0] << std::endl;
  //   i++;
  // }

  vector<vector<vector<float, tiny_dnn::aligned_allocator<float, 64>>>> w_all;

  for(int i = 0;i < nn.depth();i++){
    cout << endl << "Layer : " << i << endl;
    vector<vector<float, tiny_dnn::aligned_allocator<float, 64>>> w;
    auto weights = nn[i]->weights();
    cout << "weights : " << typeid(weights).name() << endl;
    cout << "weights.size : " << weights.size() << endl;
    if(weights.size() != 0){
      cout << "weights[0] : " << typeid(*weights[0]).name() << endl;
      cout << "weights[0].size : " << weights[0]->size() << endl;
      cout << "weights[1].size : " << weights[1]->size() << endl;
      cout << "weights[0][0] : " << typeid(weights[0][0]).name() << endl;
      cout << "weights[0][0].size : " << weights[0][0].size() << endl;
      cout << "weights[1][0][0] : " << weights[1][0][0] << endl;

      auto d = nn[i]->prev()[0]->weight_update();
      cout << "d : " << typeid(d).name() << endl;
      cout << "d.size : " << d.size() << endl;
      cout << "d[0].size : " << d[0].size() << endl;
      cout << "d[0][0] : " << d[0][0] << endl;

      auto w = nn[i]->prev()[1]->weight_update();
      cout << "w : " << typeid(w).name() << endl;
      cout << "w.size : " << w.size() << endl;
      cout << "w[0].size : " << w[0].size() << endl;
      cout << "w[0][0] : " << w[0][0] << endl;

      auto b = nn[i]->prev()[2]->weight_update();
      cout << "b : " << typeid(b).name() << endl;
      cout << "b.size : " << b.size() << endl;
      cout << "b[0].size : " << b[0].size() << endl;
      cout << "b[0][0] : " << b[0][0] << endl << endl;
    }

    // wb = {w, b}
    // tiny.cc
    // this->model[i].set_wb(w, b)

    // layer.h
    // set_w(wb){
    //   prev()[1]->update(w)
    //   prev()[2]->update(b)
    // }

    // node.h
    // update(v){
    //   data_ = v;
    // }


    // for(auto x : *weights[0]){
    //   cout << x << endl;

    // }

    // weights = [ 
    //   [1,2,3,4,5,6,...]重み
    //   [1,2,3,4,...]バイアス
    // ]
    
    // if(weights.size() != 0){
    //   w.push_back(*weights[0]);
    //   w.push_back(*weights[1]);
    // }

    // w_all.push_back(w);
  }

  cout << typeid(w_all).name() << endl;
  cout << w_all.size() << endl;
  cout << typeid(w_all[0]).name() << endl;
  cout << w_all[0].size() << endl;
  cout << typeid(w_all[0][0]).name() << endl;
  cout << w_all[0][0].size() << endl;
  cout << typeid(w_all[0][0][0]).name() << endl;
  // cout << w_all[0][0][0].size() << endl;
  
  cout << w_all[0][0][0] << endl;

  // list<list<list<float>>> l(w_all.begin(), w_all.end());
  
  cout << typeid(w_all).name() << endl;
  cout << "w_all : " << w_all.size() << endl << endl;
  for(int i = 0;i < w_all.size();i++){
    cout << "w_all[" << i << "] : " << w_all[i].size() << endl;
    if(w_all[i].size() != 0){
      for(int j = 0;j < w_all[i].size();j++){
        cout << "w_all[" << i << "][" << j << "] : " << w_all[i][j].size() << endl;

      }
    }
    cout << endl;
  }

  // for(auto ele : w_all[0][0])
  //   cout << ele << " ";

  cout << endl;
  cout << endl;

  cout << *w_all[0][0].begin() << endl;
  cout << endl;
  cout << endl;


  // std::list<std::list<std::list<float>>> list3d;
  // for (const auto& plane : w_all) {
  //   std::list<std::list<float>> list2d;
  //   for (const auto& row : plane) {
  //     list2d.push_back(std::list<float>(row.begin(), row.end()));
  //   }
  //   list3d.push_back(list2d);
  // }

  std::list<std::list<std::list<std::string>>> list3d;
  for (const auto& plane : w_all) {
    std::list<std::list<std::string>> list2d;
    for (const auto& row : plane) {
      std::list<std::string> list1d;
      for (const auto& element : row) {
        std::stringstream ss;
        ss << element;
        list1d.push_back(ss.str());
      }
      list2d.push_back(list1d);
    }
    list3d.push_back(list2d);
  }

  auto plane_iter = list3d.begin();
  auto row_iter = plane_iter->begin();
  auto element_iter = row_iter->begin();
  std::cout << *element_iter << std::endl;
  cout << typeid(*element_iter).name() << endl; 
  cout << endl;

  // for (const auto& plane : list3d) {
  //   cout << "{";
  //   for (const auto& row : plane) {
  //     cout << "{";
  //     for (const auto& element : row) {
  //       std::cout << "{" << element << "} ";
  //     }
  //     std::cout << "}" <<  std::endl;
  //   }
  //   std::cout << "}" <<  std::endl;
  // }



  vector<vector<vector<float, tiny_dnn::aligned_allocator<float, 64>>>> w_vec;
  for (const auto& plane : list3d) {
      vector<vector<float, tiny_dnn::aligned_allocator<float, 64>>> vector2d;
      for (const auto& row : plane) {
        vector<float, tiny_dnn::aligned_allocator<float, 64>> vector1d;
        for (const auto& element : row) {
          std::stringstream ss(element);
          float value;
          ss >> value;
          vector1d.push_back(value);
        }
        vector2d.push_back(vector1d);
      }
      w_vec.push_back(vector2d);
  }

  for (int j = 0; j < w_vec[0].size(); ++j) {
    for (int k = 0; k < w_vec[0][j].size(); ++k) {
      w_vec[0][j][k] = 0.0;
    }
  }

  cout << w_vec[0][0][0] << endl;

  cout << typeid(w_vec).name() << endl;
  cout << w_vec.size() << endl;

  cout << typeid(nn[0]->weights()).name() << endl;
  cout << typeid(w_vec[0]).name() << endl;
  cout << typeid(&w_vec[0]).name() << endl;
  // cout << typeid(*w_vec[0]).name() << endl;
  
  // nn[0]->weights() = w_vec[0];
  for (int i = 0; i < w_vec[0].size(); ++i) {
    std::copy(w_vec[0][i].begin(), w_vec[0][i].end(), (nn[0]->weights())[i]->begin());
  }

}

int main(int argc, char **argv) {
  std::cout << *argv << std::endl;
  if (argc != 2) {
    std::cout << "please specify image file" << std::endl;
    return 0;
  }
  recognize("LeNet-model", argv[1]);
}
