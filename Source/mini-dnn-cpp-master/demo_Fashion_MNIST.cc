#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

#include "src/layer/cuda_utilities.h"
#include <thread>
#include <vector>
#include <numeric>

const bool IS_TRAINING = false;

void saveNetworkParameters(Network& network, std::string filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    std::vector<std::vector<float>> parameters = network.get_parameters();

    for (const auto &v: parameters) {
      for (const auto &f: v) {
        file << f << " ";
      }
      file << "\n";
    }

  }
}

void loadNetworkParameters(Network& network, std::string filename) {
  std::ifstream file(filename);
  if (file.is_open()) {
    std::vector<std::vector<float>> parameters = network.get_parameters();

    for (auto &v: parameters) {
      for (auto &f: v) {
        file >> f;
      }
    }

    network.set_parameters(parameters);
    std::cout << "Parameters loaded" << std::endl;
    saveNetworkParameters(network, "../../Model/parameters_check_2.txt");
  }
}

void testing(Network& dnn, MNIST& dataset, int epoch) {
  startTimer();    
  dnn.forward(dataset.test_data);
  std::cout << "Test time: " << stopTimer() << std::endl;
  
  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout << std::endl;
  std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
  std::cout << std::endl;

  saveNetworkParameters(dnn, "../../Model/parameters_check.txt");
}

int main() {
  // data
  MNIST dataset("../data/fashion-mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // file to save parameters
  std::string filename = "../../Model/parameters.txt";
  
  // dnn
  Network dnn;
  Layer* C1 = new Conv(1, 28, 28, 6, 5, 5, 1);
  Layer* S2 = new MaxPooling(6, 24, 24, 2, 2, 2);
  Layer* C3 = new Conv(6, 12, 12, 16, 5, 5, 1);
  Layer* S4 = new MaxPooling(16, 8, 8, 2, 2, 2);
  Layer* F6 = new FullyConnected(S4->output_dim(), 120);
  Layer* F7 = new FullyConnected(120, 84);
  Layer* F8 = new FullyConnected(84, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* relu3 = new ReLU;
  Layer* relu4 = new ReLU;
  Layer* softmax = new Softmax;
  
  dnn.add_layer(C1);
  dnn.add_layer(relu1);
  dnn.add_layer(S2);
  dnn.add_layer(C3);
  dnn.add_layer(relu2);
  dnn.add_layer(S4);
  dnn.add_layer(F6);
  dnn.add_layer(relu3);
  dnn.add_layer(F7);
  dnn.add_layer(relu4);
  dnn.add_layer(F8);
  dnn.add_layer(softmax);
  
  // loss
  Loss* loss = new CrossEntropy;
  dnn.add_loss(loss);
  // train & test
  SGD opt(0.001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 5;
  const int batch_size = 128;

  if (!IS_TRAINING) {
    loadNetworkParameters(dnn, filename);
    testing(dnn, dataset, 0);

    return 0;
  }

  for (int epoch = 0; epoch < n_epoch; epoch ++) {
    shuffle_data(dataset.train_data, dataset.train_labels);
    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      int ith_batch = start_idx / batch_size;
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                    std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                    std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      if (false && ith_batch % 10 == 1) {
        std::cout << ith_batch << "-th grad: " << std::endl;
        dnn.check_gradient(x_batch, target_batch, 10);
      }

      dnn.forward(x_batch);

      dnn.backward(x_batch, target_batch);
      // display
      if (ith_batch % 50 == 0) {
        std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss() << std::endl;
      }
      // optimize
      dnn.update(opt);
      saveNetworkParameters(dnn, filename);
    }
    // test
    testing(dnn, dataset, epoch);
  }
  return 0;
}

