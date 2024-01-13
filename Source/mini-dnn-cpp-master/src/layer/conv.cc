#include "conv.h"
#include "cuda_utilities.h"
#include "../../config.h"
#include <math.h>
#include <iostream>
#include <fstream>

void printMatrix(Matrix &m, std::string fileName)
{
  std::ofstream myfile;
  myfile.open(fileName);
  myfile << m << std::endl;
  myfile.close();
}

void Conv::init() {
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  grad_bias.resize(channel_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01); 
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv::im2col(const Vector& image, Matrix& data_col) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // im2col
  data_col.resize(hw_out, hw_kernel * channel_in);
  for (int c = 0; c < channel_in; c ++) {
    Vector map = image.block(hw_in * c, 0, hw_in, 1);  // c-th channel map
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          data_col(i, c * hw_kernel + j) = 0;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          data_col(i, c * hw_kernel + j) = map(pick_idx);  // pick which pixel
        }
      }
    }
  }
}

// ORIGINAL FORWARD IMPLEMENTATION
void Conv::forwardOriginal(const Matrix& bottom) {
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  for (int i = 0; i < n_sample; i ++) {
    // im2cols
    Matrix data_col;
    im2col(bottom.col(i), data_col);
    data_cols[i] = data_col;
    // conv by product
    Matrix result = data_col * weight;  // result: (hw_out, channel_out)
    result.rowwise() += bias.transpose();
    top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
  }
}

// SEQUENTIAL FORWARD IMPLEMENTATION
void Conv::forwardVersion_0(const Matrix& bottom) {
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  for (int i = 0; i < n_sample; i ++) {
    // im2col
    Matrix data_col;
    unroll(channel_in, height_in, width_in, height_kernel, bottom.col(i), data_col);
    // Check the dimension of data_col
    data_cols[i] = data_col;
    // conv by product
    Matrix result = data_col * weight;  // result: (channel_out, hw_out)
    result.rowwise() += bias.transpose();
    top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
  }
}

// VERSION 1 OF PARALLEL FORWARD IMPLEMENTATION
void Conv::forwardVersion_1(const Matrix& bottom) {
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  float* dataColData = (float *)malloc(height_kernel * width_kernel * channel_in * height_out * width_out * sizeof(float));
  for (int i = 0; i < n_sample; i ++) {
    // im2col
    float* imageData = (float *)(bottom.col(i)).transpose().data();
    unrollGPUWrapper(channel_in, height_in, width_in, height_kernel, imageData, dataColData);
    Matrix data_col = Eigen::Map<Matrix>(dataColData, height_out * width_out, height_kernel * width_kernel * channel_in);

    data_cols[i] = data_col;
    // conv by product
    Matrix result = data_col * weight;  // result: (hw_out, channel_out)
    result.rowwise() += bias.transpose();
    top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
  }

  free(dataColData);
}

// VERSION 2 OF PARALLEL FORWARD IMPLEMENTATION
void Conv::forwardVersion_2(const Matrix& bottom) {
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);

  float* dataColData = (float *)malloc(height_kernel * width_kernel * channel_in * height_out * width_out * sizeof(float));
  float* h_C = (float*)calloc( (height_out * width_out) * channel_out, sizeof(float));

  for (int i = 0; i < n_sample; i ++) {
    // im2col
    float* imageData = (float *)(bottom.col(i)).transpose().data();
    unrollGPUWrapper(channel_in, height_in, width_in, height_kernel, imageData, dataColData);
    Matrix data_col = Eigen::Map<Matrix>(dataColData, height_out * width_out, height_kernel * width_kernel * channel_in);
    data_cols[i] = data_col;

    // conv by product
    matrixMultiplicationGPUWrapper(data_col.data(), weight.data(), h_C,
                                   height_out * width_out, height_kernel * width_kernel * channel_in, channel_out, i);
    Matrix result = Eigen::Map<Matrix>(h_C, height_out * width_out, channel_out); 
    //result.transposeInPlace(); // result: (hw_out, channel_out)
    //Matrix resultTranpose = result.transpose().eval(); // result: (hw_out, channel_out)

    /* // Print the result
    if (i == 2) {
      Matrix result_2 = data_col * weight;
      printMatrix(result, "result.txt");
      printMatrix(result_2, "result_2.txt");
      printMatrix(data_col, "data_col.txt");
      printMatrix(weight, "weight.txt");
      exit(0);
    } */

    result.rowwise() += bias.transpose();
    //result.transposeInPlace();
    top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
  }

  free(h_C);
  free(dataColData);
}

void Conv::forward(const Matrix& bottom) {
  switch (config::currentVersion)
  {
  case 0:
    Conv::forwardVersion_0(bottom);
    break;
  case 1:
    Conv::forwardVersion_1(bottom);
    break;
  case 2:
    Conv::forwardVersion_2(bottom);
    break;
  
  default:
    Conv::forwardOriginal(bottom);
    break;
  }
}

// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv::unroll(int C, int H, int W, int K, const Vector& image, Matrix& data_col) {
  int c, h, w, p, q, w_base, w_unroll, h_unroll;

  int H_out = H - K + 1; // Only for stride = 1 v
  int W_out = W - K + 1; // Only for stride = 1
  data_col.resize(H_out * W_out, C * K * K);

  for (c = 0; c < C; c++) { // For each input feature map (or each color channel)
    w_base = c * (K * K);
    for (p = 0; p < K; p++)
      for (q = 0; q < K; q++) { // For each input element in a local calculating kernel
        for (h = 0; h < H_out; h++)
          for (w = 0; w < W_out; w++) {
            w_unroll = w_base + p * K + q;
            h_unroll = h * W_out + w;
            data_col(h_unroll, w_unroll) = image(c * H * W + (h + p) * W + (w + q));
          }

      }
  }
}

// col2im, used for grad_bottom
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
// image size: Vector (height_in * width_in * channel_in)
void Conv::col2im(const Matrix& data_col, Vector& image) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // col2im
  image.resize(hw_in * channel_in);
  image.setZero();
  for (int c = 0; c < channel_in; c ++) {
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          continue;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j);  // pick which pixel
        }
      }
    }
  }
}

void Conv::backward(const Matrix& bottom, const Matrix& grad_top) {
  int n_sample = bottom.cols();
  grad_weight.setZero();
  grad_bias.setZero();
  grad_bottom.resize(height_in * width_in * channel_in, n_sample);
  grad_bottom.setZero();
  for (int i = 0; i < n_sample; i ++) {
    // im2col of grad_top
    Matrix grad_top_i = grad_top.col(i);
    Matrix grad_top_i_col = Eigen::Map<Matrix>(grad_top_i.data(),
                              height_out * width_out, channel_out);
    // d(L)/d(w) = \sum{ d(L)/d(z_i) * d(z_i)/d(w) }
    grad_weight += data_cols[i].transpose() * grad_top_i_col;
    // d(L)/d(b) = \sum{ d(L)/d(z_i) * d(z_i)/d(b) }
    grad_bias += grad_top_i_col.colwise().sum().transpose();
    // d(L)/d(x) = \sum{ d(L)/d(z_i) * d(z_i)/d(x) } = d(L)/d(z)_col * w'
    Matrix grad_bottom_i_col = grad_top_i_col * weight.transpose();
    // col2im of grad_bottom
    Vector grad_bottom_i;
    col2im(grad_bottom_i_col, grad_bottom_i);
    grad_bottom.col(i) = grad_bottom_i;
  }
}

void Conv::update(Optimizer& opt) {
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());
  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> Conv::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void Conv::set_parameters(const std::vector<float>& param) {
  if(static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> Conv::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
