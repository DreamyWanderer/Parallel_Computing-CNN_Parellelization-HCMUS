#pragma once

void startTimer();
float stopTimer();

void unrollGPUWrapper(int C, int H, int W, int K, float* image, float* data_col);
void matrixMultiplicationGPUWrapper(float* A, float *B, float *C, int m, int n, int k, int i);