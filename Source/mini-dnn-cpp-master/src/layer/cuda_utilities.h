void startTimer();
float stopTimer();
int test();

void unrollGPUWrapper(int C, int H, int W, int K, float* image, float* data_col);