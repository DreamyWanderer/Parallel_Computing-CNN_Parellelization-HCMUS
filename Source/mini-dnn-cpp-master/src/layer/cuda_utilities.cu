#include <iostream>
#include "cuda_utilities.h"

#define CHECK(call)                                                \
	{                                                              \
		const cudaError_t error = call;                            \
		if (error != cudaSuccess)                                  \
		{                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
			fprintf(stderr, "code: %d, reason: %s\n", error,       \
					cudaGetErrorString(error));                    \
			exit(EXIT_FAILURE);                                    \
		}                                                          \
	}

#define CHECK(call)                                                \
	{                                                              \
		const cudaError_t error = call;                            \
		if (error != cudaSuccess)                                  \
		{                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
			fprintf(stderr, "code: %d, reason: %s\n", error,       \
					cudaGetErrorString(error));                    \
			exit(EXIT_FAILURE);                                    \
		}                                                          \
	}
struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

static GpuTimer timer;
void startTimer()
{
    timer.Start();
}

float stopTimer()
{
    timer.Stop();

	return timer.Elapsed();
}

__global__ void unrollKernel_1(int C, int H, int W, int K, float* image, float* data_col)
{
	int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int W_unroll = H_out * W_out;

	if (t < C * W_unroll)
	{
		c = t / W_unroll;
		s = t % W_unroll;
		h_out = s / W_out;
		w_out = s % W_out;
		h_unroll = h_out * W_out + w_out;
		w_base = c * (K * K);

		for (p = 0; p < K; p++)
		{
			for (q = 0; q < K; q++)
			{
				w_unroll = w_base + p * K + q;
				data_col[w_unroll * W_unroll + h_unroll] = image[c * H * W + (h_out + p) * W + (w_out + q)];
			}
		}
	}
}

void unrollGPUWrapper(int C, int H, int W, int K, float* image, float* data_col)
{
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int W_unroll = H_out * W_out;
	int num_threads = C * H_out * W_out;
	int block_size = 1024;
	int num_blocks = ceil((float)num_threads / block_size);
	
	// Copy image to device
	float* d_image;
	CHECK(cudaMalloc(&d_image, C * H * W * sizeof(float)));
	CHECK(cudaMemcpy(d_image, image, C * H * W * sizeof(float), cudaMemcpyHostToDevice));

	// Copy data_col to device
	float* d_data_col;
	CHECK(cudaMalloc(&d_data_col, C * K * K * W_unroll * sizeof(float)));

	unrollKernel_1<<<num_blocks, block_size>>>(C, H, W, K, d_image, d_data_col);
	CHECK(cudaGetLastError());

	// Copy data_col back to host
	CHECK(cudaMemcpy(data_col, d_data_col, C * K * K * W_unroll * sizeof(float), cudaMemcpyDeviceToHost));
	// Free memory
	CHECK(cudaFree(d_image));
	CHECK(cudaFree(d_data_col));
}