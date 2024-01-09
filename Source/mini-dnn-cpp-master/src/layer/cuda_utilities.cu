#include <iostream>
#include "cuda_utilities.h"

#define TILE_WIDTH 32

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

__host__ __device__ int idx1D(int r, int c, int colSz) // Create two verision: __host__ to be callable from CPU and run on CPU, __device__ to be callable from GPU and run on GPU
{
    return r * colSz + c;
}

__host__ __device__ int idx1D_col(int r, int c, int rowSz) // Create two verision: __host__ to be callable from CPU and run on CPU, __device__ to be callable from GPU and run on GPU
{
    return c * rowSz + r;
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

__global__ void matrixMultiplicationKernel_1(float* A, float* B, float* C, int m, int n, int k, int image)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int numStride = (n - 1) / TILE_WIDTH + 1;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    for (int stride = 0; stride < numStride; stride++)
    {   
        int globalAIdx = idx1D_col(tidY, stride * TILE_WIDTH + threadIdx.x, m);
        int globalBIdx = idx1D_col(stride * TILE_WIDTH + threadIdx.y, tidX, n);

        if (tidY < m && stride * TILE_WIDTH + threadIdx.x < n)
            s_A[threadIdx.y][threadIdx.x] = A[globalAIdx];
        else
            s_A[threadIdx.y][threadIdx.x] = 0;

        if ((stride * TILE_WIDTH + threadIdx.y) < n && tidX < k)
            s_B[threadIdx.y][threadIdx.x] = B[globalBIdx];
        else
            s_B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();
        
        for (int i = 0; i < TILE_WIDTH; i++)
        {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
			//if (tidY == 4 && tidX == 0 && image == 2) printf("s_A[%d][%d] = %f, s_B[%d][%d] = %f\n", threadIdx.y, i, s_A[threadIdx.y][i], i, threadIdx.x, s_B[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if ( (tidY < m) && (tidX < k)) 
		C[idx1D_col(tidY, tidX, m)] = sum;
}

void matrixMultiplicationCPU(float* A, float *B, float *C, int m, int n, int k)
{	
	for (int r = 0; r < m; r++)
        {
            for (int c = 0; c < k; c++)
            {
                for (int i = 0; i < n; i++) 
                {
                    C[idx1D(r, c, k)] += A[idx1D(r, i, n)] * B[idx1D(i, c, k)];
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

void matrixMultiplicationGPUWrapper(float* A, float *B, float *C, int m, int n, int k, int i)
{	
	memset(C, 0, m * k * sizeof(float));

	dim3 blockSize(32, 32);
	float *d_A, *d_B, *d_C;
	const int size_A = m * n * sizeof(float);
	const int size_B = n * k * sizeof(float);
	const int size_C = m * k * sizeof(float);
	CHECK(cudaMalloc(&d_A, size_A));
	CHECK(cudaMalloc(&d_B, size_B));
	CHECK(cudaMalloc(&d_C, size_C));

	CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice));

	dim3 gridSize( (k - 1)/(blockSize.x) + 1, ( m - 1)/(blockSize.y) + 1);
	matrixMultiplicationKernel_1<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k, i);
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));
}