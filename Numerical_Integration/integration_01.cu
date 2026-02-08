#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define BLOCK_SIZE 256

// The function integrating: f(x) = x^2

__device__ float f(float x) {
    return x * x; 
}

__global__ void integrateKernel(float a, float h, int n, float* d_result) {
    // Shared memory for the final reduction step in the block
    __shared__ float cache[BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Grid-Stride Loop
    for (int i = tid; i < n; i += stride) {
        float x_i = a + i * h;
        float x_next = a + (i + 1) * h;
        local_sum += (f(x_i) + f(x_next)) * 0.5f * h;
    }

    cache[threadIdx.x] = local_sum;
    __syncthreads();

    // Parallel Reduction in Shared Memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            cache[threadIdx.x] += cache[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Atomic Add to Global Memory
    if (threadIdx.x == 0) {
        atomicAdd(d_result, cache[0]);
    }
}

int main() {
    const int n = 1000000; // Number of intervals
    const float a = 0.0f;  // Lower bound
    const float b = 1.0f;  // Upper bound
    const float h = (b - a) / n;

    float h_result = 0.0f;
    float *d_result;

    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    // We don't need millions of threads; 256 blocks of 256 threads is plenty
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = 256;

    integrateKernel<<<blocksPerGrid, threadsPerBlock>>>(a, h, n, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Integral of x^2 from " << a << " to " << b << " is: " << h_result << std::endl;
    std::cout << "Analytical Result: " << (pow(b, 3)/3.0) - (pow(a, 3)/3.0) << std::endl;

    cudaFree(d_result);
    return 0;
}
