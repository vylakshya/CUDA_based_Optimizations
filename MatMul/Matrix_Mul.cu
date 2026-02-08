#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define TILE_SIZE 32


__global__ void matMulTiledKernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
      
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); 

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads(); 
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

int main() {
    // Matrix dimensions (M x K) * (K x N) = (M x N)
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.0f); // Fill with 1.0
    std::vector<float> h_B(K * N, 2.0f); // Fill with 2.0
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, sizeA));
    checkCuda(cudaMalloc(&d_B, sizeB));
    checkCuda(cudaMalloc(&d_C, sizeC));

    checkCuda(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "Launching Kernel: Matrix " << M << "x" << K << " * " << K << "x" << N << std::endl;
    matMulTiledKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < M * N; i++) {
        if (std::abs(h_C[i] - (float)K * 2.0f) > 1e-5) {
            success = false;
            break;
        }
    }

    if (success) std::cout << "Success! Matrix Multiplication Correct." << std::endl;
    else std::cout << "Verification Failed!" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
