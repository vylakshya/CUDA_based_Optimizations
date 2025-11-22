#include<iostream>
#include<cuda_runtime.h>

using namespace std;


__global__ void VecAdd(float* A, float* B, float* C, int N){

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if( i < N)
		C[i] = A[i] + B[i];

}


int main(void){

	int N = 1000 * sizeof(float);
	float* h_A = (float*)malloc(N);
	float* h_B = (float*)malloc(N);
	float* h_C = (float*)malloc(N);
	
	for(int i = 0; i < 1000; i++){
		*(h_A + i) = i * 10;
		*(h_B + i) = i;
	}

	float* d_A;
	cudaMalloc(&d_A, N);
	float* d_B;
	cudaMalloc(&d_B, N);
	float* d_C;
	cudaMalloc(&d_C, N);

	cudaMemcpy(d_A, h_A, N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	VecAdd <<< blocksPerGrid, threadsPerBlock  >>> (d_A,d_B,d_C, N);
	
	cudaMemcpy(h_C, d_C, N, cudaMemcpyDeviceToHost);
	for(int i = 0; i < 1000; i++){
		cout << h_C[i] << "\n";
	}
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
