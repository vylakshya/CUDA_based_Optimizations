#include<iostream>
#include<cuda_runtime.h>

using namespace std;
__global__ void func(float *A, float *B, float *C, int N){
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < N && j < N){
		C[N*i + j] = A[N*i + j] + B[N*i + j];
	}
	else return ;
}

int main(){
	float *d_A, *d_B, *d_C;
	int N = 10;
	size_t Bytes = N * N * sizeof(float);	
	d_A = new float[N * N];
	d_B = new float[N * N];
	d_C = new float[N * N];
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			d_A[N*i + j] = i + j;
			d_B[N*i + j] = 2*i + j;
		}
	}
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++) cout << d_A[N*i + j] << " ";
		cout << endl;
	}
	cout << endl;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++) cout << d_B[N*i + j] << " ";
		cout << endl;
	}
	cout << endl;	
	float *c_A, *c_B, *c_C;
	cudaMalloc(&c_A, Bytes);
	cudaMalloc(&c_B, Bytes);
	cudaMalloc(&c_C, Bytes);
	cudaMemcpy(c_A, d_A, Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(c_B, d_B, Bytes, cudaMemcpyHostToDevice);
	
  	int blockSize = 16;
	dim3 threadsPerBlock(blockSize, blockSize);
	// Calculate 2D Grid size (Number of blocks needed in X and Y)
	int gridSize = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
	dim3 blocksPerGrid(gridSize, gridSize);
    
	func<<<blocksPerGrid, threadsPerBlock >>> (c_A, c_B, c_C, N);	
	cudaMemcpy(d_C, c_C, Bytes, cudaMemcpyDeviceToHost);	
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++) cout << d_C[N*i + j] << " ";
		cout << endl;
	}
	cudaFree(c_A);
	cudaFree(c_B);
	cudaFree(c_C);
	return 0;
}
