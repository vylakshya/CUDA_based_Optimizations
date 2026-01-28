#include<iostream>
#include<cuda_runtime.h>
#include<vector>

using namespace std;

__constant__ char d_message[20];

__global__ void welcome(char* msg){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	msg[idx] = d_message[idx];
}

int main(){
	char* d_msg;
	char* h_msg;
	const char mess[] = "Welcome to GPU computing\n";
	const int length = strlen(mess) + 1;

	h_msg = (char*)malloc(length * sizeof(char));
	cudaMalloc(&d_msg, length * sizeof(char));
	
	cudaMemcpyToSymbol(d_message, mess, length);

	welcome<<1, length>>(d_msg);
	
	cudaMemcpy(h_msg, d_msg, length * sizeof(char), cudaMemcpyDeviceToHost);
	h_msg[length-1] = '\0';

	cout << h_msg << "\n";

	free(h_msg);
	cudaFree(d_msg);

	 
	return 0;
}
