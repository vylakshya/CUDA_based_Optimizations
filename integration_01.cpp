#include<iostream>
#include<cuda_runtime.h>

using namespace std;


__global__ void Integration(double a, double b){


}	

int main(){

	double a, b;
	cin >> a >> b;
	double c, d;
	cudaMalloc(c,sizeof(double));
	cudaMalloc(d,sizeof(double));
	cudaMemcpy(c,a,sizeof(double),cudaMemcpyHostToDevice);


	return 0;
}
