#include<iostream>
#include<cuda_runtime.h>
using namespace std;

double dx = 1e-9;
__global__ void Integration(double dx,double a, double b, double *ans, int N){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N){
		double x = a + i*dx;
		atomicAdd(ans, (x*x)*dx);
	}
	else return ;
}
int main(){


	double a, b, ans_h = 0;
	cin >> a >> b;
	int N = (b - a)/dx;
	double *ans = &ans_h;
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
	Integration<<<blocksPerGrid, threadsPerBlock >>> (dx,a, b, ans, N);
	cout << *ans << endl;
	return 0;
}
