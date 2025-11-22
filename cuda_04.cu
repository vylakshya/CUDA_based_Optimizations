#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__ void Translate(float *V, int v_r, int v_c, float *M, int m_r, int m_c, float *T){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if(i < m_r && j < v_c){
		float sum = 0.0f;
		for(int k = 0; k < v_r; k++){
			sum += M[m_c*i + k] * V[v_c*k + j];
		}
		T[v_c*i + j] = sum;
	}
	else return ;
}	
__global__ void TransformVec(float *V, int v_r, int v_c, float *T, int T_r, int T_c, float *Transform){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if(i < T_r && j < 1){
		for(int k = 0; k < v_r; k++) Transform[v_c*i + j] += T[T_c*i + k] * V[k];
	}

}

int main(){
	int m1 = 3, n1 = 4;
	int m2 = 4, n2 = 1;
	size_t Bytes_V = m2 * n2 * sizeof(float);
	size_t Bytes_M = m1 * n1 * sizeof(float);
	size_t Bytes_T = m1 * n2 * sizeof(float);
	float *h_V, *h_T;
	h_V = new float[m2*n2];
	h_T = new float[m1*n2];

	//taking the vector as input
	cout << "Enter the initial Vector :- ";
	for(int i = 0; i < 3; i++) cin >> h_V[i];

	//Init the pivot which will translate
	h_V[3] = 1;
	for(int i = 0; i < 3; i++) cout << h_V[i] << " ";
	cout << endl;	

	int x, y, z; //The translation parameters
	cout << "Enter the changes you wanna make to translate this vector in the order of x, y and z : - ";
       	cin >> x >> y >> z;
	//Translation Matrix
	float h_M[12] = {1.0f, 0.0f, 1.0f, (float)x,
	                 0.0f, 1.0f, 0.0f, (float)y,
	                 0.0f, 0.0f, 1.0f, (float)z};
	for(int i = 0; i < m1; i++){
		for(int j = 0; j < n1; j++) cout << h_M[n1*i + j] << " ";
		cout << endl;
	}

	//GPU Memory allocation
	float *d_V, *d_M, *d_T;
	cudaMalloc(&d_V, Bytes_V);
	cudaMalloc(&d_M, Bytes_M);
	cudaMalloc(&d_T, Bytes_T);

	// Transferring data from Host(CPU) to Device(GPU) 
	cudaMemcpy(d_V, h_V, Bytes_V, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M, h_M, Bytes_M, cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, h_T, Bytes_T, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int BlocksPerGrid = (n2 + threadsPerBlock - 1)/threadsPerBlock;
	cout << "Translating the Vector to your desired position ...............\n";
	// Activating Translation Kernel
	
	TransformVec <<< BlocksPerGrid, threadsPerBlock >>> (d_V, m2, n2, d_M, m1, n1, d_T);
	
	// Transferring the calculated data back to Host(CPU) from Device(GPU)
	cudaMemcpy(h_T, d_T, Bytes_T, cudaMemcpyDeviceToHost);
	
	// Putting the desired Result out 
	cout << "The translated Vector [" << h_V[0] << ", " << h_V[1] << ", " << h_V[2] << "] is : " << endl;
	cout << "[" << h_T[0] << ", " << h_T[1] << ", " << h_T[2] << "] ";
	cout << "\n Translated Successfully!\n";
	return 0;
}
