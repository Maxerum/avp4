#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <Windows.h>
#include <cuda_runtime.h> 
#include <intrin.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#pragma comment(lib, "cudart") 

//#define SIZE_M 8
//#define SIZE_N 4
//#define COUNT_OF_THREADS 1024
//#define MAX_BLOCKS 200000

#define SIZE_M 64
#define SIZE_N 512
#define GRID_X 16
#define GRID_Y 1
#define BLOCK_X 32
#define BLOCK_Y 32
#define THREAD_ELEMENT_X 1
#define THREAD_ELEMENT_Y 4




using namespace std;

//void cpu_matrixOperation(short*, short*, int, int);
//void cuda_matrixOperation(short*, short*, bool);
//void cuda_checkStatus(cudaError_t);
void fillMatrix(int*, int, int);
bool checkEquality(int*, int*, int, int);

void showMatrix(int* mat, int sizeOfM, int sizeOfN) {
	for (int i = 0; i < sizeOfN / 4; i++) {
		for (int j = 0; j < sizeOfM / 8; j++) {
			cout << mat[(sizeOfM / 2) * i + j] << " ";
		}
		cout << endl;
	}
	/*for (int i = 0; i < size; i++) {

		cout << mat[i] << endl;
	}*/
}

void cpuMatrix(int* inMatrix, int* outMatrix, int sizeOfM, int sizeOfN) {
	/*clock_t startTime, endTime;
	startTime = clock();*/
	int orfer[] = { 1, 2 , 0 ,3 };
	//sizeOfM >> sizeOfN
	LARGE_INTEGER frequency, start, finish;
	float delay;
	QueryPerformanceFrequency(&frequency);

	QueryPerformanceCounter(&start);

	for (int h = 0; h < sizeOfM; h += 4) {
		for (auto i = 0; i < sizeOfN; i++) {

			for (auto j = h, counter = 0; counter < 4; j++, counter++) {
				int tmp = orfer[counter];
				int a = (j + 1) % 2 == 0 ? 1 : 0;
				if (h > 0) {

					a += (h / 4) * 2;
					tmp = orfer[counter] + h;
				}
				//cout << "INndexOUT " << (counter / 2) * sizeOfM / 2 + a + i * 2 * sizeOfM / 2 << " INDEXIN " << i * sizeOfM + tmp << endl;
				outMatrix[(counter / 2) * sizeOfM / 2 + a + i * 2 * sizeOfM / 2] = inMatrix[i * sizeOfM + tmp];
				//cout << outMatrix[(counter / 2) * sizeOfM / 2 + a + i * 2 * sizeOfM / 2] << " " << inMatrix[i * sizeOfM + tmp] << endl;
			}
		}
	}
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	printf("The time for cpu spend: %.3f ms\n", delay);
}





__global__ void cudaSharedKernel(int* src, int* dst)
{
	/*const int offsetX = BLOCK_X * blockIdx.x*THREAD_ELEMENT_X + threadIdx.x;
	const int offsetY = BLOCK_Y * blockIdx.y*THREAD_ELEMENT_Y + threadIdx.y;*/

	const int offsetX = 4 * blockIdx.x + threadIdx.x;
	const int offsetY = threadIdx.y;

	__shared__ int smemIn[512 * 4];
	__shared__ int smemOut[512 * 4];

	//int row = BLOCK_X*THREAD_ELEMENT_X;
	int row = 4;
	/*int a = init[offsetY * SIZE_M + offsetX + 0];
	int b = init[offsetY * SIZE_M + offsetX + 1];
	int c = init[offsetY * SIZE_M + offsetX + 2];
	int d = init[offsetY * SIZE_M + offsetX + 3];*/

	smemIn[threadIdx.y * row + threadIdx.x + 0] = src[offsetY * SIZE_M + offsetX + 0];
	smemIn[threadIdx.y * row + threadIdx.x + 1] = src[offsetY * SIZE_M + offsetX + 1];
	smemIn[threadIdx.y * row + threadIdx.x + 2] = src[offsetY * SIZE_M + offsetX + 2];
	smemIn[threadIdx.y * row + threadIdx.x + 3] = src[offsetY * SIZE_M + offsetX + 3];

	__syncthreads();

	int a = smemIn[threadIdx.y * row + threadIdx.x + 0];
	int b = smemIn[threadIdx.y * row + threadIdx.x + 1];
	int c = smemIn[threadIdx.y * row + threadIdx.x + 2];
	int d = smemIn[threadIdx.y * row + threadIdx.x + 3];

	smemOut[threadIdx.y * row + threadIdx.x  + 0] = a;
	smemOut[threadIdx.y * row + threadIdx.x  + 1] = b;
	smemOut[threadIdx.y * row + threadIdx.x  + 2] = c;
	smemOut[threadIdx.y * row + threadIdx.x  + 3] = d;

	const int offsetOutX = 2 * blockIdx.x + threadIdx.x;
	const int offsetOutY = 2 * threadIdx.y;

	__syncthreads();

	dst[offsetOutY * SIZE_M / 2 + offsetOutX + 0] = smemOut[threadIdx.y * row + threadIdx.x + 1];
	dst[offsetOutY * SIZE_M / 2 + offsetOutX + 1] = smemOut[threadIdx.y * row + threadIdx.x + 2];
	dst[offsetOutY * SIZE_M / 2 + SIZE_M / 2 + offsetOutX + 1] = smemOut[threadIdx.y * row + threadIdx.x + 3];
	dst[offsetOutY * SIZE_M / 2 + SIZE_M / 2 + offsetOutX + 0] = smemOut[threadIdx.y * row + threadIdx.x + 0];
}

__global__ void cudaKernel(int *init, int* dest) {
	//const int offsetX = BLOCK_X * blockIdx.x  + threadIdx.x;
	const int offsetX = 4 * blockIdx.x + threadIdx.x;
	const int offsetY = threadIdx.y;
	//const int offsetY = BLOCK_Y * blockIdx.y * THREAD_ELEMENT_Y + threadIdx.y;

	int a = init[offsetY * SIZE_M + offsetX + 0];
	int b = init[offsetY * SIZE_M + offsetX + 1];
	int c = init[offsetY * SIZE_M + offsetX + 2];
	int d = init[offsetY * SIZE_M + offsetX + 3];


	const int offsetOutX = 2 * blockIdx.x + threadIdx.x;
	const int offsetOutY = 2 * threadIdx.y;

	dest[offsetOutY * SIZE_M / 2 + offsetOutX + 0] = b;
	dest[offsetOutY * SIZE_M / 2 + offsetOutX + 1] = c;
	dest[offsetOutY * SIZE_M / 2 + SIZE_M / 2 + offsetOutX + 1] = d;
	dest[offsetOutY * SIZE_M / 2 + SIZE_M / 2 + offsetOutX + 0] = a;

}

void showPartOfMatrix(int *matrix) {
	for (int i = 0; i < SIZE_N / 4; i++) {
		for (int j = 0; j < SIZE_M / 4; j++) {
			cout << matrix[i * SIZE_M + j] << " ";
		}
		cout << endl;
	}
}


void cudaSharedMatrix(int *init, int *dest) {
	float resultTime;

	int* deviceInMatrix;
	int* deviceOutMatrix;
	//события для замера времени в CUDA

	cudaEvent_t cuda_startTime;
	cudaEvent_t cuda_endTime;
	//Создание событий
	cudaEventCreate(&cuda_startTime);
	cudaEventCreate(&cuda_endTime);

	cudaMalloc(&deviceInMatrix, (SIZE_M * SIZE_N * sizeof(int)));
	cudaMalloc(&deviceOutMatrix, (SIZE_N * SIZE_M * sizeof(int)));
	cudaMemcpy(deviceInMatrix, init, SIZE_M * SIZE_N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGrid(16, 1);
	dim3 dimBlock(1, 512);

	//? вопросов много
	/*int row_len = GRID_X * BLOCK_X * THREAD_ELEMENT_X;*/
	cudaEventRecord(cuda_startTime, 0);


	cudaSharedKernel << <dimGrid, dimBlock >> > (deviceInMatrix, deviceOutMatrix);



	cudaPeekAtLastError();
	cudaDeviceSynchronize();
	cudaEventRecord(cuda_endTime, 0);
	cudaEventSynchronize(cuda_endTime);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, cuda_startTime, cuda_endTime);

	cudaEventDestroy(cuda_startTime);
	cudaEventDestroy(cuda_endTime);

	printf("The time for cuda with shared memory spend: %.3f ms\n", elapsedTime);

	cudaMemcpy(init, deviceInMatrix, SIZE_N*SIZE_M * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(dest, deviceOutMatrix, SIZE_N*SIZE_M * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(deviceInMatrix);
	cudaFree(deviceOutMatrix);
}


void cudaMatrix(int *init, int *dest) {
	float resultTime;

	int* deviceInMatrix;
	int* deviceOutMatrix;
	//события для замера времени в CUDA

	cudaEvent_t cuda_startTime;
	cudaEvent_t cuda_endTime;
	//Создание событий
	cudaEventCreate(&cuda_startTime);
	cudaEventCreate(&cuda_endTime);

	cudaMalloc(&deviceInMatrix, (SIZE_M*SIZE_N * sizeof(int)));
	cudaMalloc(&deviceOutMatrix, (SIZE_N * SIZE_M * sizeof(int)));
	cudaMemcpy(deviceInMatrix, init, SIZE_M * SIZE_N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGrid(16, 1);
	dim3 dimBlock(1, 512);

	//? вопросов много
	/*int row_len = GRID_X * BLOCK_X * THREAD_ELEMENT_X;*/
	cudaEventRecord(cuda_startTime, 0);

	cudaKernel << <dimGrid, dimBlock >> > (deviceInMatrix, deviceOutMatrix);

	cudaPeekAtLastError();
	cudaDeviceSynchronize();
	cudaEventRecord(cuda_endTime, 0);
	cudaEventSynchronize(cuda_endTime);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, cuda_startTime, cuda_endTime);

	cudaEventDestroy(cuda_startTime);
	cudaEventDestroy(cuda_endTime);

	printf("The time for cuda with global memory spend: %.3f ms\n", elapsedTime);

	cudaMemcpy(init, deviceInMatrix, SIZE_N*SIZE_M * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(dest, deviceOutMatrix, SIZE_N*SIZE_M * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(deviceInMatrix);
	cudaFree(deviceOutMatrix);
}

int main() {
	//выделение памяти под матрицы в cpu
	int* initMatrix = (int*)malloc(SIZE_M * SIZE_N * sizeof(int));
	int* cpu_outMatrix = (int*)malloc(SIZE_M * SIZE_N * sizeof(int));
	int* cuda_outMatrix = (int*)malloc(SIZE_M * SIZE_N * sizeof(int));
	int* cuda_outMatrixSharedMemory = (int*)malloc(SIZE_M * SIZE_N * sizeof(int));

	fillMatrix(initMatrix, SIZE_M, SIZE_N);

	cpuMatrix(initMatrix, cpu_outMatrix, SIZE_M, SIZE_N);
	//showPartOfMatrix(cpu_outMatrix);
	//showMatrix(cpu_outMatrix, SIZE_M, SIZE_N);

	cudaMatrix(initMatrix, cuda_outMatrix);
	//showPartOfMatrix(cuda_outMatrix);

	//showMatrix(cuda_outMatrix, SIZE_M, SIZE_N);
	cudaSharedMatrix(initMatrix, cuda_outMatrixSharedMemory);
	//showMatrix(cuda_outMatrixSharedMemory, SIZE_M, SIZE_N);

	//showMatrix(cpu_outMatrix, SIZE_M/2 , SIZE_N * 2);
	if (checkEquality(cuda_outMatrixSharedMemory, cuda_outMatrix, SIZE_M, SIZE_N) && checkEquality(cuda_outMatrix, cpu_outMatrix, SIZE_M, SIZE_N)) {
		cout << "Results are equals!" << endl;
	}
	else {
		cout << "Results are NOT equals!" << endl;
	}


	free(initMatrix);
	free(cpu_outMatrix);
	free(cuda_outMatrix);
	free(cuda_outMatrixSharedMemory);
}

void cpu_matrixOperation(short* inMatrix, short* outMatrix, int sizeOfM, int sizeOfN) {
	clock_t startTime, endTime;
	startTime = clock();
	//sizeOfM >> sizeOfN
	for (auto i = 0; i < sizeOfM; i++) {
		for (auto j = 0; j < sizeOfN; j++) {
			int a = (j + 1) % 2 == 0 ? 1 : 0;
			outMatrix[(j / 2) * sizeOfM * 2 + a + i * 2] = inMatrix[i + sizeOfM * j];
		}
	}
	endTime = clock();
	printf("CPU time: %lf seconds\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
}


void fillMatrix(int* matrix, int sizeM, int sizeN)
{
	int counter = 0;
	for (int i = 0; i < sizeN; ++i)
	{
		for (int j = 0; j < sizeM; ++j)
		{
			matrix[sizeM * i + j] = counter++;
		}
	}
}

//void cuda_checkStatus(cudaError_t cudaStatus) {
//	if (cudaStatus != cudaSuccess) {
//		cout << "CUDA return error code: " << cudaStatus;
//		cout << " " << cudaGetErrorString(cudaStatus) << endl;
//		exit(-1);
//	}
//}

bool checkEquality(int* inMatrix, int* outMatrix, int sizeOfM, int sizeOfN) {
	for (int i = 0; i < sizeOfN * sizeOfM; i++) {
		if (inMatrix[i] != outMatrix[i]) {
			return false;
		}
	}
	return true;
}
