
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <Windows.h>
#include <intrin.h>

 
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#pragma comment(lib, "cudart") 

#define SIZE_M 128
#define SIZE_N 1024

using namespace std;

void fillMatrix(unsigned int*, int, int);
bool compareMatricies(unsigned int*, unsigned int*, int, int);

void showMatrix(unsigned int* mat, int sizeOfM, int sizeOfN) {
	for (int i = 0; i < sizeOfN / 64; i++) {
		for (int j = 0; j < sizeOfM / 16; j++) {
			cout << mat[(sizeOfM / 2) * i + j] << " ";
		}
		cout << endl;
	}
}

void cpuMatrix(unsigned int* inMatrix, unsigned int* outMatrix, int sizeOfM, int sizeOfN) {
	int orfer[] = { 1, 2 , 0 ,3 };
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
				outMatrix[(counter / 2) * sizeOfM / 2 + a + i * 2 * sizeOfM / 2] = inMatrix[i * sizeOfM + tmp];
			}
		}
	}
	QueryPerformanceCounter(&finish);
	delay = (finish.QuadPart - start.QuadPart) * 1000.0f / frequency.QuadPart;
	printf("The time for cpu spend: %.3f ms\n", delay);
}


__global__ void cudaSharedKernel(unsigned int* src, unsigned int* dst)
{

	const int offsetX = 4 * blockIdx.x + threadIdx.x;
	const int offsetY = threadIdx.y;

	__shared__ int smemIn[1024 * 4];
	__shared__ int smemOut[1024 * 4];

	smemIn[threadIdx.y * 4 + threadIdx.x + 0] = src[offsetY * SIZE_M + offsetX + 0];
	smemIn[threadIdx.y * 4 + threadIdx.x + 1] = src[offsetY * SIZE_M + offsetX + 1];
	smemIn[threadIdx.y * 4 + threadIdx.x + 2] = src[offsetY * SIZE_M + offsetX + 2];
	smemIn[threadIdx.y * 4 + threadIdx.x + 3] = src[offsetY * SIZE_M + offsetX + 3];

	__syncthreads();

	int a = smemIn[threadIdx.y * 4 + threadIdx.x + 0];
	int b = smemIn[threadIdx.y * 4 + threadIdx.x + 1];
	int c = smemIn[threadIdx.y * 4 + threadIdx.x + 2];
	int d = smemIn[threadIdx.y * 4 + threadIdx.x + 3];

	smemOut[threadIdx.y * 4 + threadIdx.x + 0] = a;
	smemOut[threadIdx.y * 4 + threadIdx.x + 1] = b;
	smemOut[threadIdx.y * 4 + threadIdx.x + 2] = c;
	smemOut[threadIdx.y * 4 + threadIdx.x + 3] = d;

	const int offsetOutX = 2 * blockIdx.x + threadIdx.x;
	const int offsetOutY = 2 * threadIdx.y;

	__syncthreads();

	dst[offsetOutY * SIZE_M / 2 + offsetOutX + 0] = smemOut[threadIdx.y * 4 + threadIdx.x + 1];
	dst[offsetOutY * SIZE_M / 2 + offsetOutX + 1] = smemOut[threadIdx.y * 4 + threadIdx.x + 2];
	dst[offsetOutY * SIZE_M / 2 + SIZE_M / 2 + offsetOutX + 1] = smemOut[threadIdx.y * 4 + threadIdx.x + 3];
	dst[offsetOutY * SIZE_M / 2 + SIZE_M / 2 + offsetOutX + 0] = smemOut[threadIdx.y * 4 + threadIdx.x + 0];
}

__global__ void cudaKernel(unsigned int *init, unsigned int* dest) {
	int offsetX = 4 * blockIdx.x + threadIdx.x;
	int offsetY = threadIdx.y;

	int a = init[offsetY * SIZE_M + offsetX + 0];
	int b = init[offsetY * SIZE_M + offsetX + 1];
	int c = init[offsetY * SIZE_M + offsetX + 2];
	int d = init[offsetY * SIZE_M + offsetX + 3];

	int offsetOutX = 2 * blockIdx.x + threadIdx.x;
	int offsetOutY = 2 * threadIdx.y;

	dest[offsetOutY * SIZE_M / 2 + offsetOutX + 0] = b;
	dest[offsetOutY * SIZE_M / 2 + offsetOutX + 1] = c;
	dest[offsetOutY * SIZE_M / 2 + SIZE_M / 2 + offsetOutX + 1] = d;
	dest[offsetOutY * SIZE_M / 2 + SIZE_M / 2 + offsetOutX + 0] = a;
}

void cudaSharedMatrix(unsigned int *init, unsigned int *dest) {
	float resultTime;

	unsigned int* deviceInMatrix;
	unsigned int* deviceOutMatrix;
	//события для замера времени в CUDA

	cudaEvent_t cuda_startTime;
	cudaEvent_t cuda_endTime;
	//Создание событий
	cudaEventCreate(&cuda_startTime);
	cudaEventCreate(&cuda_endTime);

	cudaMalloc(&deviceInMatrix, (SIZE_M * SIZE_N * sizeof(int)));
	cudaMalloc(&deviceOutMatrix, (SIZE_N * SIZE_M * sizeof(int)));
	cudaMemcpy(deviceInMatrix, init, SIZE_M * SIZE_N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGrid(32, 1);
	dim3 dimBlock(1, 1024);
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


void cudaMatrix(unsigned int *init, unsigned int *dest) {
	float resultTime;

	unsigned int* deviceInMatrix;
	unsigned int* deviceOutMatrix;
	//события для замера времени в CUDA

	cudaEvent_t cuda_startTime;
	cudaEvent_t cuda_endTime;
	//Создание событий
	cudaEventCreate(&cuda_startTime);
	cudaEventCreate(&cuda_endTime);

	cudaMalloc(&deviceInMatrix, (SIZE_M*SIZE_N * sizeof(int)));
	cudaMalloc(&deviceOutMatrix, (SIZE_N * SIZE_M * sizeof(int)));
	cudaMemcpy(deviceInMatrix, init, SIZE_M * SIZE_N * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGrid(32, 1);
	dim3 dimBlock(1, 1024);

	cudaEventRecord(cuda_startTime, 0);

	cudaKernel <<<dimGrid, dimBlock >>> (deviceInMatrix, deviceOutMatrix);

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
	//unsigned int* initMatrix = (unsigned int*)malloc(SIZE_M * SIZE_N * sizeof(int));
	unsigned int* initMatrix = (unsigned int*)malloc(SIZE_M * SIZE_N * sizeof(int));
	unsigned int* cpu_outMatrix = (unsigned int*)malloc(SIZE_M * SIZE_N * sizeof(int));
	unsigned int* cuda_outMatrix = (unsigned int*)malloc(SIZE_M * SIZE_N * sizeof(int));
	unsigned int* cuda_outMatrixSharedMemory = (unsigned int*)malloc(SIZE_M * SIZE_N * sizeof(int));
	
	fillMatrix(initMatrix, SIZE_M, SIZE_N);

	cpuMatrix(initMatrix, cpu_outMatrix, SIZE_M, SIZE_N);
	//showPartOfMatrix(cpu_outMatrix);
	showMatrix(cpu_outMatrix, SIZE_M, SIZE_N);

	cudaMatrix(initMatrix, cuda_outMatrix);
	//showPartOfMatrix(cuda_outMatrix);

	showMatrix(cuda_outMatrix, SIZE_M, SIZE_N);
	cudaSharedMatrix(initMatrix, cuda_outMatrixSharedMemory);
	showMatrix(cuda_outMatrixSharedMemory, SIZE_M, SIZE_N);

	//showMatrix(cpu_outMatrix, SIZE_M/2 , SIZE_N * 2);
	if (compareMatricies(cuda_outMatrix, cpu_outMatrix, SIZE_M, SIZE_N) && compareMatricies(cuda_outMatrix, cpu_outMatrix, SIZE_M, SIZE_N)) {
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


void fillMatrix(unsigned int* matrix, int sizeM, int sizeN)
{	
	curandGenerator_t generator;

	/*curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
	curandGenerate(generator, matrix, sizeM * sizeN);
	curandDestroyGenerator(generator);*/
	
	unsigned int* devData;

	cudaMalloc(&devData, sizeM * sizeN);

	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, time(NULL));

	curandGenerate(generator, (unsigned int*)devData, sizeM * sizeN / sizeof(unsigned int));

	cudaMemcpy(matrix, devData, sizeM * sizeN, cudaMemcpyDeviceToHost);
	curandDestroyGenerator(generator);

	cudaFree(devData);
	/*int counter = 0;

	for (int i = 0; i < sizeN; ++i)
	{
		for (int j = 0; j < sizeM; ++j)
		{
			matrix[sizeM * i + j] = counter++;
		}
	}*/
}

bool compareMatricies(unsigned int* inMatrix, unsigned int* outMatrix, int sizeOfM, int sizeOfN) {
	for (int i = 0; i < sizeOfN * sizeOfM; i++) {
		if (inMatrix[i] != outMatrix[i]) {
			return false;
		}
	}
	return true;
}
