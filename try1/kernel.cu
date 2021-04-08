#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream> 
#include <cuda_runtime.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>
#include <ctime>
#include <cmath>

#pragma comment(lib, "cudart") 

#define SIZE_M 8
#define SIZE_N 4
#define COUNT_OF_THREADS 1024
#define MAX_BLOCKS 200000

using namespace std;

void cpu_matrixOperation(short*, short*, int, int);
void cuda_matrixOperation(short*, short*, bool);
//void cuda_checkStatus(cudaError_t);
void fillMatrix(short*, int, int);
bool checkEquality(short*, short*, int, int);

void showMatrix(short* mat, int sizeOfM, int sizeOfN) {
	for (int i = 0; i < sizeOfN; i++) {
		for (int j = 0; j < sizeOfM; j++) {
			cout << mat[sizeOfM * i + j] << " " ;
		}
		cout << endl;
	}
	/*for (int i = 0; i < size; i++) {

		cout << mat[i] << endl;
	}*/
}

void cpuMatrix(short* inMatrix, short* outMatrix, int sizeOfM, int sizeOfN) {
	clock_t startTime, endTime;
	startTime = clock();
	int orfer[] = { 1, 2 , 0 ,3 };
	//sizeOfM >> sizeOfN
	for (int h = 0; h < sizeOfM; h += 4) {
		for (auto i = 0; i < sizeOfN; i++) {
			
			for (auto j = h,  counter = 0; counter < 4; j++, counter++) {
				int tmp = orfer[counter] ;
				int a = (j + 1) % 2 == 0 ? 1 : 0;
				if (h > 0) {
					a += 2;
					tmp = orfer[counter] + h ;
				}
				cout << "INndexOUT " << (counter / 2) * sizeOfM / 2 + a + i * 2 * sizeOfM / 2 << " INDEXIN " << i * sizeOfM + tmp << endl;
				outMatrix[(counter / 2) * sizeOfM / 2 + a + i * 2 * sizeOfM / 2] = inMatrix[i * sizeOfM + tmp];

				/*	outMatrix[(i + 1) * sizeOfM / 2] = inMatrix[i * sizeOfM + j];
					j++;
					outMatrix[(i) * sizeOfM / 2 + j - 1] = inMatrix[i * sizeOfM + j];
					j++;
					outMatrix[(i) * sizeOfM / 2 + j - 1] = inMatrix[i * sizeOfM + j];
					j++;
					outMatrix[(i + 1) * sizeOfM / 2 + j - 2] = inMatrix[i * sizeOfM + j];
					i += 2;*/

			}
		}
	}
	endTime = clock();
	printf("CPU time: %lf seconds\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
}



int main() {
	//выделение памяти под матрицы в cpu
	auto* initMatrix = (short*)malloc(SIZE_M * SIZE_N * sizeof(short));
	auto* cpu_outMatrix = (short*)malloc(SIZE_M * SIZE_N * sizeof(short));
	auto* cuda_outMatrix = (short*)malloc(SIZE_M * SIZE_N * sizeof(short));
	auto* cuda_outMatrixSharedMemory = (short*)malloc(SIZE_M * SIZE_N * sizeof(short));

	fillMatrix(initMatrix, SIZE_M, SIZE_N);
	showMatrix(initMatrix, SIZE_M , SIZE_N);
	/*cuda_matrixOperation(initMatrix, cuda_outMatrix, false);
	cuda_matrixOperation(initMatrix, cuda_outMatrixSharedMemory, true);*/

	/*cpu_matrixOperation(initMatrix, cpu_outMatrix, SIZE_M, SIZE_N);*/
	cpuMatrix(initMatrix, cpu_outMatrix, SIZE_M, SIZE_N);
	showMatrix(cpu_outMatrix, SIZE_M/2 , SIZE_N * 2);
	if (checkEquality(cuda_outMatrix, cpu_outMatrix, SIZE_M, SIZE_N)
		&& checkEquality(cuda_outMatrixSharedMemory, cuda_outMatrix, SIZE_M, SIZE_N)) {
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

__global__ void cuda_matrixOperationKernel(int* inMatrix, short* outMatrix, int numOfBlocksInRow) {
	int remainderElements = SIZE_M % COUNT_OF_THREADS;

	if (remainderElements != 0 && (blockIdx.x + 1) % numOfBlocksInRow == 0 && threadIdx.x >= remainderElements) {
		return;
	}

	int *startOfResultRow = &inMatrix[SIZE_M * (blockIdx.x / numOfBlocksInRow)];
	outMatrix = &outMatrix[SIZE_M * (blockIdx.x / numOfBlocksInRow) * 2];

	int elements = 0;
	int countOfThreads = 0;

	if (remainderElements != 0 && (blockIdx.x + 1) % numOfBlocksInRow == 0) {
		countOfThreads = remainderElements;
	}
	else {
		countOfThreads = COUNT_OF_THREADS;
	}

	if (threadIdx.x < (countOfThreads / 2)) {
		elements = startOfResultRow[(blockIdx.x % numOfBlocksInRow) * COUNT_OF_THREADS / 2 + threadIdx.x];
	}
	else {
		elements = startOfResultRow[threadIdx.x % (countOfThreads / 2) + SIZE_M / 2 + (blockIdx.x % numOfBlocksInRow) * COUNT_OF_THREADS / 2];
	}

	short firstElement = (short)elements;
	short secondElement = (short)(elements >> 16);

	int offset = COUNT_OF_THREADS * 2 * (blockIdx.x % numOfBlocksInRow);

	if (threadIdx.x < (countOfThreads / 2)) {
		outMatrix[threadIdx.x * 2 * 2 + offset] = firstElement;
		outMatrix[(threadIdx.x * 2 + 1) * 2 + offset] = secondElement;
	}
	else {
		outMatrix[(threadIdx.x - countOfThreads / 2) * 2 * 2 + 1 + offset] = firstElement;
		outMatrix[((threadIdx.x - countOfThreads / 2) * 2 + 1) * 2 + 1 + offset] = secondElement;
	}
}

//global - ключевое слово, предназначено для указания како
//ядро - функция, которая описывает последовательность операций , которые выполнятся на каждой нити параллельно
__global__ void cuda_matrixSharedMemoryOperationKernel(int* inMatrix, int* outMatrix, int numOfBlocksInRow) {
	int remainderElements = SIZE_M % COUNT_OF_THREADS;

	__shared__ int sharedMemory[COUNT_OF_THREADS];
	__shared__ short sharedMemoryOut[COUNT_OF_THREADS * 2];

	if (remainderElements != 0 && (blockIdx.x + 1) % numOfBlocksInRow == 0 && threadIdx.x >= remainderElements) {
		return;
	}

	int *startOfResultRow = &inMatrix[SIZE_M * (blockIdx.x / numOfBlocksInRow)];
	outMatrix = &outMatrix[SIZE_M * (blockIdx.x / numOfBlocksInRow)];

	int countOfThreads = 0;

	if (remainderElements != 0 && (blockIdx.x + 1) % numOfBlocksInRow == 0) {
		countOfThreads = remainderElements;
	}
	else {
		countOfThreads = COUNT_OF_THREADS;
	}

	if (threadIdx.x < (countOfThreads / 2)) {
		sharedMemory[threadIdx.x] = startOfResultRow[(blockIdx.x % numOfBlocksInRow) * COUNT_OF_THREADS / 2 + threadIdx.x];
	}
	else {
		sharedMemory[threadIdx.x] = startOfResultRow[threadIdx.x % (countOfThreads / 2) + SIZE_M / 2 + (blockIdx.x % numOfBlocksInRow) * COUNT_OF_THREADS / 2];
	}

	int elements = sharedMemory[threadIdx.x];
	short firstElement = (short)elements;
	short secondElement = (short)(elements >> 16);

	int offset = COUNT_OF_THREADS * 2 * (blockIdx.x % numOfBlocksInRow);

	if (threadIdx.x < (countOfThreads / 2)) {
		sharedMemoryOut[threadIdx.x * 2 * 2] = firstElement;
		sharedMemoryOut[(threadIdx.x * 2 + 1) * 2] = secondElement;
	}
	else {
		sharedMemoryOut[(threadIdx.x - countOfThreads / 2) * 2 * 2 + 1] = firstElement;
		sharedMemoryOut[((threadIdx.x - countOfThreads / 2) * 2 + 1) * 2 + 1] = secondElement;
	}

	__syncthreads();

	outMatrix[offset / 2 + threadIdx.x] = ((int*)sharedMemoryOut)[threadIdx.x];
}

void cuda_matrixOperation(short* inMatrix, short* outMatrix, bool optimizationFlag) {
	float resultTime;

	short* device_inMatrix;
	short* device_outMatrix;
	//события для замера времени в CUDA
	cudaEvent_t cuda_startTime;
	cudaEvent_t cuda_endTime;
	//Создание событий
	cudaEventCreate(&cuda_startTime);
	cudaEventCreate(&cuda_endTime);

	//блок состоит из нитей
	int numOfBlocksInRow = (int)ceil((double)SIZE_M / COUNT_OF_THREADS);//количество блоков в строке
	int blocksNeeded = (SIZE_N * numOfBlocksInRow) / 2;// количество блокво которое понадобится
	int maxBlocksPerIteration = MAX_BLOCKS - MAX_BLOCKS % numOfBlocksInRow; // 

	for (int i = 0, int times = 0; i < blocksNeeded; i += maxBlocksPerIteration, times++) {
		int blocksInIteration = (blocksNeeded - i) < maxBlocksPerIteration ? blocksNeeded - i : maxBlocksPerIteration;// ничего не понял

		int numOfRows = (blocksInIteration / numOfBlocksInRow) * 2;
		//выделение памяти для gpu
		cudaMalloc(&device_inMatrix, SIZE_M * numOfRows * sizeof(short));
		cudaMalloc(&device_outMatrix, SIZE_M  * numOfRows * sizeof(short));
		//из памяти хоста скопировал в память устройства GPU
		cudaMemcpy(
			device_inMatrix,
			&inMatrix[SIZE_M * (maxBlocksPerIteration / numOfBlocksInRow) * 2 * times],
			SIZE_M * numOfRows * sizeof(short), cudaMemcpyHostToDevice);

		//Специальная структура для определения 3-хмерных абстракций
		//количество нитей в каждой блоке
		dim3 blockSize(COUNT_OF_THREADS);
		//кол-во блоков
		dim3 gridSize(blocksInIteration);
		//
		cudaEventRecord(cuda_startTime, NULL);
		// << >> - определяет конфигурацию запуска нитей; 1-й параметр - задает кол-во блоков по каждому из 3-х измерений,  2-й параметр - количество нитей в блоке 
		if (optimizationFlag) {
			//разобраться за терминологию cuda (вооооооооот)
			//лаунчим gridsize  копий функций для выполнения параллельно
			cuda_matrixSharedMemoryOperationKernel << < gridSize, blockSize >> > ((int*)device_inMatrix, (int*)device_outMatrix, numOfBlocksInRow);
		}
		else {
			cuda_matrixOperationKernel << < gridSize, blockSize >> > ((int*)device_inMatrix, device_outMatrix, numOfBlocksInRow);
		}

		cudaPeekAtLastError();
		//Считываем время из событий
		cudaEventRecord(cuda_endTime, NULL);
		cudaEventSynchronize(cuda_endTime);

		cudaEventElapsedTime(&resultTime, cuda_startTime, cuda_endTime);

		if (optimizationFlag) {
			printf("%d: CUDA time with optimization: %lf seconds\n", times, (double)resultTime / CLOCKS_PER_SEC);
		}
		else {
			printf("%d: CUDA time: %lf seconds\n", times, (double)resultTime / CLOCKS_PER_SEC);
		}

		cudaMemcpy(
			&outMatrix[SIZE_M * (maxBlocksPerIteration / numOfBlocksInRow) * 2 * times],
			device_outMatrix,
			SIZE_M * numOfRows * sizeof(short), cudaMemcpyDeviceToHost);

		cudaFree(device_inMatrix);
		cudaFree(device_outMatrix);
	}
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



void fillMatrix(short* matrix, int sizeOfM, int sizeOfN) {
	for (int i = 0; i < sizeOfN; i++) {
		for (int j = 0; j < sizeOfM; j++) {
			matrix[sizeOfM * i + j] = rand() % 20 + 1;
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

bool checkEquality(short* inMatrix, short* outMatrix, int sizeOfM, int sizeOfN) {
	for (int i = 0; i < sizeOfN * sizeOfM; i++) {
		if (inMatrix[i] != outMatrix[i]) {
			return false;
		}
	}
	return true;
}


//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
