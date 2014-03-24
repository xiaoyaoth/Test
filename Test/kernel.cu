#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "thrust\scan.h"
#include "thrust\sort.h"
#include "thrust\device_ptr.h"
#include "thrust\device_vector.h"
#include <stdio.h>

#define AGENTNO 1024
#define BUFFERSIZE 2048
#define BLOCK_SIZE 128
#define DICE 0.9
#define VERBOSE 0

#define checkCudaErrors(err)	__checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}

#define getLastCudaError(msg)	__getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
		system("PAUSE");
		exit(-1);
	}
}

__device__ unsigned int numAg;
__device__ unsigned int numAgTemp;
int numAg_h;

class GRandom;
class GAgent;
class GModel;

class GRandom {
	curandState rState;
public:
	__device__ GRandom(int seed, int idx){
		curand_init(seed, idx, 0, &rState);
	}
	__device__ float genUniform(){
		return curand_uniform(&rState);
	}
};

class GModel {
	GAgent **alist;
};

class GAgent{
public:
	int id;
	GRandom *random;
	__device__ GAgent(){
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		this->random = new GRandom(2345, idx);
	}
	__device__ virtual void step(GModel *gm) = 0;
};

class AgentOne : public GAgent {
	__device__ void step(GModel *gm) {
		printf("%d ", this->id);
	}
};

class AgentTwo : public GAgent {
	__device__ void step(GModel *gm) {
	}
};

__global__ void init(GAgent **alist){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	AgentOne *ag = new AgentOne();
	ag->id = idx;
	alist[idx] = ag;
}

__device__ int atomicAdd1(int* address, int val)
{
	unsigned int old = *address, assumed;
	do {
		assumed = old;
		old = atomicCAS(address, assumed, (val + assumed)); 
	} while (assumed != old);
	return old;
}

__global__ void setNumAg(){
	numAg = numAgTemp;
}

__global__ void insert(GAgent **alist, GModel *gm){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	GAgent *ag = alist[idx];
	float dice = ag->random->genUniform();
	ag->step(gm);
	if (dice < DICE) {
		AgentOne *newAg = new AgentOne();
		int newIdx = atomicInc((unsigned int *)&numAg, BUFFERSIZE);
		newAg->id = newIdx;
		alist[newIdx] = newAg;
	}
	float test = dice;
}

__global__ void remove(GAgent **alist, GModel *gm){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAg /2) {
		atomicDec(&numAgTemp, BUFFERSIZE);
		GAgent *ag = alist[idx];
		delete ag;
		alist[idx] = NULL;
	}
}

__global__ void check(GAgent **alist) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numAg) {
		GAgent *ag = alist[idx];
		float dice = ag->random->genUniform();
	}
}

struct AListComp {
	//returning true means the one with satisfied condition will be put in the front
	__host__ __device__
		bool operator()(const GAgent *a, const GAgent *b) {
			if (a != NULL)
				return true;
			return false;
	}
};

void sortAList(GAgent **alist){
	thrust::device_ptr<GAgent *> alist_ptr(alist);
	typedef thrust::device_vector<GAgent *>::iterator Iter;
	Iter key_begin(alist_ptr);
	Iter key_end(alist_ptr + numAg_h);
	thrust::sort(key_begin, key_end, AListComp());
	getLastCudaError("sort_hash_kernel");
}

int main()
{
	AgentOne *ag = new AgentOne();
	ag->id = 100;
	delete ag;
	ag->id = 0;

	numAg_h = AGENTNO;
	printf("size of curandState: %d\n", sizeof(curandState));
	int GRID_SIZE = (int)(AGENTNO/BLOCK_SIZE);
	cudaMemcpyToSymbol(numAgTemp, &numAg_h, sizeof(int), 0, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	GAgent **a_dev;
	cudaMalloc((void**)&a_dev, BUFFERSIZE*sizeof(GAgent*));
	cudaMemset(a_dev, 0, BUFFERSIZE*sizeof(GAgent*));

	GModel *gm_dev;
	cudaMalloc((void**)&gm_dev, sizeof(GModel));

	init<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev);

	cudaEventRecord(start, 0);
	//insert<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev, gm_dev);
	for (int i = 0; i < 10; i++) {
		setNumAg<<<1, 1>>>();
		remove<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev, gm_dev);
		sortAList(a_dev);
		cudaMemcpyFromSymbol(&numAg_h, numAgTemp, sizeof(int), 0, cudaMemcpyDeviceToHost);
		GRID_SIZE = numAg_h%BLOCK_SIZE==0 ? numAg_h/BLOCK_SIZE : numAg_h/BLOCK_SIZE + 1;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time = 1.0;
	cudaEventElapsedTime(&time, start, stop);
	printf("time: %f\n", time);
	cudaMemcpyFromSymbol(&numAg_h, numAg, sizeof(int), 0, cudaMemcpyDeviceToHost);
	printf("numAg: %d\n", numAg_h);

	//GRID_SIZE = (int)(BUFFERSIZE/BLOCK_SIZE);
	//check<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev);

	system("PAUSE");
	return 0;
}

__global__ void scanInit(int *a_dev){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	a_dev[idx] = 1;
}

__global__ void scanInsert(){
	AgentOne *ag = new AgentOne();
}

int main2(){
	int *a_dev;
	checkCudaErrors(cudaMalloc((void**)&a_dev, AGENTNO * sizeof(int)));
	thrust::device_ptr<int> a_ptr(a_dev);
	thrust::device_vector<int>::iterator key_begin(a_ptr);
	thrust::device_vector<int>::iterator key_end(a_ptr + AGENTNO);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int GRID_SIZE = AGENTNO/BLOCK_SIZE;
	int SMEM_SIZE = BLOCK_SIZE * sizeof(int);
	scanInit<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev);

	cudaEventRecord(start, 0);
	//scanInsert<<<GRID_SIZE, BLOCK_SIZE>>>();
	thrust::sort(key_begin, key_end);
	thrust::inclusive_scan(key_begin, key_end, a_ptr);
	cudaEventRecord(stop, 0);  
	cudaEventSynchronize(stop);

	float insertTime = 0;
	cudaEventElapsedTime(&insertTime, start, stop);
	printf("insert time: %f\n", insertTime);

	int *a_host = (int*)malloc(AGENTNO * sizeof(int));
	cudaMemcpy(a_host, a_dev, sizeof(int) * AGENTNO, cudaMemcpyDeviceToHost);
	printf("%d ", a_host[AGENTNO-1]);
	system("PAUSE");
	return 0;
}