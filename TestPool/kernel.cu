#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust\sort.h"
#include "thrust\device_vector.h"
#include "thrust\device_ptr.h"

#include <stdio.h>

#ifndef __CLASS_POOL__
#define __CLASS_POOL__

#define BLOCK_SIZE 128
#define GRID_SIZE(n) (n%BLOCK_SIZE==0 ? n/BLOCK_SIZE : n/BLOCK_SIZE + 1)

template<class Obj> class Pool;

namespace poolUtil{
	template<class Obj> __global__ void initPool(Pool<Obj> *pDev);
	template<class Obj> __global__ void cleanupDevice(Pool<Obj> *pDev);
	template<class Obj> void cleanup(Pool<Obj> *pHost, Pool<Obj> *pDev);
};
template<class Obj> class Pool 
{
public:
	/* pointer array, elements are pointers points to elements in data array
	Since pointers are light weighted, it will be manipulated in sorting, etc. */
	Obj **ptrArray;

	/* keeping the actual Objects data */
	Obj *dataArray;

	/* objects to be deleted will be marked as delete */
	bool *delMark;

	/*the pointers in the ptrArray are one-to-one associated with the elem in dataArray.
	No matter what the data is, the pointer points to the same data.
	However, the ptrArray and delMark are one-to-one corresponding, i.e., ptrArray is sorted
	with delMark*/

	unsigned int numElem;
	unsigned int numElemMax;
	unsigned int incCount;
	unsigned int decCount;
public:
	__device__ Obj* add();
	__device__ void remove(int idx);
	__device__ void link();
	__host__ void alloc(int nElem, int nElemMax);
	__host__ Pool(int nElem, int nElemMax);

	friend __global__ void poolUtil::initPool(Pool<Obj> *pDev);
	friend __global__ void poolUtil::cleanupDevice(Pool<Obj> *pDev);
	friend void poolUtil::cleanup(Pool<Obj> *pHost, Pool<Obj> *pDev);
};

//Pool implementation
template<class Obj> __device__ Obj* Pool<Obj>::add()
{
	int idx = atomicInc(&incCount, numElemMax-numElem) + numElem;
	this->delMark[idx] = false;
	return this->ptrArray[idx];
}
template<class Obj> __device__ void Pool<Obj>::remove(int idx)
{
	delMark[idx] = true;
	atomicInc(&decCount, numElem);
}
template<class Obj> __host__ void Pool<Obj>::alloc(int nElem, int nElemMax){
	printf("sizeof obj in Pool<Obj>::alloc: %d\n", sizeof(Obj));
	this->numElem = nElem;
	this->numElemMax = nElemMax;
	this->incCount = 0;
	this->decCount = 0;
	cudaMalloc((void**)&this->delMark, nElemMax * sizeof(bool));
	cudaMalloc((void**)&this->dataArray, nElemMax * sizeof(Obj));
	cudaMalloc((void**)&this->ptrArray, nElemMax * sizeof(Obj*));
	cudaMemset(this->dataArray, 0xff, nElemMax * sizeof(Obj));
	cudaMemset(this->ptrArray, 0x00, nElemMax * sizeof(Obj*));
	cudaMemset(this->delMark, 1, nElemMax * sizeof(bool));
}
template<class Obj> __host__ Pool<Obj>::Pool(int nElem, int nElemMax){
	this->alloc(nElem, nElemMax);
}

//poolUtil implementation
template<class Obj> __global__ void poolUtil::initPool(Pool<Obj> *pDev) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < pDev->numElemMax)
		pDev->ptrArray[idx] = &pDev->dataArray[idx];
}
template<class Obj> __global__ void poolUtil::cleanupDevice(Pool<Obj> *pDev)
{
	pDev->numElem = pDev->numElem + pDev->incCount - pDev->decCount;
	pDev->incCount = 0;
	pDev->decCount = 0;
}
template<class Obj> void poolUtil::cleanup(Pool<Obj> *pHost, Pool<Obj> *pDev)
{
	/**/
	void **ptrArrayLocal = (void**)pHost->ptrArray;
	bool *delMarkLocal = pHost->delMark;

	thrust::device_ptr<void*> objPtr(ptrArrayLocal);
	thrust::device_ptr<bool> dMarkPtr(delMarkLocal);
	typedef thrust::device_vector<void*>::iterator objIter;
	typedef thrust::device_vector<bool>::iterator dMarkIter;
	dMarkIter keyBegin(dMarkPtr);
	dMarkIter keyEnd(dMarkPtr + pHost->numElemMax);
	objIter valBegin(objPtr);
	thrust::sort_by_key(keyBegin, keyEnd, valBegin, thrust::less<int>());

	cleanupDevice<<<1, 1>>>(pDev);
	cudaMemcpy(pHost, pDev, sizeof(Pool<Obj>), cudaMemcpyDeviceToHost);
}

#endif

class GAgent {
public:
	int id;
	float x;
	float y;
};
__global__ void init(Pool<GAgent> *pool) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int numElemLocal = pool->numElem;
	int numElemMaxLocal = pool->numElemMax;
	if (idx < numElemLocal) {
		pool->delMark[idx] = false;
		pool->dataArray[idx].id = idx;
	}
}
__global__ void step(Pool<GAgent> *pool, int round, int *resDev){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int numElemLocal = pool->numElem;
	int numElemMaxLocal = pool->numElemMax;
	GAgent *ptr = NULL;
	if (idx < numElemLocal) {
		resDev[idx] = pool->ptrArray[idx]->id;
	}
	if (idx < numElemLocal / 4) {
		pool->remove(idx);
	}
	if (idx >= numElemLocal *3 / 4) {
		ptr = pool->add();
		ptr->id = round;
	}
}

int main(int argc, char **argv){
	int numElem = atoi(argv[1]);
	int numElemMax = atoi(argv[2]);
	int round = atoi(argv[3]);

	int *res = (int*)malloc(numElemMax * sizeof(int));
	int *resDev;
	cudaMalloc((void**)&resDev, numElemMax * sizeof(int));

	printf("size of Pool<int>: %d\n", sizeof(Pool<GAgent>));
	Pool<GAgent> *poolHost = new Pool<GAgent>(numElem, numElemMax);
	Pool<GAgent> *poolDevice;
	cudaMalloc((void**)&poolDevice, sizeof(Pool<GAgent>));
	cudaMemcpy(poolDevice, poolHost, sizeof(Pool<GAgent>), cudaMemcpyHostToDevice);

	int initGridSize = GRID_SIZE(numElemMax), stepGridSize = GRID_SIZE(numElem);
	poolUtil::initPool<<<initGridSize, BLOCK_SIZE>>>(poolDevice);
	init<<<stepGridSize, BLOCK_SIZE>>>(poolDevice);

	for (int i =0; i < round; i++) {
		printf("step: %d\n", i);
		stepGridSize = GRID_SIZE(poolHost->numElem);
		step<<<stepGridSize, BLOCK_SIZE>>>(poolDevice, i, resDev);
		cudaMemcpy(res, resDev, poolHost->numElem * sizeof(int), cudaMemcpyDeviceToHost);
		for (unsigned int j = 0; j < poolHost->numElem; j++)
			printf("%d\t", res[j]);
		printf("\nnumElem: %d\n", poolHost->numElem);
		poolUtil::cleanup(poolHost, poolDevice);
	}
	system("PAUSE");
	return 0;
}