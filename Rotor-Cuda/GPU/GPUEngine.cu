/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cstdlib>
#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"

#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUBase58.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
	return;
}

// ---------------------------------------------------------------------------------------

namespace {

const uint32_t kDefaultBatchSteps = 8;
const uint32_t kMaxBatchSteps = 64;

uint32_t ResolveBatchSteps() {
	const char* env = getenv("ROTOR_GPU_BATCH_STEPS");
	if (env == NULL || *env == 0) {
		return kDefaultBatchSteps;
	}

	char* end = NULL;
	unsigned long parsed = strtoul(env, &end, 10);
	if (end == env || parsed == 0UL) {
		return kDefaultBatchSteps;
	}
	if (parsed > kMaxBatchSteps) {
		parsed = kMaxBatchSteps;
	}
	return (uint32_t)parsed;
}

uint32_t ResolveDeviceMaxFound(uint32_t hostMaxFound, uint32_t batchSteps) {
	uint64_t scaled = (uint64_t)hostMaxFound * (uint64_t)batchSteps;
	const uint64_t kMaxBufferedHits = 1024ULL * 1024ULL;
	if (scaled == 0) {
		scaled = hostMaxFound;
	}
	if (scaled > kMaxBufferedHits) {
		scaled = kMaxBufferedHits;
	}
	return (uint32_t)scaled;
}

void WaitForKernel(cudaEvent_t evt, bool spinWait) {
	if (spinWait) {
		CudaSafeCall(cudaEventSynchronize(evt));
		return;
	}

	while (true) {
		cudaError_t status = cudaEventQuery(evt);
		if (status == cudaSuccess) {
			return;
		}
		if (status != cudaErrorNotReady) {
			CudaSafeCall(status);
		}
		Timer::SleepMillis(1);
	}
}

} // namespace

// ---------------------------------------------------------------------------------------

// mode multiple addresses
__global__ void compute_keys_mode_ma(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES,
	uint64_t* keys, uint32_t batchSteps, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	for (uint32_t batch = 0; batch < batchSteps; ++batch) {
		ComputeKeysSEARCH_MODE_MA(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
			(int32_t)(batch * STEP_SIZE), maxFound, found);
	}

}

__global__ void compute_keys_comp_mode_ma(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t batchSteps, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	for (uint32_t batch = 0; batch < batchSteps; ++batch) {
		ComputeKeysSEARCH_MODE_MA(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
			(int32_t)(batch * STEP_SIZE), maxFound, found);
	}

}

// mode single address
__global__ void compute_keys_mode_sa(uint32_t mode, uint32_t* hash160, uint64_t* keys, uint32_t batchSteps, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	for (uint32_t batch = 0; batch < batchSteps; ++batch) {
		ComputeKeysSEARCH_MODE_SA(mode, keys + xPtr, keys + yPtr, hash160,
			(int32_t)(batch * STEP_SIZE), maxFound, found);
	}

}

__global__ void compute_keys_comp_mode_sa(uint32_t mode, uint32_t* hash160, uint64_t* keys, uint32_t batchSteps, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	for (uint32_t batch = 0; batch < batchSteps; ++batch) {
		ComputeKeysSEARCH_MODE_SA(mode, keys + xPtr, keys + yPtr, hash160,
			(int32_t)(batch * STEP_SIZE), maxFound, found);
	}

}

// mode multiple x points
__global__ void compute_keys_comp_mode_mx(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t batchSteps, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	for (uint32_t batch = 0; batch < batchSteps; ++batch) {
		ComputeKeysSEARCH_MODE_MX(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
			(int32_t)(batch * STEP_SIZE), maxFound, found);
	}

}

// mode single x point
__global__ void compute_keys_comp_mode_sx(uint32_t mode, uint32_t* xpoint, uint64_t* keys, uint32_t batchSteps, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	for (uint32_t batch = 0; batch < batchSteps; ++batch) {
		ComputeKeysSEARCH_MODE_SX(mode, keys + xPtr, keys + yPtr, xpoint,
			(int32_t)(batch * STEP_SIZE), maxFound, found);
	}

}

// ---------------------------------------------------------------------------------------
// ethereum

__global__ void compute_keys_mode_eth_ma(uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t batchSteps, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	for (uint32_t batch = 0; batch < batchSteps; ++batch) {
		ComputeKeysSEARCH_ETH_MODE_MA(keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES,
			(int32_t)(batch * STEP_SIZE), maxFound, found);
	}

}

__global__ void compute_keys_mode_eth_sa(uint32_t* hash, uint64_t* keys, uint32_t batchSteps, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	for (uint32_t batch = 0; batch < batchSteps; ++batch) {
		ComputeKeysSEARCH_ETH_MODE_SA(keys + xPtr, keys + yPtr, hash,
			(int32_t)(batch * STEP_SIZE), maxFound, found);
	}

}

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major, int minor)
{

	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
		{0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
		{0x30, 192},
		{0x32, 192},
		{0x35, 192},
		{0x37, 192},
		{0x50, 128},
		{0x52, 128},
		{0x53, 128},
		{0x60,  64},
		{0x61, 128},
		{0x62, 128},
		{0x70,  64},
		{0x72,  64},
		{0x75,  64},
		{0x80,  64},
		{0x86, 128},
		{-1, -1}
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	return 0;

}

// ----------------------------------------------------------------------------

GPUEngine::GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
	int searchMode, int compMode, int coinType, int64_t BLOOM_SIZE, uint64_t BLOOM_BITS,
	uint8_t BLOOM_HASHES, const uint8_t* BLOOM_DATA, uint8_t* DATA, uint64_t TOTAL_COUNT, bool rKey)
{

	// Initialise CUDA
	inputHashORxpoint = NULL;
	inputHashORxpointPinned = NULL;
	inputBloomLookUp = NULL;
	inputBloomLookUpPinned = NULL;
	inputKey = NULL;
	inputKeyPinned = NULL;
	outputBuffer = NULL;
	outputBufferPinned = NULL;
	__2Gnx = NULL;
	__2Gny = NULL;
	_Gx = NULL;
	_Gy = NULL;
	kernelStream = NULL;
	kernelCompletedEvent = NULL;

	this->nbThreadPerGroup = nbThreadPerGroup;
	this->searchMode = searchMode;
	this->compMode = compMode;
	this->coinType = coinType;
	this->rKey = rKey;
	this->batchSteps = ResolveBatchSteps();
	this->batchKeyCount = (uint64_t)STEP_SIZE * (uint64_t)this->batchSteps;
	this->maxFound = maxFound;
	this->deviceMaxFound = ResolveDeviceMaxFound(maxFound, this->batchSteps);

	this->BLOOM_SIZE = BLOOM_SIZE;
	this->BLOOM_BITS = BLOOM_BITS;
	this->BLOOM_HASHES = BLOOM_HASHES;
	this->DATA = DATA;
	this->TOTAL_COUNT = TOTAL_COUNT;

	initialised = false;

	int deviceCount = 0;
	CudaSafeCall(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	CudaSafeCall(cudaSetDevice(gpuId));

	cudaDeviceProp deviceProp;
	CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId));

	if (nbThreadGroup == -1)
		nbThreadGroup = deviceProp.multiProcessorCount * 8;

	this->nbThread = nbThreadGroup * nbThreadPerGroup;
	this->outputSize = (deviceMaxFound * ITEM_SIZE_A + 4);
	if (this->searchMode == (int)SEARCH_MODE_MX)
		this->outputSize = (deviceMaxFound * ITEM_SIZE_X + 4);

	char tmp[512];
	sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d) Batch x%u",
		gpuId, deviceProp.name, deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		nbThread / nbThreadPerGroup,
		nbThreadPerGroup,
		this->batchSteps);
	deviceName = std::string(tmp);

	// Prefer L1 (We do not use __shared__ at all)
	CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	size_t stackSize = 49152;
	CudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));
	CudaSafeCall(cudaStreamCreateWithFlags(&kernelStream, cudaStreamNonBlocking));
	CudaSafeCall(cudaEventCreateWithFlags(&kernelCompletedEvent, cudaEventDisableTiming));

	// Allocate memory
	CudaSafeCall(cudaMalloc((void**)&inputKey, nbThread * 32 * 2));
	CudaSafeCall(cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CudaSafeCall(cudaMalloc((void**)&outputBuffer, outputSize));
	CudaSafeCall(cudaHostAlloc(&outputBufferPinned, outputSize, cudaHostAllocDefault));

	CudaSafeCall(cudaMalloc((void**)&inputBloomLookUp, BLOOM_SIZE));
	CudaSafeCall(cudaHostAlloc(&inputBloomLookUpPinned, BLOOM_SIZE, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	memcpy(inputBloomLookUpPinned, BLOOM_DATA, BLOOM_SIZE);

	CudaSafeCall(cudaMemcpyAsync(inputBloomLookUp, inputBloomLookUpPinned, BLOOM_SIZE, cudaMemcpyHostToDevice, kernelStream));
	CudaSafeCall(cudaStreamSynchronize(kernelStream));
	CudaSafeCall(cudaFreeHost(inputBloomLookUpPinned));
	inputBloomLookUpPinned = NULL;

	// generator table
	InitGenratorTable(secp);


	CudaSafeCall(cudaGetLastError());

	this->compMode = SEARCH_COMPRESSED;
	initialised = true;

}

// ----------------------------------------------------------------------------

GPUEngine::GPUEngine(Secp256K1* secp, int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
	int searchMode, int compMode, int coinType, const uint32_t* hashORxpoint, bool rKey)
{

	// Initialise CUDA
	inputHashORxpoint = NULL;
	inputHashORxpointPinned = NULL;
	inputBloomLookUp = NULL;
	inputBloomLookUpPinned = NULL;
	inputKey = NULL;
	inputKeyPinned = NULL;
	outputBuffer = NULL;
	outputBufferPinned = NULL;
	__2Gnx = NULL;
	__2Gny = NULL;
	_Gx = NULL;
	_Gy = NULL;
	kernelStream = NULL;
	kernelCompletedEvent = NULL;

	this->nbThreadPerGroup = nbThreadPerGroup;
	this->searchMode = searchMode;
	this->compMode = compMode;
	this->coinType = coinType;
	this->rKey = rKey;
	this->batchSteps = ResolveBatchSteps();
	this->batchKeyCount = (uint64_t)STEP_SIZE * (uint64_t)this->batchSteps;
	this->maxFound = maxFound;
	this->deviceMaxFound = ResolveDeviceMaxFound(maxFound, this->batchSteps);

	initialised = false;

	int deviceCount = 0;
	CudaSafeCall(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	CudaSafeCall(cudaSetDevice(gpuId));

	cudaDeviceProp deviceProp;
	CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId));

	if (nbThreadGroup == -1)
		nbThreadGroup = deviceProp.multiProcessorCount * 8;

	this->nbThread = nbThreadGroup * nbThreadPerGroup;
	this->outputSize = (deviceMaxFound * ITEM_SIZE_A + 4);
	if (this->searchMode == (int)SEARCH_MODE_SX)
		this->outputSize = (deviceMaxFound * ITEM_SIZE_X + 4);

	char tmp[512];
	sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d) Batch x%u",
		gpuId, deviceProp.name, deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		nbThread / nbThreadPerGroup,
		nbThreadPerGroup,
		this->batchSteps);
	deviceName = std::string(tmp);

	// Prefer L1 (We do not use __shared__ at all)
	CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	size_t stackSize = 49152;
	CudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));
	CudaSafeCall(cudaStreamCreateWithFlags(&kernelStream, cudaStreamNonBlocking));
	CudaSafeCall(cudaEventCreateWithFlags(&kernelCompletedEvent, cudaEventDisableTiming));

	// Allocate memory
	CudaSafeCall(cudaMalloc((void**)&inputKey, nbThread * 32 * 2));
	CudaSafeCall(cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CudaSafeCall(cudaMalloc((void**)&outputBuffer, outputSize));
	CudaSafeCall(cudaHostAlloc(&outputBufferPinned, outputSize, cudaHostAllocDefault));

	int K_SIZE = 5;
	if (this->searchMode == (int)SEARCH_MODE_SX)
		K_SIZE = 8;

	CudaSafeCall(cudaMalloc((void**)&inputHashORxpoint, K_SIZE * sizeof(uint32_t)));
	CudaSafeCall(cudaHostAlloc(&inputHashORxpointPinned, K_SIZE * sizeof(uint32_t), cudaHostAllocWriteCombined | cudaHostAllocMapped));

	memcpy(inputHashORxpointPinned, hashORxpoint, K_SIZE * sizeof(uint32_t));

	CudaSafeCall(cudaMemcpyAsync(inputHashORxpoint, inputHashORxpointPinned, K_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice, kernelStream));
	CudaSafeCall(cudaStreamSynchronize(kernelStream));
	CudaSafeCall(cudaFreeHost(inputHashORxpointPinned));
	inputHashORxpointPinned = NULL;

	// generator table
	InitGenratorTable(secp);


	CudaSafeCall(cudaGetLastError());

	this->compMode = SEARCH_COMPRESSED;
	initialised = true;

}

// ----------------------------------------------------------------------------

void GPUEngine::InitGenratorTable(Secp256K1* secp)
{

	// generator table
	uint64_t* _2GnxPinned;
	uint64_t* _2GnyPinned;

	uint64_t* GxPinned;
	uint64_t* GyPinned;

	uint64_t size = (uint64_t)GRP_SIZE;

	CudaSafeCall(cudaMalloc((void**)&__2Gnx, 4 * sizeof(uint64_t)));
	CudaSafeCall(cudaHostAlloc(&_2GnxPinned, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CudaSafeCall(cudaMalloc((void**)&__2Gny, 4 * sizeof(uint64_t)));
	CudaSafeCall(cudaHostAlloc(&_2GnyPinned, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped));

	size_t TSIZE = (size / 2) * 4 * sizeof(uint64_t);
	CudaSafeCall(cudaMalloc((void**)&_Gx, TSIZE));
	CudaSafeCall(cudaHostAlloc(&GxPinned, TSIZE, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CudaSafeCall(cudaMalloc((void**)&_Gy, TSIZE));
	CudaSafeCall(cudaHostAlloc(&GyPinned, TSIZE, cudaHostAllocWriteCombined | cudaHostAllocMapped));


	Point* Gn = new Point[size];
	Point g = secp->G;
	Gn[0] = g;
	g = secp->DoubleDirect(g);
	Gn[1] = g;
	for (int i = 2; i < size; i++) {
		g = secp->AddDirect(g, secp->G);
		Gn[i] = g;
	}
	// _2Gn = CPU_GRP_SIZE*G
	Point _2Gn = secp->DoubleDirect(Gn[size / 2 - 1]);

	int nbDigit = 4;
	for (int i = 0; i < nbDigit; i++) {
		_2GnxPinned[i] = _2Gn.x.bits64[i];
		_2GnyPinned[i] = _2Gn.y.bits64[i];
	}
	for (int i = 0; i < size / 2; i++) {
		for (int j = 0; j < nbDigit; j++) {
			GxPinned[i * nbDigit + j] = Gn[i].x.bits64[j];
			GyPinned[i * nbDigit + j] = Gn[i].y.bits64[j];
		}
	}

	delete[] Gn;

	CudaSafeCall(cudaMemcpy(__2Gnx, _2GnxPinned, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaFreeHost(_2GnxPinned));
	_2GnxPinned = NULL;

	CudaSafeCall(cudaMemcpy(__2Gny, _2GnyPinned, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaFreeHost(_2GnyPinned));
	_2GnyPinned = NULL;

	CudaSafeCall(cudaMemcpy(_Gx, GxPinned, TSIZE, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaFreeHost(GxPinned));
	GxPinned = NULL;

	CudaSafeCall(cudaMemcpy(_Gy, GyPinned, TSIZE, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaFreeHost(GyPinned));
	GyPinned = NULL;

	CudaSafeCall(cudaMemcpyToSymbol(_2Gnx, &__2Gnx, sizeof(uint64_t*)));
	CudaSafeCall(cudaMemcpyToSymbol(_2Gny, &__2Gny, sizeof(uint64_t*)));
	CudaSafeCall(cudaMemcpyToSymbol(Gx, &_Gx, sizeof(uint64_t*)));
	CudaSafeCall(cudaMemcpyToSymbol(Gy, &_Gy, sizeof(uint64_t*)));

}

// ----------------------------------------------------------------------------

int GPUEngine::GetGroupSize()
{
	return GRP_SIZE;
}

// ----------------------------------------------------------------------------

void GPUEngine::PrintCudaInfo()
{
	const char* sComputeMode[] = {
		"Multiple host threads",
		"Only one host thread",
		"No host thread",
		"Multiple process threads",
		"Unknown",
		NULL
	};

	int deviceCount = 0;
	CudaSafeCall(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	for (int i = 0; i < deviceCount; i++) {
		CudaSafeCall(cudaSetDevice(i));
		cudaDeviceProp deviceProp;
		CudaSafeCall(cudaGetDeviceProperties(&deviceProp, i));
		printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
			i, deviceProp.name, deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			deviceProp.major, deviceProp.minor, (double)deviceProp.totalGlobalMem / 1048576.0,
			sComputeMode[deviceProp.computeMode]);
	}
}

// ----------------------------------------------------------------------------

GPUEngine::~GPUEngine()
{
	if (kernelCompletedEvent != NULL)
		CudaSafeCall(cudaEventDestroy(kernelCompletedEvent));
	if (kernelStream != NULL)
		CudaSafeCall(cudaStreamDestroy(kernelStream));

	if (inputKey != NULL)
		CudaSafeCall(cudaFree(inputKey));
	if (searchMode == (int)SEARCH_MODE_MA || searchMode == (int)SEARCH_MODE_MX) {
		if (inputBloomLookUp != NULL)
			CudaSafeCall(cudaFree(inputBloomLookUp));
	}
	else {
		if (inputHashORxpoint != NULL)
			CudaSafeCall(cudaFree(inputHashORxpoint));
	}

	if (outputBufferPinned != NULL)
		CudaSafeCall(cudaFreeHost(outputBufferPinned));
	if (outputBuffer != NULL)
		CudaSafeCall(cudaFree(outputBuffer));

	if (__2Gnx != NULL)
		CudaSafeCall(cudaFree(__2Gnx));
	if (__2Gny != NULL)
		CudaSafeCall(cudaFree(__2Gny));
	if (_Gx != NULL)
		CudaSafeCall(cudaFree(_Gx));
	if (_Gy != NULL)
		CudaSafeCall(cudaFree(_Gy));

	if (inputKeyPinned != NULL)
		CudaSafeCall(cudaFreeHost(inputKeyPinned));
}

// ----------------------------------------------------------------------------

int GPUEngine::GetNbThread()
{
	return nbThread;
}

// ----------------------------------------------------------------------------

uint64_t GPUEngine::GetBatchKeyCount() const
{
	return batchKeyCount;
}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_MA()
{

	// Reset nbFound
	CudaSafeCall(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), kernelStream));

	// Call the kernel (Perform batchSteps * STEP_SIZE keys per thread)
	if (coinType == COIN_BTC) {
		if (compMode == SEARCH_COMPRESSED) {
			compute_keys_comp_mode_ma << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, kernelStream >> >
				(compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, batchSteps, deviceMaxFound, outputBuffer);
		}
		else {
			compute_keys_mode_ma << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, kernelStream >> >
				(compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, batchSteps, deviceMaxFound, outputBuffer);
		}
	}
	else {
		compute_keys_mode_eth_ma << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, kernelStream >> >
			(inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, batchSteps, deviceMaxFound, outputBuffer);
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
		return false;
	}
	CudaSafeCall(cudaEventRecord(kernelCompletedEvent, kernelStream));
	return true;

}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_MX()
{

	// Reset nbFound
	CudaSafeCall(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), kernelStream));

	// Call the kernel (Perform batchSteps * STEP_SIZE keys per thread)
	if (compMode == SEARCH_COMPRESSED) {
		compute_keys_comp_mode_mx << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, kernelStream >> >
			(compMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, batchSteps, deviceMaxFound, outputBuffer);
	}
	else {
		printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
		return false;
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
		return false;
	}
	CudaSafeCall(cudaEventRecord(kernelCompletedEvent, kernelStream));
	return true;
}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_SA()
{

	// Reset nbFound
	CudaSafeCall(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), kernelStream));

	// Call the kernel (Perform batchSteps * STEP_SIZE keys per thread)
	if (coinType == COIN_BTC) {
		if (compMode == SEARCH_COMPRESSED) {
			compute_keys_comp_mode_sa << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, kernelStream >> >
				(compMode, inputHashORxpoint, inputKey, batchSteps, deviceMaxFound, outputBuffer);
		}
		else {
			compute_keys_mode_sa << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, kernelStream >> >
				(compMode, inputHashORxpoint, inputKey, batchSteps, deviceMaxFound, outputBuffer);
		}
	}
	else {
		compute_keys_mode_eth_sa << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, kernelStream >> >
			(inputHashORxpoint, inputKey, batchSteps, deviceMaxFound, outputBuffer);
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
		return false;
	}
	CudaSafeCall(cudaEventRecord(kernelCompletedEvent, kernelStream));
	return true;

}

// ----------------------------------------------------------------------------

bool GPUEngine::callKernelSEARCH_MODE_SX()
{

	// Reset nbFound
	CudaSafeCall(cudaMemsetAsync(outputBuffer, 0, sizeof(uint32_t), kernelStream));

	// Call the kernel (Perform batchSteps * STEP_SIZE keys per thread)
	if (compMode == SEARCH_COMPRESSED) {
		compute_keys_comp_mode_sx << < nbThread / nbThreadPerGroup, nbThreadPerGroup, 0, kernelStream >> >
			(compMode, inputHashORxpoint, inputKey, batchSteps, deviceMaxFound, outputBuffer);
	}
	else {
		printf("GPUEngine: PubKeys search doesn't support uncompressed\n");
		return false;
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
		return false;
	}
	CudaSafeCall(cudaEventRecord(kernelCompletedEvent, kernelStream));
	return true;
}

// ----------------------------------------------------------------------------

bool GPUEngine::SetKeys(Point* p)
{
	// Sets the starting keys for each thread
	// p must contains nbThread public keys
	for (int i = 0; i < nbThread; i += nbThreadPerGroup) {
		for (int j = 0; j < nbThreadPerGroup; j++) {

			inputKeyPinned[8 * i + j + 0 * nbThreadPerGroup] = p[i + j].x.bits64[0];
			inputKeyPinned[8 * i + j + 1 * nbThreadPerGroup] = p[i + j].x.bits64[1];
			inputKeyPinned[8 * i + j + 2 * nbThreadPerGroup] = p[i + j].x.bits64[2];
			inputKeyPinned[8 * i + j + 3 * nbThreadPerGroup] = p[i + j].x.bits64[3];

			inputKeyPinned[8 * i + j + 4 * nbThreadPerGroup] = p[i + j].y.bits64[0];
			inputKeyPinned[8 * i + j + 5 * nbThreadPerGroup] = p[i + j].y.bits64[1];
			inputKeyPinned[8 * i + j + 6 * nbThreadPerGroup] = p[i + j].y.bits64[2];
			inputKeyPinned[8 * i + j + 7 * nbThreadPerGroup] = p[i + j].y.bits64[3];

		}
	}

	// Fill device memory
	if (rKey) {
		CudaSafeCall(cudaMemcpyAsync(inputKey, inputKeyPinned, nbThread * 32 * 2, cudaMemcpyHostToDevice, kernelStream));
	}
	else {
		CudaSafeCall(cudaMemcpy(inputKey, inputKeyPinned, nbThread * 32 * 2, cudaMemcpyHostToDevice));
	}

	if (!rKey) {
		// We do not need the input pinned memory anymore
		CudaSafeCall(cudaFreeHost(inputKeyPinned));
		inputKeyPinned = NULL;
	}

	switch (searchMode) {
	case (int)SEARCH_MODE_MA:
		return callKernelSEARCH_MODE_MA();
		break;
	case (int)SEARCH_MODE_SA:
		return callKernelSEARCH_MODE_SA();
		break;
	case (int)SEARCH_MODE_MX:
		return callKernelSEARCH_MODE_MX();
		break;
	case (int)SEARCH_MODE_SX:
		return callKernelSEARCH_MODE_SX();
		break;
	default:
		return false;
		break;
	}
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_MA(std::vector<ITEM>& dataFound, bool spinWait)
{

	dataFound.clear();

	WaitForKernel(kernelCompletedEvent, spinWait);
	CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, sizeof(uint32_t), cudaMemcpyDeviceToHost, kernelStream));
	CudaSafeCall(cudaStreamSynchronize(kernelStream));

	// Look for data found
	uint32_t nbFound = outputBufferPinned[0];
	if (nbFound > deviceMaxFound) {
		nbFound = deviceMaxFound;
	}

	if (nbFound > 0) {
		CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE_A + sizeof(uint32_t), cudaMemcpyDeviceToHost, kernelStream));
		CudaSafeCall(cudaStreamSynchronize(kernelStream));
	}

	for (uint32_t i = 0; i < nbFound; i++) {

		uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE_A32 + 1);
		uint8_t* hash = (uint8_t*)(itemPtr + 3);
		if (CheckBinary(hash, 20) > 0) {

			ITEM it;
			it.thId = itemPtr[0];
			it.incr = (int32_t)itemPtr[1];
			it.mode = itemPtr[2] != 0U;
			it.hash = (uint8_t*)(itemPtr + 3);
			dataFound.push_back(it);
		}
	}
	return callKernelSEARCH_MODE_MA();
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_SA(std::vector<ITEM>& dataFound, bool spinWait)
{

	dataFound.clear();

	WaitForKernel(kernelCompletedEvent, spinWait);
	CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, sizeof(uint32_t), cudaMemcpyDeviceToHost, kernelStream));
	CudaSafeCall(cudaStreamSynchronize(kernelStream));

	// Look for data found
	uint32_t nbFound = outputBufferPinned[0];
	if (nbFound > deviceMaxFound) {
		nbFound = deviceMaxFound;
	}

	if (nbFound > 0) {
		CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE_A + sizeof(uint32_t), cudaMemcpyDeviceToHost, kernelStream));
		CudaSafeCall(cudaStreamSynchronize(kernelStream));
	}

	for (uint32_t i = 0; i < nbFound; i++) {
		uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE_A32 + 1);
		ITEM it;
		it.thId = itemPtr[0];
		it.incr = (int32_t)itemPtr[1];
		it.mode = itemPtr[2] != 0U;
		it.hash = (uint8_t*)(itemPtr + 3);
		dataFound.push_back(it);
	}
	return callKernelSEARCH_MODE_SA();
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_MX(std::vector<ITEM>& dataFound, bool spinWait)
{

	dataFound.clear();

	WaitForKernel(kernelCompletedEvent, spinWait);
	CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, sizeof(uint32_t), cudaMemcpyDeviceToHost, kernelStream));
	CudaSafeCall(cudaStreamSynchronize(kernelStream));

	// Look for data found
	uint32_t nbFound = outputBufferPinned[0];
	if (nbFound > deviceMaxFound) {
		nbFound = deviceMaxFound;
	}

	if (nbFound > 0) {
		CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE_X + sizeof(uint32_t), cudaMemcpyDeviceToHost, kernelStream));
		CudaSafeCall(cudaStreamSynchronize(kernelStream));
	}

	for (uint32_t i = 0; i < nbFound; i++) {

		uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE_X32 + 1);
		uint8_t* pubkey = (uint8_t*)(itemPtr + 3);

		if (CheckBinary(pubkey, 32) > 0) {

			ITEM it;
			it.thId = itemPtr[0];
			it.incr = (int32_t)itemPtr[1];
			it.mode = itemPtr[2] != 0U;
			it.hash = (uint8_t*)(itemPtr + 3);
			dataFound.push_back(it);
		}
	}
	return callKernelSEARCH_MODE_MX();
}

// ----------------------------------------------------------------------------

bool GPUEngine::LaunchSEARCH_MODE_SX(std::vector<ITEM>& dataFound, bool spinWait)
{

	dataFound.clear();

	WaitForKernel(kernelCompletedEvent, spinWait);
	CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, sizeof(uint32_t), cudaMemcpyDeviceToHost, kernelStream));
	CudaSafeCall(cudaStreamSynchronize(kernelStream));

	// Look for data found
	uint32_t nbFound = outputBufferPinned[0];
	if (nbFound > deviceMaxFound) {
		nbFound = deviceMaxFound;
	}

	if (nbFound > 0) {
		CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE_X + sizeof(uint32_t), cudaMemcpyDeviceToHost, kernelStream));
		CudaSafeCall(cudaStreamSynchronize(kernelStream));
	}

	for (uint32_t i = 0; i < nbFound; i++) {

		uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE_X32 + 1);
		ITEM it;
		it.thId = itemPtr[0];
		it.incr = (int32_t)itemPtr[1];
		it.mode = itemPtr[2] != 0U;
		it.hash = (uint8_t*)(itemPtr + 3);
		dataFound.push_back(it);
	}
	return callKernelSEARCH_MODE_SX();
}

// ----------------------------------------------------------------------------

int GPUEngine::CheckBinary(const uint8_t* _x, int K_LENGTH)
{
	uint8_t* temp_read;
	uint64_t half, min, max, current; //, current_offset
	int64_t rcmp;
	int32_t r = 0;
	min = 0;
	current = 0;
	max = TOTAL_COUNT;
	half = TOTAL_COUNT;
	while (!r && half >= 1) {
		half = (max - min) / 2;
		temp_read = DATA + ((current + half) * K_LENGTH);
		rcmp = memcmp(_x, temp_read, K_LENGTH);
		if (rcmp == 0) {
			r = 1;  //Found!!
		}
		else {
			if (rcmp < 0) { //data < temp_read
				max = (max - half);
			}
			else { // data > temp_read
				min = (min + half);
			}
			current = min;
		}
	}
	return r;
}
