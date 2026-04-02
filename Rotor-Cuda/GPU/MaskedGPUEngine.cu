#include "MaskedGPUEngine.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstring>

#include "GPUMath.h"
#include "GPUHash.h"

#define CudaSafeCall(err) __cudaSafeCall((err), __FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char* file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

namespace {

struct DevicePoint {
    uint64_t x[4];
    uint64_t y[4];
    uint64_t z[4];
    uint8_t set;
};

struct MaskedGPUHotConfig {
    uint32_t suffixLen;
    uint32_t target[5];
    uint8_t radices[MASKED_GPU_MAX_SUFFIX];
    uint8_t bound[MASKED_GPU_MAX_SUFFIX];
    uint8_t minValue[MASKED_GPU_MAX_SUFFIX];
    uint8_t maxValue[MASKED_GPU_MAX_SUFFIX];
    uint8_t posFlags[MASKED_GPU_MAX_SUFFIX];
    uint8_t nonZeroPossibleFromPos[MASKED_GPU_MAX_SUFFIX + 1];
    uint8_t values[MASKED_GPU_MAX_SUFFIX][MASKED_GPU_MAX_CHOICES];
    uint16_t invalidNextMask[MASKED_GPU_RULE_DIM][MASKED_GPU_RULE_DIM];
};

__device__ __constant__ MaskedGPUHotConfig c_maskedHotCfg;
__device__ __constant__ MaskedGPUTask c_maskedTask;

__device__ __forceinline__ void Copy256(uint64_t* dst, const uint64_t* src) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
}

__device__ __forceinline__ bool IsOne256(const uint64_t* value) {
    return value[0] == 1ULL && value[1] == 0ULL && value[2] == 0ULL && value[3] == 0ULL;
}

__device__ __forceinline__ bool IsGeP(const uint64_t* value) {
    if (value[3] != 0xFFFFFFFFFFFFFFFFULL) return value[3] > 0xFFFFFFFFFFFFFFFFULL;
    if (value[2] != 0xFFFFFFFFFFFFFFFFULL) return value[2] > 0xFFFFFFFFFFFFFFFFULL;
    if (value[1] != 0xFFFFFFFFFFFFFFFFULL) return value[1] > 0xFFFFFFFFFFFFFFFFULL;
    return value[0] >= 0xFFFFFFFEFFFFFC2FULL;
}

__device__ __forceinline__ void SubP256(uint64_t* value) {
    USUBO1(value[0], 0xFFFFFFFEFFFFFC2FULL);
    USUBC1(value[1], 0xFFFFFFFFFFFFFFFFULL);
    USUBC1(value[2], 0xFFFFFFFFFFFFFFFFULL);
    USUB1(value[3], 0xFFFFFFFFFFFFFFFFULL);
}

__device__ __forceinline__ void ModDouble256(uint64_t* result, const uint64_t* value) {
    uint64_t carry;
    UADDO(result[0], value[0], value[0]);
    UADDC(result[1], value[1], value[1]);
    UADDC(result[2], value[2], value[2]);
    UADDC(result[3], value[3], value[3]);
    UADD(carry, 0ULL, 0ULL);
    if (carry || IsGeP(result)) {
        SubP256(result);
    }
}

__device__ __forceinline__ void SetAffine(DevicePoint* point, const uint64_t* x, const uint64_t* y) {
    Copy256(point->x, x);
    Copy256(point->y, y);
    point->z[0] = 1ULL;
    point->z[1] = 0ULL;
    point->z[2] = 0ULL;
    point->z[3] = 0ULL;
    point->set = 1;
}

__device__ __noinline__ void ReducePoint(DevicePoint* point) {
    if (!point->set || IsOne256(point->z)) {
        return;
    }

    uint64_t inv320[5];
    inv320[0] = point->z[0];
    inv320[1] = point->z[1];
    inv320[2] = point->z[2];
    inv320[3] = point->z[3];
    inv320[4] = 0ULL;
    _ModInv(inv320);

    uint64_t inv[4];
    inv[0] = inv320[0];
    inv[1] = inv320[1];
    inv[2] = inv320[2];
    inv[3] = inv320[3];

    _ModMult(point->x, inv);
    _ModMult(point->y, inv);
    point->z[0] = 1ULL;
    point->z[1] = 0ULL;
    point->z[2] = 0ULL;
    point->z[3] = 0ULL;
}

__device__ __noinline__ void AddAffine(DevicePoint* point, const uint64_t* x2, const uint64_t* y2) {
    if (!point->set) {
        SetAffine(point, x2, y2);
        return;
    }

    uint64_t ax[4];
    uint64_t ay[4];
    uint64_t u1[4];
    uint64_t v1[4];
    uint64_t u[4];
    uint64_t v[4];
    uint64_t us2[4];
    uint64_t vs2[4];
    uint64_t vs3[4];
    uint64_t us2w[4];
    uint64_t vs2v2[4];
    uint64_t two_vs2v2[4];
    uint64_t a[4];
    uint64_t vs3u2[4];

    Copy256(ax, x2);
    Copy256(ay, y2);

    _ModMult(u1, ay, point->z);
    _ModMult(v1, ax, point->z);
    ModSub256(u, u1, point->y);
    ModSub256(v, v1, point->x);
    _ModSqr(us2, u);
    _ModSqr(vs2, v);
    _ModMult(vs3, vs2, v);
    _ModMult(us2w, us2, point->z);
    _ModMult(vs2v2, vs2, point->x);
    ModDouble256(two_vs2v2, vs2v2);

    Copy256(a, us2w);
    ModSub256(a, vs3);
    ModSub256(a, two_vs2v2);

    _ModMult(point->x, v, a);
    _ModMult(vs3u2, vs3, point->y);
    ModSub256(point->y, vs2v2, a);
    _ModMult(point->y, u);
    ModSub256(point->y, vs3u2);
    _ModMult(point->z, vs3, point->z);
    point->set = 1;
}

__device__ __forceinline__ bool MatchHash160(const uint32_t* candidate) {
    return candidate[0] == c_maskedHotCfg.target[0] &&
           candidate[1] == c_maskedHotCfg.target[1] &&
           candidate[2] == c_maskedHotCfg.target[2] &&
           candidate[3] == c_maskedHotCfg.target[3] &&
           candidate[4] == c_maskedHotCfg.target[4];
}

__device__ __forceinline__ uint16_t GetInvalidNextMask(int last2, int last1) {
    return c_maskedHotCfg.invalidNextMask[last2 + 1][last1 + 1];
}

__device__ __forceinline__ int PointTableIndex(int pos, int slot) {
    return (((pos * MASKED_GPU_MAX_CHOICES) + slot) * 4);
}

__device__ __forceinline__ void LoadPointXY(const uint64_t* pointXTable,
                                            const uint64_t* pointYTable,
                                            int pos,
                                            int slot,
                                            uint64_t* x,
                                            uint64_t* y) {
    const int idx = PointTableIndex(pos, slot);
    x[0] = __ldg(pointXTable + idx + 0);
    x[1] = __ldg(pointXTable + idx + 1);
    x[2] = __ldg(pointXTable + idx + 2);
    x[3] = __ldg(pointXTable + idx + 3);
    y[0] = __ldg(pointYTable + idx + 0);
    y[1] = __ldg(pointYTable + idx + 1);
    y[2] = __ldg(pointYTable + idx + 2);
    y[3] = __ldg(pointYTable + idx + 3);
}

__device__ __forceinline__ void AddChoicePoint(DevicePoint* point,
                                               const uint64_t* pointXTable,
                                               const uint64_t* pointYTable,
                                               int pos,
                                               int slot) {
    uint64_t px[4];
    uint64_t py[4];
    LoadPointXY(pointXTable, pointYTable, pos, slot, px, py);
    AddAffine(point, px, py);
}

template<int CoinType, int CompMode>
__device__ __noinline__ bool MatchCandidate(DevicePoint* point, uint32_t* matchMode) {
    uint32_t h[5];

    if (CoinType == COIN_ETH) {
        _GetHashKeccak160(point->x, point->y, h);
        if (MatchHash160(h)) {
            *matchMode = 0U;
            return true;
        }
        return false;
    }

    if (CompMode == SEARCH_COMPRESSED) {
        _GetHash160Comp(point->x, (uint8_t)(point->y[0] & 1ULL), (uint8_t*)h);
        if (MatchHash160(h)) {
            *matchMode = 1U;
            return true;
        }
        return false;
    }

    if (CompMode == SEARCH_UNCOMPRESSED) {
        _GetHash160(point->x, point->y, (uint8_t*)h);
        if (MatchHash160(h)) {
            *matchMode = 0U;
            return true;
        }
        return false;
    }

    _GetHash160Comp(point->x, (uint8_t)(point->y[0] & 1ULL), (uint8_t*)h);
    if (MatchHash160(h)) {
        *matchMode = 1U;
        return true;
    }

    _GetHash160(point->x, point->y, (uint8_t*)h);
    if (MatchHash160(h)) {
        *matchMode = 0U;
        return true;
    }

    return false;
}

template<int CoinType, int CompMode>
__global__ void masked_search_kernel(uint32_t* foundCount,
                                     MaskedGPUHit* foundHits,
                                     uint32_t maxFound,
                                     const uint64_t* pointXTable,
                                     const uint64_t* pointYTable) {
    const uint64_t globalIdx = c_maskedTask.batchStart +
        (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x);

    if (globalIdx >= c_maskedTask.batchStart + c_maskedTask.batchCount) {
        return;
    }

    if (c_maskedTask.hasNonZero == 0U &&
        c_maskedHotCfg.nonZeroPossibleFromPos[c_maskedTask.startPos] == 0U) {
        return;
    }

    uint8_t choiceSlots[MASKED_GPU_MAX_SUFFIX];
    uint64_t decode = globalIdx;
    for (int pos = (int)c_maskedHotCfg.suffixLen - 1; pos >= (int)c_maskedTask.startPos; --pos) {
        const uint8_t radix = c_maskedHotCfg.radices[pos];
        choiceSlots[pos] = (uint8_t)(decode % (uint64_t)radix);
        decode /= (uint64_t)radix;
    }

    DevicePoint point;
    point.set = c_maskedTask.pointSet;
    Copy256(point.x, c_maskedTask.baseX);
    Copy256(point.y, c_maskedTask.baseY);
    Copy256(point.z, c_maskedTask.baseZ);

    int last1 = (int)c_maskedTask.last1;
    int last2 = (int)c_maskedTask.last2;
    int cmpState = (int)c_maskedTask.cmpState;
    bool hasNonZero = c_maskedTask.hasNonZero != 0U;

    for (int pos = (int)c_maskedTask.startPos; pos < (int)c_maskedHotCfg.suffixLen; ++pos) {
        const uint8_t slot = choiceSlots[pos];
        const uint8_t value = c_maskedHotCfg.values[pos][slot];
        const uint16_t invalidMask = GetInvalidNextMask(last2, last1);
        if ((invalidMask & (uint16_t)(1U << value)) != 0U) {
            return;
        }

        const uint8_t posFlags = c_maskedHotCfg.posFlags[pos];
        if (cmpState == 0) {
            if ((posFlags & MASKED_GPU_POS_ALL_GT_BOUND) != 0U) {
                return;
            }
            if ((posFlags & MASKED_GPU_POS_ALL_LT_BOUND) != 0U) {
                cmpState = -1;
            }
            else if ((posFlags & MASKED_GPU_POS_ALL_EQ_BOUND) == 0U) {
                const uint8_t bound = c_maskedHotCfg.bound[pos];
                if (value > bound) {
                    return;
                }
                if (value < bound) {
                    cmpState = -1;
                }
            }
        }

        last2 = last1;
        last1 = (int)value;

        if ((posFlags & MASKED_GPU_POS_HAS_NONZERO) != 0U) {
            if ((posFlags & MASKED_GPU_POS_HAS_ZERO) == 0U) {
                AddChoicePoint(&point, pointXTable, pointYTable, pos, slot);
                hasNonZero = true;
            }
            else if (value != 0U) {
                AddChoicePoint(&point, pointXTable, pointYTable, pos, slot);
                hasNonZero = true;
            }
        }
    }

    if (!hasNonZero || !point.set) {
        return;
    }

    ReducePoint(&point);

    uint32_t matchMode = 0U;
    if (!MatchCandidate<CoinType, CompMode>(&point, &matchMode)) {
        return;
    }

    uint32_t pos = atomicAdd(foundCount, 1U);
    if (pos < maxFound) {
        foundHits[pos].localIndex = globalIdx - c_maskedTask.batchStart;
        foundHits[pos].mode = matchMode;
        foundHits[pos].reserved = 0U;
    }
}

int ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x20, 32}, {0x21, 48}, {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
        {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64}, {0x61, 128}, {0x62, 128},
        {0x70, 64}, {0x72, 64}, {0x75, 64}, {0x80, 64}, {0x86, 128}, {-1, -1}
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

} // namespace

MaskedGPUEngine::MaskedGPUEngine(int gpuId,
                                 int nbThreadGroup,
                                 int nbThreadPerGroup,
                                 uint32_t maxFound,
                                 const MaskedGPUCharsetConfig& config)
    : gpuId_(gpuId)
    , nbThreadGroup_(nbThreadGroup)
    , nbThreadPerGroup_(nbThreadPerGroup > 0 ? nbThreadPerGroup : 128)
    , nbThread_(0)
    , maxFound_(maxFound)
    , initialised_(false)
    , deviceName_()
    , outputCount_(NULL)
    , outputCountPinned_(NULL)
    , outputHits_(NULL)
    , outputHitsPinned_(NULL)
    , pointX_(NULL)
    , pointY_(NULL)
    , compMode_(config.compMode)
    , coinType_(config.coinType)
    , stream_(NULL)
    , kernelDone_(NULL) {

    int deviceCount = 0;
    CudaSafeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("MaskedGPUEngine: There are no available device(s) that support CUDA\n");
        return;
    }

    CudaSafeCall(cudaSetDevice(gpuId_));

    cudaDeviceProp deviceProp;
    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId_));

    if (nbThreadGroup_ <= 0) {
        nbThreadGroup_ = deviceProp.multiProcessorCount * 8;
    }
    nbThread_ = nbThreadGroup_ * nbThreadPerGroup_;

    char tmp[512];
    sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
        gpuId_, deviceProp.name, deviceProp.multiProcessorCount,
        ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
        nbThreadGroup_, nbThreadPerGroup_);
    deviceName_ = std::string(tmp);

    CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    CudaSafeCall(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CudaSafeCall(cudaEventCreateWithFlags(&kernelDone_, cudaEventDisableTiming));

    CudaSafeCall(cudaMalloc((void**)&outputCount_, sizeof(uint32_t)));
    CudaSafeCall(cudaHostAlloc(&outputCountPinned_, sizeof(uint32_t), cudaHostAllocDefault));
    CudaSafeCall(cudaMalloc((void**)&outputHits_, sizeof(MaskedGPUHit) * maxFound_));
    CudaSafeCall(cudaHostAlloc(&outputHitsPinned_, sizeof(MaskedGPUHit) * maxFound_, cudaHostAllocDefault));

    const size_t pointTableWords = (size_t)MASKED_GPU_MAX_SUFFIX * (size_t)MASKED_GPU_MAX_CHOICES * 4U;
    const size_t pointBytes = pointTableWords * sizeof(uint64_t);
    CudaSafeCall(cudaMalloc((void**)&pointX_, pointBytes));
    CudaSafeCall(cudaMalloc((void**)&pointY_, pointBytes));
    CudaSafeCall(cudaMemcpyAsync(pointX_, &config.pointX[0][0][0], pointBytes, cudaMemcpyHostToDevice, stream_));
    CudaSafeCall(cudaMemcpyAsync(pointY_, &config.pointY[0][0][0], pointBytes, cudaMemcpyHostToDevice, stream_));

    MaskedGPUHotConfig hot = {};
    hot.suffixLen = config.suffixLen;
    std::memcpy(hot.target, config.target, sizeof(hot.target));
    std::memcpy(hot.radices, config.radices, sizeof(hot.radices));
    std::memcpy(hot.bound, config.bound, sizeof(hot.bound));
    std::memcpy(hot.minValue, config.minValue, sizeof(hot.minValue));
    std::memcpy(hot.maxValue, config.maxValue, sizeof(hot.maxValue));
    std::memcpy(hot.posFlags, config.posFlags, sizeof(hot.posFlags));
    std::memcpy(hot.nonZeroPossibleFromPos, config.nonZeroPossibleFromPos, sizeof(hot.nonZeroPossibleFromPos));
    std::memcpy(hot.values, config.values, sizeof(hot.values));
    std::memcpy(hot.invalidNextMask, config.invalidNextMask, sizeof(hot.invalidNextMask));

    CudaSafeCall(cudaMemcpyToSymbol(c_maskedHotCfg, &hot, sizeof(MaskedGPUHotConfig)));
    CudaSafeCall(cudaStreamSynchronize(stream_));
    CudaSafeCall(cudaGetLastError());
    initialised_ = true;
}

MaskedGPUEngine::~MaskedGPUEngine() {
    if (kernelDone_ != NULL) {
        CudaSafeCall(cudaEventDestroy(kernelDone_));
    }
    if (stream_ != NULL) {
        CudaSafeCall(cudaStreamDestroy(stream_));
    }
    if (pointY_ != NULL) {
        CudaSafeCall(cudaFree(pointY_));
    }
    if (pointX_ != NULL) {
        CudaSafeCall(cudaFree(pointX_));
    }
    if (outputHitsPinned_ != NULL) {
        CudaSafeCall(cudaFreeHost(outputHitsPinned_));
    }
    if (outputHits_ != NULL) {
        CudaSafeCall(cudaFree(outputHits_));
    }
    if (outputCountPinned_ != NULL) {
        CudaSafeCall(cudaFreeHost(outputCountPinned_));
    }
    if (outputCount_ != NULL) {
        CudaSafeCall(cudaFree(outputCount_));
    }
}

bool MaskedGPUEngine::IsInitialised() const {
    return initialised_;
}

int MaskedGPUEngine::GetNbThread() const {
    return nbThread_;
}

int MaskedGPUEngine::GetGroupSize() const {
    return nbThreadPerGroup_;
}

const std::string& MaskedGPUEngine::GetDeviceName() const {
    return deviceName_;
}

bool MaskedGPUEngine::SearchBatch(const MaskedGPUTask& task, std::vector<MaskedGPUHit>& hits) {
    hits.clear();
    if (!initialised_) {
        return false;
    }

    CudaSafeCall(cudaMemcpyToSymbolAsync(c_maskedTask, &task, sizeof(MaskedGPUTask), 0, cudaMemcpyHostToDevice, stream_));
    CudaSafeCall(cudaMemsetAsync(outputCount_, 0, sizeof(uint32_t), stream_));

    if (coinType_ == COIN_ETH) {
        masked_search_kernel<COIN_ETH, SEARCH_COMPRESSED><<<nbThreadGroup_, nbThreadPerGroup_, 0, stream_>>>(outputCount_, outputHits_, maxFound_, pointX_, pointY_);
    }
    else if (compMode_ == SEARCH_COMPRESSED) {
        masked_search_kernel<COIN_BTC, SEARCH_COMPRESSED><<<nbThreadGroup_, nbThreadPerGroup_, 0, stream_>>>(outputCount_, outputHits_, maxFound_, pointX_, pointY_);
    }
    else if (compMode_ == SEARCH_UNCOMPRESSED) {
        masked_search_kernel<COIN_BTC, SEARCH_UNCOMPRESSED><<<nbThreadGroup_, nbThreadPerGroup_, 0, stream_>>>(outputCount_, outputHits_, maxFound_, pointX_, pointY_);
    }
    else {
        masked_search_kernel<COIN_BTC, SEARCH_BOTH><<<nbThreadGroup_, nbThreadPerGroup_, 0, stream_>>>(outputCount_, outputHits_, maxFound_, pointX_, pointY_);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("MaskedGPUEngine: Kernel: %s\n", cudaGetErrorString(err));
        return false;
    }

    CudaSafeCall(cudaEventRecord(kernelDone_, stream_));
    CudaSafeCall(cudaEventSynchronize(kernelDone_));

    CudaSafeCall(cudaMemcpyAsync(outputCountPinned_, outputCount_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
    CudaSafeCall(cudaStreamSynchronize(stream_));
    uint32_t count = *outputCountPinned_;
    if (count > maxFound_) {
        count = maxFound_;
    }
    if (count > 0) {
        CudaSafeCall(cudaMemcpyAsync(outputHitsPinned_, outputHits_, sizeof(MaskedGPUHit) * count, cudaMemcpyDeviceToHost, stream_));
        CudaSafeCall(cudaStreamSynchronize(stream_));
        hits.assign(outputHitsPinned_, outputHitsPinned_ + count);
    }
    return true;
}
