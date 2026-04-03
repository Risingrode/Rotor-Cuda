#include "MaskedGPUEngine.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

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

static const uint16_t kInvalidRuleState = 0xFFFFU;
static const int8_t kInvalidCmpState = 2;
static const uint32_t kDefaultFastBatchSteps = 16U;

struct DevicePoint {
    uint64_t x[4];
    uint64_t y[4];
    uint64_t z[4];
    uint8_t set;
};

struct MaskedGPUHotConfig {
    uint32_t suffixLen;
    uint32_t target[5];
    uint8_t gpuStartPos;
    uint8_t segmentCount;
    uint8_t segmentComboCap;
    uint8_t reserved0;
    uint8_t radices[MASKED_GPU_MAX_SUFFIX];
    uint8_t radixShift[MASKED_GPU_MAX_SUFFIX];
    uint8_t bound[MASKED_GPU_MAX_SUFFIX];
    uint8_t minValue[MASKED_GPU_MAX_SUFFIX];
    uint8_t maxValue[MASKED_GPU_MAX_SUFFIX];
    uint8_t posFlags[MASKED_GPU_MAX_SUFFIX];
    uint8_t nonZeroPossibleFromPos[MASKED_GPU_MAX_SUFFIX + 1];
    uint8_t values[MASKED_GPU_MAX_SUFFIX][MASKED_GPU_MAX_CHOICES];
    uint16_t invalidNextMask[MASKED_GPU_RULE_DIM][MASKED_GPU_RULE_DIM];
    uint8_t segmentStart[MASKED_GPU_MAX_SEGMENTS];
    uint8_t segmentLen[MASKED_GPU_MAX_SEGMENTS];
    uint8_t segmentRadix[MASKED_GPU_MAX_SEGMENTS];
    uint8_t segmentRadixShift[MASKED_GPU_MAX_SEGMENTS];
};

__device__ __constant__ MaskedGPUHotConfig c_maskedHotCfg;

uint32_t NormalizeBatchSteps(uint32_t requested) {
    if (requested == 8U || requested == 16U || requested == 32U) {
        return requested;
    }
    if (requested == 0U) {
        return kDefaultFastBatchSteps;
    }
    if (requested < 8U) {
        return 8U;
    }
    if (requested < 16U) {
        return 16U;
    }
    return 32U;
}

int PackRuleStateHost(int last2, int last1) {
    return ((last2 + 1) * MASKED_GPU_RULE_DIM) + (last1 + 1);
}

int RuleTransitionIndexHost(int segment, int combo, int state) {
    return ((segment * MASKED_GPU_SEGMENT_COMBO_CAP) + combo) * MASKED_GPU_RULE_STATE_COUNT + state;
}

int BoundTransitionIndexHost(int segment, int combo, int cmpState) {
    return ((segment * MASKED_GPU_SEGMENT_COMBO_CAP) + combo) * MASKED_GPU_CMP_STATE_COUNT + (cmpState + 1);
}

__device__ __forceinline__ uint32_t PackRuleStateDevice(int last2, int last1) {
    return (uint32_t)(((last2 + 1) * MASKED_GPU_RULE_DIM) + (last1 + 1));
}

__device__ __forceinline__ int RuleTransitionIndexDevice(int segment, int combo, uint32_t state) {
    return (((segment * MASKED_GPU_SEGMENT_COMBO_CAP) + combo) * MASKED_GPU_RULE_STATE_COUNT) + (int)state;
}

__device__ __forceinline__ int BoundTransitionIndexDevice(int segment, int combo, int cmpState) {
    return (((segment * MASKED_GPU_SEGMENT_COMBO_CAP) + combo) * MASKED_GPU_CMP_STATE_COUNT) + (cmpState + 1);
}

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

__device__ __forceinline__ void AddPWrap256(uint64_t* value) {
    UADDO1(value[0], 0x1000003D1ULL);
    UADDC1(value[1], 0ULL);
    UADDC1(value[2], 0ULL);
    UADD1(value[3], 0ULL);
}

__device__ __forceinline__ void ModDouble256(uint64_t* result, const uint64_t* value) {
    uint64_t carry;
    UADDO(result[0], value[0], value[0]);
    UADDC(result[1], value[1], value[1]);
    UADDC(result[2], value[2], value[2]);
    UADDC(result[3], value[3], value[3]);
    UADD(carry, 0ULL, 0ULL);
    if (carry) {
        AddPWrap256(result);
    }
    if (IsGeP(result)) {
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

__device__ __forceinline__ int SegmentPointTableIndex(int segment, int combo) {
    return (((segment * MASKED_GPU_SEGMENT_COMBO_CAP) + combo) * 4);
}

__device__ __forceinline__ int SegmentValueTableIndex(int segment, int combo, int offset) {
    return (((segment * MASKED_GPU_SEGMENT_COMBO_CAP) + combo) * MASKED_GPU_MAX_SEGMENT_LEN) + offset;
}

__device__ __forceinline__ void LoadSegmentPointXY(const uint64_t* segmentPointXTable,
                                                   const uint64_t* segmentPointYTable,
                                                   int segment,
                                                   int combo,
                                                   uint64_t* x,
                                                   uint64_t* y) {
    const int idx = SegmentPointTableIndex(segment, combo);
    x[0] = __ldg(segmentPointXTable + idx + 0);
    x[1] = __ldg(segmentPointXTable + idx + 1);
    x[2] = __ldg(segmentPointXTable + idx + 2);
    x[3] = __ldg(segmentPointXTable + idx + 3);
    y[0] = __ldg(segmentPointYTable + idx + 0);
    y[1] = __ldg(segmentPointYTable + idx + 1);
    y[2] = __ldg(segmentPointYTable + idx + 2);
    y[3] = __ldg(segmentPointYTable + idx + 3);
}

__device__ __forceinline__ void AddSegmentPoint(DevicePoint* point,
                                                const uint64_t* segmentPointXTable,
                                                const uint64_t* segmentPointYTable,
                                                int segment,
                                                int combo) {
    uint64_t px[4];
    uint64_t py[4];
    LoadSegmentPointXY(segmentPointXTable, segmentPointYTable, segment, combo, px, py);
    AddAffine(point, px, py);
}

__device__ __forceinline__ void DecodeSegmentCombos(uint64_t candidateIndex, uint8_t* segmentCombos) {
    uint64_t decode = candidateIndex;
    for (int segment = (int)c_maskedHotCfg.segmentCount - 1; segment >= 0; --segment) {
        const uint8_t shift = c_maskedHotCfg.segmentRadixShift[segment];
        if (shift != 0xFFU) {
            if (shift == 0U) {
                segmentCombos[segment] = 0U;
            }
            else {
                const uint64_t mask = (1ULL << shift) - 1ULL;
                segmentCombos[segment] = (uint8_t)(decode & mask);
                decode >>= shift;
            }
        }
        else {
            const uint8_t radix = c_maskedHotCfg.segmentRadix[segment];
            segmentCombos[segment] = (uint8_t)(decode % (uint64_t)radix);
            decode /= (uint64_t)radix;
        }
    }
}

__device__ __forceinline__ void InitDevicePoint(DevicePoint* point, const MaskedGPUTask& task) {
    point->set = task.pointSet;
    Copy256(point->x, task.baseX);
    Copy256(point->y, task.baseY);
    Copy256(point->z, task.baseZ);
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

__device__ __forceinline__ bool MatchCandidateBTCUncompressed(DevicePoint* point) {
    uint32_t h[5];
    _GetHash160(point->x, point->y, (uint8_t*)h);
    return MatchHash160(h);
}

template<int CoinType, int CompMode>
__device__ __forceinline__ void ProcessGenericCandidate(const MaskedGPUTask& task,
                                                        uint64_t localIndex,
                                                        uint32_t* foundCount,
                                                        MaskedGPUHit* foundHits,
                                                        uint32_t maxFound,
                                                        const uint8_t* segmentPointSetTable,
                                                        const uint8_t* segmentValueTable,
                                                        const uint64_t* segmentPointXTable,
                                                        const uint64_t* segmentPointYTable) {
    const uint64_t candidateIndex = task.batchStart + localIndex;

    uint8_t segmentCombos[MASKED_GPU_MAX_SEGMENTS];
    DecodeSegmentCombos(candidateIndex, segmentCombos);

    DevicePoint point;
    InitDevicePoint(&point, task);

    int last1 = (int)task.last1;
    int last2 = (int)task.last2;
    int cmpState = (int)task.cmpState;
    bool hasNonZero = task.hasNonZero != 0U;

    for (int segment = 0; segment < (int)c_maskedHotCfg.segmentCount; ++segment) {
        const uint8_t combo = segmentCombos[segment];
        const int startPos = (int)c_maskedHotCfg.segmentStart[segment];
        const int segLen = (int)c_maskedHotCfg.segmentLen[segment];

        for (int offset = 0; offset < segLen; ++offset) {
            const int pos = startPos + offset;
            const uint8_t value = segmentValueTable[SegmentValueTableIndex(segment, combo, offset)];
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
        }

        if (segmentPointSetTable[(segment * MASKED_GPU_SEGMENT_COMBO_CAP) + combo] != 0U) {
            AddSegmentPoint(&point, segmentPointXTable, segmentPointYTable, segment, combo);
            hasNonZero = true;
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
        foundHits[pos].localIndex = localIndex;
        foundHits[pos].mode = matchMode;
        foundHits[pos].reserved = 0U;
    }
}

template<int CoinType, int CompMode>
__global__ void masked_search_kernel(const MaskedGPUTask task,
                                     uint32_t* foundCount,
                                     MaskedGPUHit* foundHits,
                                     uint32_t maxFound,
                                     const uint8_t* segmentPointSetTable,
                                     const uint8_t* segmentValueTable,
                                     const uint64_t* segmentPointXTable,
                                     const uint64_t* segmentPointYTable) {
    if (task.hasNonZero == 0U &&
        c_maskedHotCfg.nonZeroPossibleFromPos[task.startPos] == 0U) {
        return;
    }

    const uint64_t stride = (uint64_t)(gridDim.x * blockDim.x);
    for (uint64_t localIndex = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x);
         localIndex < task.batchCount;
         localIndex += stride) {
        ProcessGenericCandidate<CoinType, CompMode>(task, localIndex, foundCount, foundHits, maxFound,
                                                    segmentPointSetTable, segmentValueTable,
                                                    segmentPointXTable, segmentPointYTable);
    }
}

__global__ void masked_search_kernel_v2_btc_uncompressed(const MaskedGPUTask task,
                                                         uint32_t* foundCount,
                                                         MaskedGPUHit* foundHits,
                                                         uint32_t maxFound,
                                                         const uint8_t* segmentPointSetTable,
                                                         const uint64_t* segmentPointXTable,
                                                         const uint64_t* segmentPointYTable,
                                                         const uint16_t* ruleTransition,
                                                         const int8_t* boundTransition) {
    if (task.hasNonZero == 0U &&
        c_maskedHotCfg.nonZeroPossibleFromPos[task.startPos] == 0U) {
        return;
    }

    const uint64_t stride = (uint64_t)(gridDim.x * blockDim.x);
    for (uint64_t localIndex = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x);
         localIndex < task.batchCount;
         localIndex += stride) {
        const uint64_t candidateIndex = task.batchStart + localIndex;
        uint8_t segmentCombos[MASKED_GPU_MAX_SEGMENTS];
        DecodeSegmentCombos(candidateIndex, segmentCombos);

        DevicePoint point;
        InitDevicePoint(&point, task);

        uint32_t ruleState = PackRuleStateDevice((int)task.last2, (int)task.last1);
        int cmpState = (int)task.cmpState;
        bool hasNonZero = task.hasNonZero != 0U;
        bool valid = true;

        for (int segment = 0; segment < (int)c_maskedHotCfg.segmentCount; ++segment) {
            const int combo = (int)segmentCombos[segment];
            const uint16_t nextRuleState = ruleTransition[RuleTransitionIndexDevice(segment, combo, ruleState)];
            if (nextRuleState == kInvalidRuleState) {
                valid = false;
                break;
            }
            ruleState = (uint32_t)nextRuleState;

            const int8_t nextCmpState = boundTransition[BoundTransitionIndexDevice(segment, combo, cmpState)];
            if (nextCmpState == kInvalidCmpState) {
                valid = false;
                break;
            }
            cmpState = (int)nextCmpState;

            if (segmentPointSetTable[(segment * MASKED_GPU_SEGMENT_COMBO_CAP) + combo] != 0U) {
                AddSegmentPoint(&point, segmentPointXTable, segmentPointYTable, segment, combo);
                hasNonZero = true;
            }
        }

        if (!valid || !hasNonZero || !point.set) {
            continue;
        }

        ReducePoint(&point);
        if (!MatchCandidateBTCUncompressed(&point)) {
            continue;
        }

        const uint32_t pos = atomicAdd(foundCount, 1U);
        if (pos < maxFound) {
            foundHits[pos].localIndex = localIndex;
            foundHits[pos].mode = 0U;
            foundHits[pos].reserved = 0U;
        }
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
                                 const MaskedGPUCharsetConfig& config,
                                 const MaskedGPUFastPathTables& fastTables)
    : gpuId_(gpuId)
    , nbThreadGroup_(nbThreadGroup)
    , nbThreadPerGroup_(nbThreadPerGroup)
    , nbThread_(0)
    , smCount_(0)
    , maxFound_(maxFound)
    , batchSteps_(1)
    , launchCandidateCount_(0)
    , initialised_(false)
    , fastPathV2_(config.gpuMode == MASKED_GPU_MODE_V2_FASTPATH)
    , autotuneEnabled_(config.autotune != 0)
    , requestedBatchSteps_(config.requestedBatchSteps)
    , deviceName_()
    , deviceBaseName_()
    , slotCount_(1)
    , launchCursor_(0)
    , collectCursor_(0)
    , pendingCount_(0)
    , segmentPointSet_(NULL)
    , segmentValues_(NULL)
    , segmentPointX_(NULL)
    , segmentPointY_(NULL)
    , fastRuleTransition_(NULL)
    , fastBoundTransition_(NULL)
    , compMode_(config.compMode)
    , coinType_(config.coinType) {

    for (int i = 0; i < kMaxSlots; i++) {
        outputCount_[i] = NULL;
        outputCountPinned_[i] = NULL;
        outputHits_[i] = NULL;
        outputHitsPinned_[i] = NULL;
        slotActive_[i] = false;
        slotBatchStart_[i] = 0ULL;
        slotBatchCount_[i] = 0ULL;
#ifdef ROTOR_MASKED_CUDA_TYPES_AVAILABLE
        streams_[i] = NULL;
        kernelDone_[i] = NULL;
#endif
    }

    int deviceCount = 0;
    CudaSafeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("MaskedGPUEngine: There are no available device(s) that support CUDA\n");
        return;
    }

    CudaSafeCall(cudaSetDevice(gpuId_));

    cudaDeviceProp deviceProp;
    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId_));
    smCount_ = deviceProp.multiProcessorCount;
    deviceBaseName_ = std::string(deviceProp.name);

    if (nbThreadGroup_ <= 0) {
        nbThreadGroup_ = smCount_ * (fastPathV2_ ? 12 : 8);
    }
    if (nbThreadPerGroup_ <= 0) {
        nbThreadPerGroup_ = fastPathV2_ ? 256 : 128;
    }

    slotCount_ = fastPathV2_ ? 2 : 1;
    ApplyLaunchConfig(nbThreadGroup_, nbThreadPerGroup_, fastPathV2_ ? NormalizeBatchSteps((uint32_t)requestedBatchSteps_) : 1U,
                      deviceProp.major, deviceProp.minor);

    CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    for (int i = 0; i < slotCount_; i++) {
        CudaSafeCall(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
        CudaSafeCall(cudaEventCreateWithFlags(&kernelDone_[i], cudaEventDisableTiming));
        CudaSafeCall(cudaMalloc((void**)&outputCount_[i], sizeof(uint32_t)));
        CudaSafeCall(cudaHostAlloc(&outputCountPinned_[i], sizeof(uint32_t), cudaHostAllocDefault));
        CudaSafeCall(cudaMalloc((void**)&outputHits_[i], sizeof(MaskedGPUHit) * maxFound_));
        CudaSafeCall(cudaHostAlloc(&outputHitsPinned_[i], sizeof(MaskedGPUHit) * maxFound_, cudaHostAllocDefault));
    }

    const size_t segmentPointSetBytes = sizeof(config.segmentPointSet);
    const size_t segmentValueBytes = sizeof(config.segmentValues);
    const size_t segmentPointBytes = sizeof(config.segmentPointX);
    CudaSafeCall(cudaMalloc((void**)&segmentPointSet_, segmentPointSetBytes));
    CudaSafeCall(cudaMalloc((void**)&segmentValues_, segmentValueBytes));
    CudaSafeCall(cudaMalloc((void**)&segmentPointX_, segmentPointBytes));
    CudaSafeCall(cudaMalloc((void**)&segmentPointY_, segmentPointBytes));
    CudaSafeCall(cudaMemcpyAsync(segmentPointSet_, &config.segmentPointSet[0][0], segmentPointSetBytes, cudaMemcpyHostToDevice, streams_[0]));
    CudaSafeCall(cudaMemcpyAsync(segmentValues_, &config.segmentValues[0][0][0], segmentValueBytes, cudaMemcpyHostToDevice, streams_[0]));
    CudaSafeCall(cudaMemcpyAsync(segmentPointX_, &config.segmentPointX[0][0][0], segmentPointBytes, cudaMemcpyHostToDevice, streams_[0]));
    CudaSafeCall(cudaMemcpyAsync(segmentPointY_, &config.segmentPointY[0][0][0], segmentPointBytes, cudaMemcpyHostToDevice, streams_[0]));

    if (fastPathV2_) {
        const size_t expectedRuleCount = (size_t)MASKED_GPU_MAX_SEGMENTS * (size_t)MASKED_GPU_SEGMENT_COMBO_CAP * (size_t)MASKED_GPU_RULE_STATE_COUNT;
        const size_t expectedBoundCount = (size_t)MASKED_GPU_MAX_SEGMENTS * (size_t)MASKED_GPU_SEGMENT_COMBO_CAP * (size_t)MASKED_GPU_CMP_STATE_COUNT;
        if (fastTables.ruleTransition.size() != expectedRuleCount || fastTables.boundTransition.size() != expectedBoundCount) {
            throw std::runtime_error("MaskedGPUEngine fastpath tables are incomplete");
        }
        CudaSafeCall(cudaMalloc((void**)&fastRuleTransition_, sizeof(uint16_t) * expectedRuleCount));
        CudaSafeCall(cudaMalloc((void**)&fastBoundTransition_, sizeof(int8_t) * expectedBoundCount));
        CudaSafeCall(cudaMemcpyAsync(fastRuleTransition_, &fastTables.ruleTransition[0], sizeof(uint16_t) * expectedRuleCount, cudaMemcpyHostToDevice, streams_[0]));
        CudaSafeCall(cudaMemcpyAsync(fastBoundTransition_, &fastTables.boundTransition[0], sizeof(int8_t) * expectedBoundCount, cudaMemcpyHostToDevice, streams_[0]));
    }

    MaskedGPUHotConfig hot = {};
    hot.suffixLen = config.suffixLen;
    std::memcpy(hot.target, config.target, sizeof(hot.target));
    hot.gpuStartPos = config.gpuStartPos;
    hot.segmentCount = config.segmentCount;
    hot.segmentComboCap = config.segmentComboCap;
    std::memcpy(hot.radices, config.radices, sizeof(hot.radices));
    std::memcpy(hot.radixShift, config.radixShift, sizeof(hot.radixShift));
    std::memcpy(hot.bound, config.bound, sizeof(hot.bound));
    std::memcpy(hot.minValue, config.minValue, sizeof(hot.minValue));
    std::memcpy(hot.maxValue, config.maxValue, sizeof(hot.maxValue));
    std::memcpy(hot.posFlags, config.posFlags, sizeof(hot.posFlags));
    std::memcpy(hot.nonZeroPossibleFromPos, config.nonZeroPossibleFromPos, sizeof(hot.nonZeroPossibleFromPos));
    std::memcpy(hot.values, config.values, sizeof(hot.values));
    std::memcpy(hot.invalidNextMask, config.invalidNextMask, sizeof(hot.invalidNextMask));
    std::memcpy(hot.segmentStart, config.segmentStart, sizeof(hot.segmentStart));
    std::memcpy(hot.segmentLen, config.segmentLen, sizeof(hot.segmentLen));
    std::memcpy(hot.segmentRadix, config.segmentRadix, sizeof(hot.segmentRadix));
    std::memcpy(hot.segmentRadixShift, config.segmentRadixShift, sizeof(hot.segmentRadixShift));

    CudaSafeCall(cudaMemcpyToSymbol(c_maskedHotCfg, &hot, sizeof(MaskedGPUHotConfig)));
    CudaSafeCall(cudaStreamSynchronize(streams_[0]));
    CudaSafeCall(cudaGetLastError());
    initialised_ = true;
}

MaskedGPUEngine::~MaskedGPUEngine() {
    for (int i = 0; i < kMaxSlots; i++) {
        if (kernelDone_[i] != NULL) {
            CudaSafeCall(cudaEventDestroy(kernelDone_[i]));
        }
        if (streams_[i] != NULL) {
            CudaSafeCall(cudaStreamDestroy(streams_[i]));
        }
        if (outputHitsPinned_[i] != NULL) {
            CudaSafeCall(cudaFreeHost(outputHitsPinned_[i]));
        }
        if (outputHits_[i] != NULL) {
            CudaSafeCall(cudaFree(outputHits_[i]));
        }
        if (outputCountPinned_[i] != NULL) {
            CudaSafeCall(cudaFreeHost(outputCountPinned_[i]));
        }
        if (outputCount_[i] != NULL) {
            CudaSafeCall(cudaFree(outputCount_[i]));
        }
    }
    if (fastBoundTransition_ != NULL) {
        CudaSafeCall(cudaFree(fastBoundTransition_));
    }
    if (fastRuleTransition_ != NULL) {
        CudaSafeCall(cudaFree(fastRuleTransition_));
    }
    if (segmentPointY_ != NULL) {
        CudaSafeCall(cudaFree(segmentPointY_));
    }
    if (segmentPointX_ != NULL) {
        CudaSafeCall(cudaFree(segmentPointX_));
    }
    if (segmentValues_ != NULL) {
        CudaSafeCall(cudaFree(segmentValues_));
    }
    if (segmentPointSet_ != NULL) {
        CudaSafeCall(cudaFree(segmentPointSet_));
    }
}

void MaskedGPUEngine::UpdateDeviceName(int major, int minor) {
    char tmp[512];
    if (fastPathV2_) {
        sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d) Batch x%u [V2 fastpath]",
            gpuId_, deviceBaseName_.c_str(), smCount_, ConvertSMVer2Cores(major, minor),
            nbThreadGroup_, nbThreadPerGroup_, batchSteps_);
    }
    else {
        sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
            gpuId_, deviceBaseName_.c_str(), smCount_, ConvertSMVer2Cores(major, minor),
            nbThreadGroup_, nbThreadPerGroup_);
    }
    deviceName_ = std::string(tmp);
}

void MaskedGPUEngine::ApplyLaunchConfig(int nbThreadGroup, int nbThreadPerGroup, uint32_t batchSteps, int major, int minor) {
    nbThreadGroup_ = std::max(1, nbThreadGroup);
    nbThreadPerGroup_ = std::max(32, nbThreadPerGroup);
    batchSteps_ = fastPathV2_ ? NormalizeBatchSteps(batchSteps) : 1U;
    nbThread_ = nbThreadGroup_ * nbThreadPerGroup_;
    launchCandidateCount_ = fastPathV2_ ? ((uint64_t)nbThread_ * (uint64_t)batchSteps_) : (uint64_t)nbThread_;
    UpdateDeviceName(major, minor);
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

bool MaskedGPUEngine::IsFastPathV2() const {
    return fastPathV2_;
}

uint32_t MaskedGPUEngine::GetBatchSteps() const {
    return batchSteps_;
}

uint64_t MaskedGPUEngine::GetLaunchCandidateCount() const {
    return launchCandidateCount_;
}

bool MaskedGPUEngine::LaunchOnSlot(int slot, const MaskedGPUTask& task) {
    if (!initialised_) {
        return false;
    }

    cudaStream_t stream = streams_[slot];
    CudaSafeCall(cudaMemsetAsync(outputCount_[slot], 0, sizeof(uint32_t), stream));

    if (fastPathV2_) {
        masked_search_kernel_v2_btc_uncompressed<<<nbThreadGroup_, nbThreadPerGroup_, 0, stream>>>(
            task, outputCount_[slot], outputHits_[slot], maxFound_,
            segmentPointSet_, segmentPointX_, segmentPointY_,
            fastRuleTransition_, fastBoundTransition_);
    }
    else if (coinType_ == COIN_ETH) {
        masked_search_kernel<COIN_ETH, SEARCH_COMPRESSED><<<nbThreadGroup_, nbThreadPerGroup_, 0, stream>>>(
            task, outputCount_[slot], outputHits_[slot], maxFound_,
            segmentPointSet_, segmentValues_, segmentPointX_, segmentPointY_);
    }
    else if (compMode_ == SEARCH_COMPRESSED) {
        masked_search_kernel<COIN_BTC, SEARCH_COMPRESSED><<<nbThreadGroup_, nbThreadPerGroup_, 0, stream>>>(
            task, outputCount_[slot], outputHits_[slot], maxFound_,
            segmentPointSet_, segmentValues_, segmentPointX_, segmentPointY_);
    }
    else if (compMode_ == SEARCH_UNCOMPRESSED) {
        masked_search_kernel<COIN_BTC, SEARCH_UNCOMPRESSED><<<nbThreadGroup_, nbThreadPerGroup_, 0, stream>>>(
            task, outputCount_[slot], outputHits_[slot], maxFound_,
            segmentPointSet_, segmentValues_, segmentPointX_, segmentPointY_);
    }
    else {
        masked_search_kernel<COIN_BTC, SEARCH_BOTH><<<nbThreadGroup_, nbThreadPerGroup_, 0, stream>>>(
            task, outputCount_[slot], outputHits_[slot], maxFound_,
            segmentPointSet_, segmentValues_, segmentPointX_, segmentPointY_);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("MaskedGPUEngine: Kernel: %s\n", cudaGetErrorString(err));
        return false;
    }

    CudaSafeCall(cudaEventRecord(kernelDone_[slot], stream));
    slotBatchStart_[slot] = task.batchStart;
    slotBatchCount_[slot] = task.batchCount;
    return true;
}

bool MaskedGPUEngine::CollectFromSlot(int slot, std::vector<MaskedGPUHit>& hits, uint64_t& batchStart, uint64_t& batchCount, bool wait) {
    hits.clear();
    if (!slotActive_[slot]) {
        return false;
    }

    if (wait) {
        CudaSafeCall(cudaEventSynchronize(kernelDone_[slot]));
    }
    else {
        cudaError_t status = cudaEventQuery(kernelDone_[slot]);
        if (status == cudaErrorNotReady) {
            return false;
        }
        if (status != cudaSuccess) {
            CudaSafeCall(status);
        }
    }

    cudaStream_t stream = streams_[slot];
    CudaSafeCall(cudaMemcpyAsync(outputCountPinned_[slot], outputCount_[slot], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CudaSafeCall(cudaStreamSynchronize(stream));
    uint32_t count = *outputCountPinned_[slot];
    if (count > maxFound_) {
        count = maxFound_;
    }
    if (count > 0) {
        CudaSafeCall(cudaMemcpyAsync(outputHitsPinned_[slot], outputHits_[slot], sizeof(MaskedGPUHit) * count, cudaMemcpyDeviceToHost, stream));
        CudaSafeCall(cudaStreamSynchronize(stream));
        hits.assign(outputHitsPinned_[slot], outputHitsPinned_[slot] + count);
    }

    batchStart = slotBatchStart_[slot];
    batchCount = slotBatchCount_[slot];
    slotActive_[slot] = false;
    slotBatchStart_[slot] = 0ULL;
    slotBatchCount_[slot] = 0ULL;
    return true;
}

bool MaskedGPUEngine::SearchBatch(const MaskedGPUTask& task, std::vector<MaskedGPUHit>& hits) {
    if (!LaunchOnSlot(0, task)) {
        return false;
    }
    slotActive_[0] = true;
    uint64_t batchStart = 0;
    uint64_t batchCount = 0;
    bool ok = CollectFromSlot(0, hits, batchStart, batchCount, true);
    return ok;
}

bool MaskedGPUEngine::SubmitBatch(const MaskedGPUTask& task) {
    if (!initialised_) {
        return false;
    }
    if (pendingCount_ >= slotCount_) {
        return false;
    }

    int slot = launchCursor_;
    if (slotActive_[slot]) {
        return false;
    }
    if (!LaunchOnSlot(slot, task)) {
        return false;
    }
    slotActive_[slot] = true;
    launchCursor_ = (launchCursor_ + 1) % slotCount_;
    pendingCount_++;
    return true;
}

bool MaskedGPUEngine::CollectBatch(std::vector<MaskedGPUHit>& hits, uint64_t& batchStart, uint64_t& batchCount, bool wait) {
    if (pendingCount_ <= 0) {
        hits.clear();
        batchStart = 0ULL;
        batchCount = 0ULL;
        return false;
    }

    int slot = collectCursor_;
    if (!CollectFromSlot(slot, hits, batchStart, batchCount, wait)) {
        return false;
    }
    collectCursor_ = (collectCursor_ + 1) % slotCount_;
    pendingCount_--;
    return true;
}

bool MaskedGPUEngine::HasPendingBatches() const {
    return pendingCount_ > 0;
}

double MaskedGPUEngine::BenchmarkConfig(const MaskedGPUTask& sampleTask, uint64_t availableCount, int nbThreadGroup, int nbThreadPerGroup, uint32_t batchSteps) {
    if (!fastPathV2_ || availableCount == 0) {
        return 0.0;
    }

    cudaDeviceProp deviceProp;
    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId_));
    ApplyLaunchConfig(nbThreadGroup, nbThreadPerGroup, batchSteps, deviceProp.major, deviceProp.minor);

    const uint64_t perLaunch = std::min<uint64_t>(availableCount, launchCandidateCount_);
    if (perLaunch == 0) {
        return 0.0;
    }

    std::vector<MaskedGPUHit> hits;
    MaskedGPUTask task = sampleTask;
    task.batchStart = 0ULL;
    task.batchCount = perLaunch;

    const int iterations = 4;
    uint64_t totalTried = 0ULL;
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        if (!LaunchOnSlot(0, task)) {
            return 0.0;
        }
        slotActive_[0] = true;
        uint64_t batchStart = 0ULL;
        uint64_t batchCount = 0ULL;
        if (!CollectFromSlot(0, hits, batchStart, batchCount, true)) {
            return 0.0;
        }
        totalTried += batchCount;
        task.batchStart += batchCount;
        if (task.batchStart >= availableCount) {
            task.batchStart = 0ULL;
        }
    }
    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - begin).count();
    if (seconds <= 0.0) {
        return 0.0;
    }
    return (double)totalTried / seconds;
}

bool MaskedGPUEngine::AutoTune(const MaskedGPUTask& sampleTask, uint64_t availableCount, bool userSpecifiedGrid) {
    if (!fastPathV2_ || !initialised_) {
        return true;
    }

    std::vector<uint32_t> batchCandidates;
    if (requestedBatchSteps_ != 0) {
        batchCandidates.push_back(NormalizeBatchSteps((uint32_t)requestedBatchSteps_));
    }
    else {
        batchCandidates.push_back(8U);
        batchCandidates.push_back(16U);
        batchCandidates.push_back(32U);
    }

    std::vector<int> blockCandidates;
    std::vector<int> groupCandidates;
    if (userSpecifiedGrid) {
        blockCandidates.push_back(nbThreadPerGroup_);
        groupCandidates.push_back(nbThreadGroup_);
    }
    else {
        blockCandidates.push_back(128);
        blockCandidates.push_back(256);
        groupCandidates.push_back(std::max(1, smCount_ * 8));
        groupCandidates.push_back(std::max(1, smCount_ * 12));
        groupCandidates.push_back(std::max(1, smCount_ * 16));
    }

    if (!autotuneEnabled_) {
        if (requestedBatchSteps_ == 0) {
            cudaDeviceProp deviceProp;
            CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId_));
            ApplyLaunchConfig(groupCandidates[0], blockCandidates[0], kDefaultFastBatchSteps, deviceProp.major, deviceProp.minor);
        }
        return true;
    }

    cudaDeviceProp deviceProp;
    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId_));

    double bestRate = -1.0;
    int bestGroups = nbThreadGroup_;
    int bestBlock = nbThreadPerGroup_;
    uint32_t bestBatchSteps = batchSteps_;

    for (size_t gi = 0; gi < groupCandidates.size(); gi++) {
        for (size_t bi = 0; bi < blockCandidates.size(); bi++) {
            for (size_t si = 0; si < batchCandidates.size(); si++) {
                double rate = BenchmarkConfig(sampleTask, availableCount, groupCandidates[gi], blockCandidates[bi], batchCandidates[si]);
                if (rate > bestRate) {
                    bestRate = rate;
                    bestGroups = groupCandidates[gi];
                    bestBlock = blockCandidates[bi];
                    bestBatchSteps = batchCandidates[si];
                }
            }
        }
    }

    ApplyLaunchConfig(bestGroups, bestBlock, bestBatchSteps, deviceProp.major, deviceProp.minor);
    return true;
}
