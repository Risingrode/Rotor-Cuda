#ifndef MASKEDGPUENGINE_H
#define MASKEDGPUENGINE_H

#include <stdint.h>
#include <string>
#include <vector>
#include "GPUEngine.h"

#ifdef WITHGPU
#include <cuda_runtime.h>
#endif

#define MASKED_GPU_MAX_SUFFIX 23
#define MASKED_GPU_MAX_CHOICES 16
#define MASKED_GPU_RULE_DIM 17

enum MaskedGPUPosFlags {
    MASKED_GPU_POS_HAS_ZERO      = 1 << 0,
    MASKED_GPU_POS_HAS_NONZERO   = 1 << 1,
    MASKED_GPU_POS_ALL_LT_BOUND  = 1 << 2,
    MASKED_GPU_POS_ALL_EQ_BOUND  = 1 << 3,
    MASKED_GPU_POS_ALL_GT_BOUND  = 1 << 4
};

struct MaskedGPUCharsetConfig {
    uint32_t suffixLen;
    uint32_t compMode;
    uint32_t coinType;
    uint32_t target[5];
    uint8_t forbidTripleSame;
    uint8_t forbidTripleRun;
    uint8_t reserved0[2];
    uint8_t radices[MASKED_GPU_MAX_SUFFIX];
    uint8_t bound[MASKED_GPU_MAX_SUFFIX];
    uint8_t minValue[MASKED_GPU_MAX_SUFFIX];
    uint8_t maxValue[MASKED_GPU_MAX_SUFFIX];
    uint8_t posFlags[MASKED_GPU_MAX_SUFFIX];
    uint8_t nonZeroPossibleFromPos[MASKED_GPU_MAX_SUFFIX + 1];
    uint8_t values[MASKED_GPU_MAX_SUFFIX][MASKED_GPU_MAX_CHOICES];
    uint16_t invalidNextMask[MASKED_GPU_RULE_DIM][MASKED_GPU_RULE_DIM];
    uint64_t pointX[MASKED_GPU_MAX_SUFFIX][MASKED_GPU_MAX_CHOICES][4];
    uint64_t pointY[MASKED_GPU_MAX_SUFFIX][MASKED_GPU_MAX_CHOICES][4];
};

struct MaskedGPUTask {
    uint64_t baseX[4];
    uint64_t baseY[4];
    uint64_t baseZ[4];
    uint64_t batchStart;
    uint64_t batchCount;
    uint8_t pointSet;
    uint8_t hasNonZero;
    int8_t last1;
    int8_t last2;
    int8_t cmpState;
    uint8_t startPos;
    uint8_t reserved1[2];
};

struct MaskedGPUHit {
    uint64_t localIndex;
    uint32_t mode;
    uint32_t reserved;
};

class MaskedGPUEngine {
public:
    MaskedGPUEngine(int gpuId,
                    int nbThreadGroup,
                    int nbThreadPerGroup,
                    uint32_t maxFound,
                    const MaskedGPUCharsetConfig& config);
    ~MaskedGPUEngine();

    bool IsInitialised() const;
    int GetNbThread() const;
    int GetGroupSize() const;
    const std::string& GetDeviceName() const;

    bool SearchBatch(const MaskedGPUTask& task, std::vector<MaskedGPUHit>& hits);

private:
    int gpuId_;
    int nbThreadGroup_;
    int nbThreadPerGroup_;
    int nbThread_;
    uint32_t maxFound_;
    bool initialised_;
    std::string deviceName_;

    uint32_t* outputCount_;
    uint32_t* outputCountPinned_;
    MaskedGPUHit* outputHits_;
    MaskedGPUHit* outputHitsPinned_;
    uint64_t* pointX_;
    uint64_t* pointY_;
    uint32_t compMode_;
    uint32_t coinType_;

#ifdef WITHGPU
    cudaStream_t stream_;
    cudaEvent_t kernelDone_;
#endif
};

#endif
