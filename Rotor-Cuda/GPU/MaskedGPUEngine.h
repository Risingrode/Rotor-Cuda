#ifndef MASKEDGPUENGINE_H
#define MASKEDGPUENGINE_H

#include <stdint.h>
#include <string>
#include <vector>
#include "GPUEngine.h"

#if defined(WITHGPU) || defined(__CUDACC__)
#define ROTOR_MASKED_CUDA_TYPES_AVAILABLE 1
#include <cuda_runtime.h>
#endif

#define MASKED_GPU_MAX_SUFFIX 23
#define MASKED_GPU_MAX_CHOICES 16
#define MASKED_GPU_RULE_DIM 17
#define MASKED_GPU_RULE_STATE_COUNT (MASKED_GPU_RULE_DIM * MASKED_GPU_RULE_DIM)
#define MASKED_GPU_CMP_STATE_COUNT 3
#define MASKED_GPU_MAX_SEGMENTS MASKED_GPU_MAX_SUFFIX
#define MASKED_GPU_SEGMENT_COMBO_CAP_V1 64
#define MASKED_GPU_SEGMENT_COMBO_CAP_V2_FASTPATH 243
#define MASKED_GPU_SEGMENT_COMBO_CAP MASKED_GPU_SEGMENT_COMBO_CAP_V2_FASTPATH
#define MASKED_GPU_MAX_SEGMENT_LEN MASKED_GPU_MAX_SUFFIX
#define MASKED_GPU_MODE_V1 1
#define MASKED_GPU_MODE_V2_FASTPATH 2

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
    uint8_t gpuMode;
    uint8_t requestedBatchSteps;
    uint8_t gpuStartPos;
    uint8_t segmentCount;
    uint8_t segmentComboCap;
    uint8_t autotune;
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
    uint8_t segmentPointSet[MASKED_GPU_MAX_SEGMENTS][MASKED_GPU_SEGMENT_COMBO_CAP];
    uint8_t segmentValues[MASKED_GPU_MAX_SEGMENTS][MASKED_GPU_SEGMENT_COMBO_CAP][MASKED_GPU_MAX_SEGMENT_LEN];
    uint64_t segmentPointX[MASKED_GPU_MAX_SEGMENTS][MASKED_GPU_SEGMENT_COMBO_CAP][4];
    uint64_t segmentPointY[MASKED_GPU_MAX_SEGMENTS][MASKED_GPU_SEGMENT_COMBO_CAP][4];
};

struct MaskedGPUFastPathTables {
    std::vector<uint16_t> ruleTransition;
    std::vector<int8_t> boundTransition;
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
                    const MaskedGPUCharsetConfig& config,
                    const MaskedGPUFastPathTables& fastTables);
    ~MaskedGPUEngine();

    bool IsInitialised() const;
    int GetNbThread() const;
    int GetGroupSize() const;
    const std::string& GetDeviceName() const;
    bool IsFastPathV2() const;
    uint32_t GetBatchSteps() const;
    uint64_t GetLaunchCandidateCount() const;
    bool AutoTune(const MaskedGPUTask& sampleTask, uint64_t availableCount, bool userSpecifiedGrid);

    bool SearchBatch(const MaskedGPUTask& task, std::vector<MaskedGPUHit>& hits);
    bool SubmitBatch(const MaskedGPUTask& task);
    bool CollectBatch(std::vector<MaskedGPUHit>& hits, uint64_t& batchStart, uint64_t& batchCount, bool wait);
    bool HasPendingBatches() const;

private:
    enum { kMaxSlots = 2 };

    int gpuId_;
    int nbThreadGroup_;
    int nbThreadPerGroup_;
    int nbThread_;
    int smCount_;
    uint32_t maxFound_;
    uint32_t batchSteps_;
    uint64_t launchCandidateCount_;
    bool initialised_;
    bool fastPathV2_;
    bool autotuneEnabled_;
    uint8_t requestedBatchSteps_;
    std::string deviceName_;
    std::string deviceBaseName_;

    uint32_t* outputCount_[kMaxSlots];
    uint32_t* outputCountPinned_[kMaxSlots];
    MaskedGPUHit* outputHits_[kMaxSlots];
    MaskedGPUHit* outputHitsPinned_[kMaxSlots];
    bool slotActive_[kMaxSlots];
    uint64_t slotBatchStart_[kMaxSlots];
    uint64_t slotBatchCount_[kMaxSlots];
    int slotCount_;
    int launchCursor_;
    int collectCursor_;
    int pendingCount_;
    uint8_t* segmentPointSet_;
    uint8_t* segmentValues_;
    uint64_t* segmentPointX_;
    uint64_t* segmentPointY_;
    uint16_t* fastRuleTransition_;
    int8_t* fastBoundTransition_;
    uint32_t compMode_;
    uint32_t coinType_;

#ifdef ROTOR_MASKED_CUDA_TYPES_AVAILABLE
    cudaStream_t streams_[kMaxSlots];
    cudaEvent_t kernelDone_[kMaxSlots];
#endif

    void UpdateDeviceName(int major, int minor);
    void ApplyLaunchConfig(int nbThreadGroup, int nbThreadPerGroup, uint32_t batchSteps, int major, int minor);
    bool LaunchOnSlot(int slot, const MaskedGPUTask& task);
    bool CollectFromSlot(int slot, std::vector<MaskedGPUHit>& hits, uint64_t& batchStart, uint64_t& batchCount, bool wait);
    double BenchmarkConfig(const MaskedGPUTask& sampleTask, uint64_t availableCount, int nbThreadGroup, int nbThreadPerGroup, uint32_t batchSteps);
};

#endif
