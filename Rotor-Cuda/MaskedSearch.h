#ifndef MASKEDSEARCH_H
#define MASKEDSEARCH_H

#include <string>
#include <vector>

struct MaskedSearchConfig {
    bool forbidTripleSame;
    bool forbidTripleRun;
    bool strictTailModes;
    bool showProbabilities;
    bool useGpu;
    std::vector<int> gpuIds;
    std::vector<int> gridSize;
    std::string probabilityProfile;

    MaskedSearchConfig()
        : forbidTripleSame(true)
        , forbidTripleRun(true)
        , strictTailModes(false)
        , showProbabilities(true)
        , useGpu(false)
        , probabilityProfile("auto") {
    }
};

int RunMaskedSearch(const std::string& targetLabel,
                    const std::vector<unsigned char>& targetBytes,
                    int compMode,
                    int searchMode,
                    int coinType,
                    const std::string& outputFile,
                    const std::string& prefix,
                    const std::string& suffixSetsSpec,
                    int threadCount,
                    int display,
                    const MaskedSearchConfig& config,
                    bool& should_exit);

#endif
