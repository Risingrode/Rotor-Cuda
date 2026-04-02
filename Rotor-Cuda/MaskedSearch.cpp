#include "MaskedSearch.h"

#include "GPU/GPUEngine.h"
#ifdef WITHGPU
#include "GPU/MaskedGPUEngine.h"
#endif
#include "SECP256k1.h"
#include "Int.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

static const uint64_t kTaskHardCap = 1ULL << 20;
static const uint64_t kAttemptFlushThreshold = 4096ULL;
static const char* kSample41Prefix = "D0AC934BA9987E529BF3150373B63BD06849D740A";

char HexCharUpper(int value) {
    static const char* kHex = "0123456789ABCDEF";
    return kHex[value & 0xF];
}

bool IsHexChar(char c) {
    unsigned char uc = static_cast<unsigned char>(c);
    return std::isxdigit(uc) != 0;
}

int HexValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

std::string ToLower(const std::string& input) {
    std::string out = input;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
        });
    return out;
}

std::string Trim(const std::string& s) {
    std::string::size_type begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) begin++;

    std::string::size_type end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;

    return s.substr(begin, end - begin);
}

std::string NormalizePrefixHex(const std::string& rawPrefix) {
    std::string prefix = Trim(rawPrefix);
    if (prefix.size() >= 2 && prefix[0] == '0' && (prefix[1] == 'x' || prefix[1] == 'X')) {
        prefix.erase(0, 2);
    }

    std::string normalized;
    normalized.reserve(prefix.size());
    for (size_t i = 0; i < prefix.size(); i++) {
        char c = prefix[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }
        if (!IsHexChar(c)) {
            throw std::runtime_error("--prefix contains non-hex characters");
        }
        normalized.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    return normalized;
}

std::vector<std::string> SplitSets(const std::string& spec) {
    std::vector<std::string> out;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        out.push_back(Trim(item));
    }
    return out;
}

std::string ParseCharsetSpec(const std::string& rawSpec) {
    std::string spec = Trim(rawSpec);
    if (spec.empty()) {
        throw std::runtime_error("Each suffix character set must be non-empty");
    }

    bool seen[16];
    std::memset(seen, 0, sizeof(seen));
    std::string result;

    for (size_t i = 0; i < spec.size();) {
        char c = spec[i];

        if (std::isspace(static_cast<unsigned char>(c)) || c == '[' || c == ']') {
            i++;
            continue;
        }

        if (c == '*' || c == '?') {
            for (int v = 0; v < 16; v++) {
                if (!seen[v]) {
                    seen[v] = true;
                    result.push_back(HexCharUpper(v));
                }
            }
            i++;
            continue;
        }

        if (!IsHexChar(c)) {
            throw std::runtime_error("Invalid character in --suffixsets: '" + std::string(1, c) + "'");
        }

        int begin = HexValue(c);
        if (i + 2 < spec.size() && spec[i + 1] == '-' && IsHexChar(spec[i + 2])) {
            int end = HexValue(spec[i + 2]);
            if (begin > end) {
                throw std::runtime_error("Invalid descending range in --suffixsets: '" + spec.substr(i, 3) + "'");
            }
            for (int v = begin; v <= end; v++) {
                if (!seen[v]) {
                    seen[v] = true;
                    result.push_back(HexCharUpper(v));
                }
            }
            i += 3;
            continue;
        }

        if (!seen[begin]) {
            seen[begin] = true;
            result.push_back(HexCharUpper(begin));
        }
        i++;
    }

    if (result.empty()) {
        throw std::runtime_error("A suffix character set expanded to zero hex digits");
    }

    return result;
}

std::string PadHex(const std::string& hex, size_t width) {
    if (hex.size() >= width) {
        return hex;
    }
    return std::string(width - hex.size(), '0') + hex;
}

std::string FormatThousands(uint64_t value) {
    std::string digits;
    do {
        digits.push_back(static_cast<char>('0' + (value % 10)));
        value /= 10;
    } while (value != 0);

    std::string out;
    for (size_t i = 0; i < digits.size(); i++) {
        if (i > 0 && (i % 3) == 0) {
            out.push_back(',');
        }
        out.push_back(digits[i]);
    }
    std::reverse(out.begin(), out.end());
    return out;
}

std::string FormatRate(double perSecond) {
    char buffer[64];
    if (perSecond >= 1000000000.0) {
        std::snprintf(buffer, sizeof(buffer), "%.2f Gk/s", perSecond / 1000000000.0);
    }
    else if (perSecond >= 1000000.0) {
        std::snprintf(buffer, sizeof(buffer), "%.2f Mk/s", perSecond / 1000000.0);
    }
    else if (perSecond >= 1000.0) {
        std::snprintf(buffer, sizeof(buffer), "%.2f Kk/s", perSecond / 1000.0);
    }
    else {
        std::snprintf(buffer, sizeof(buffer), "%.2f k/s", perSecond);
    }
    return std::string(buffer);
}

std::string FormatKeyspace(long double log2Value) {
    if (log2Value <= 0.0L) {
        return "1 candidate";
    }

    long double log10Value = log2Value * std::log10(2.0L);
    long double exponent = std::floor(log10Value);
    long double mantissa = std::pow(10.0L, log10Value - exponent);

    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "~2^%.2Lf (~%.2Lf x 10^%.0Lf)", log2Value, mantissa, exponent);
    return std::string(buffer);
}

std::string FormatPercent(double probability) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.3f%%", probability * 100.0);
    return std::string(buffer);
}

std::string FormatDouble(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.4f", value);
    return std::string(buffer);
}

int Popcount16(uint16_t value) {
    int count = 0;
    while (value != 0) {
        count += (value & 1U);
        value >>= 1U;
    }
    return count;
}

const char* CoinName(int coinType) {
    return coinType == COIN_BTC ? "BITCOIN" : "ETHEREUM";
}

const char* CompModeName(int compMode) {
    switch (compMode) {
    case SEARCH_COMPRESSED:
        return "COMPRESSED";
    case SEARCH_UNCOMPRESSED:
        return "UNCOMPRESSED";
    case SEARCH_BOTH:
        return "COMPRESSED & UNCOMPRESSED";
    default:
        return "UNKNOWN";
    }
}

struct PrefixInfo {
    std::string normalizedHex;
    std::array<uint8_t, 16> counts;
    std::array<uint16_t, 6> bucketMask;
    std::array<int, 6> bucketSizes;
    int digitCount;
    int letterCount;
    int adjacentRepeatPairs;
    std::string patternKey;
    std::string missingChars;

    PrefixInfo()
        : digitCount(0)
        , letterCount(0)
        , adjacentRepeatPairs(0) {
        counts.fill(0);
        bucketMask.fill(0);
        bucketSizes.fill(0);
    }
};

PrefixInfo AnalyzePrefix(const std::string& normalizedPrefix) {
    PrefixInfo info;
    info.normalizedHex = normalizedPrefix;

    for (size_t i = 0; i < normalizedPrefix.size(); i++) {
        int value = HexValue(normalizedPrefix[i]);
        if (value < 0) {
            throw std::runtime_error("Internal error while analyzing prefix hex");
        }
        info.counts[static_cast<size_t>(value)]++;
        if (value < 10) info.digitCount++;
        else info.letterCount++;
        if (i > 0 && normalizedPrefix[i] == normalizedPrefix[i - 1]) {
            info.adjacentRepeatPairs++;
        }
    }

    std::array<int, 42> histogram;
    histogram.fill(0);
    for (int value = 0; value < 16; value++) {
        int freq = info.counts[static_cast<size_t>(value)];
        if (freq >= 0 && freq < static_cast<int>(histogram.size())) {
            histogram[static_cast<size_t>(freq)]++;
        }
        if (freq > 0 && freq < static_cast<int>(info.bucketMask.size())) {
            info.bucketMask[static_cast<size_t>(freq)] |= static_cast<uint16_t>(1U << value);
            info.bucketSizes[static_cast<size_t>(freq)]++;
        }
        if (freq == 0) {
            info.missingChars.push_back(HexCharUpper(value));
        }
    }

    std::ostringstream oss;
    bool first = true;
    for (size_t freq = 1; freq < histogram.size(); freq++) {
        if (histogram[freq] == 0) continue;
        if (!first) oss << ' ';
        first = false;
        oss << freq << ':' << histogram[freq];
    }
    info.patternKey = oss.str();

    return info;
}

struct ProbabilityProfile {
    std::string name;
    std::string exactPrefixHex;
    int prefixDigitCount;
    int prefixLetterCount;
    std::string prefixPatternKey;
    std::vector<double> tailDigitProb;
    std::vector<double> uniqueCountProb;
    std::array< std::vector<double>, 6 > prefixBucketHitProb;
    std::unordered_map<std::string, double> tailModeProb;
    double unknownTailModeProb;

    ProbabilityProfile()
        : prefixDigitCount(0)
        , prefixLetterCount(0)
        , unknownTailModeProb(1e-3) {
    }
};

void SetTailDigitProbability(ProbabilityProfile& profile, int digits, double percent) {
    if (digits >= 0 && digits < static_cast<int>(profile.tailDigitProb.size())) {
        profile.tailDigitProb[static_cast<size_t>(digits)] = percent / 100.0;
    }
}

void SetUniqueProbability(ProbabilityProfile& profile, int uniqueCount, uint32_t count, uint32_t total) {
    if (uniqueCount >= 0 && uniqueCount < static_cast<int>(profile.uniqueCountProb.size()) && total != 0) {
        profile.uniqueCountProb[static_cast<size_t>(uniqueCount)] = static_cast<double>(count) / static_cast<double>(total);
    }
}

void SetBucketHitProbability(ProbabilityProfile& profile, int bucketFreq, int hits, uint32_t count, uint32_t total) {
    if (bucketFreq <= 0 || bucketFreq >= static_cast<int>(profile.prefixBucketHitProb.size()) || total == 0) {
        return;
    }
    if (hits >= static_cast<int>(profile.prefixBucketHitProb[static_cast<size_t>(bucketFreq)].size())) {
        profile.prefixBucketHitProb[static_cast<size_t>(bucketFreq)].resize(static_cast<size_t>(hits + 1), 0.0);
    }
    profile.prefixBucketHitProb[static_cast<size_t>(bucketFreq)][static_cast<size_t>(hits)] = static_cast<double>(count) / static_cast<double>(total);
}

void AddTailMode(ProbabilityProfile& profile, const std::string& modeKey, double percent) {
    profile.tailModeProb[modeKey] = percent / 100.0;
}

std::string BuildProbabilityEntryList(const std::vector<double>& probs, const std::string& separator) {
    std::ostringstream oss;
    bool first = true;
    for (size_t i = 0; i < probs.size(); i++) {
        if (probs[i] <= 0.0) continue;
        if (!first) oss << separator;
        first = false;
        oss << i << ':' << FormatPercent(probs[i]);
    }
    return oss.str();
}

std::string BuildTailModeSummary(const std::unordered_map<std::string, double>& modeProb) {
    std::vector< std::pair<std::string, double> > items;
    items.reserve(modeProb.size());
    for (std::unordered_map<std::string, double>::const_iterator it = modeProb.begin(); it != modeProb.end(); ++it) {
        items.push_back(*it);
    }
    std::sort(items.begin(), items.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    std::ostringstream oss;
    for (size_t i = 0; i < items.size(); i++) {
        if (i > 0) oss << " | ";
        oss << items[i].first << '=' << FormatPercent(items[i].second);
    }
    return oss.str();
}

ProbabilityProfile BuildSample41Profile() {
    ProbabilityProfile profile;
    profile.name = "sample41";
    profile.exactPrefixHex = kSample41Prefix;
    profile.prefixDigitCount = 28;
    profile.prefixLetterCount = 13;
    profile.prefixPatternKey = "1:5 2:3 3:4 4:2 5:2";
    profile.tailDigitProb.assign(24, 0.0);
    profile.uniqueCountProb.assign(17, 0.0);

    SetTailDigitProbability(profile, 14, 16.482);
    SetTailDigitProbability(profile, 15, 16.273);
    SetTailDigitProbability(profile, 16, 14.081);
    SetTailDigitProbability(profile, 13, 14.018);
    SetTailDigitProbability(profile, 12, 10.213);
    SetTailDigitProbability(profile, 17, 9.699);
    SetTailDigitProbability(profile, 11, 5.665);
    SetTailDigitProbability(profile, 18, 5.316);
    SetTailDigitProbability(profile, 10, 2.830);
    SetTailDigitProbability(profile, 19, 2.328);
    SetTailDigitProbability(profile, 9, 1.363);
    SetTailDigitProbability(profile, 20, 0.843);
    SetTailDigitProbability(profile, 8, 0.449);
    SetTailDigitProbability(profile, 21, 0.234);
    SetTailDigitProbability(profile, 7, 0.124);

    const uint32_t totalUniqueSamples = 500000;
    SetUniqueProbability(profile, 6, 1, totalUniqueSamples);
    SetUniqueProbability(profile, 7, 27, totalUniqueSamples);
    SetUniqueProbability(profile, 8, 550, totalUniqueSamples);
    SetUniqueProbability(profile, 9, 5237, totalUniqueSamples);
    SetUniqueProbability(profile, 10, 28284, totalUniqueSamples);
    SetUniqueProbability(profile, 11, 85865, totalUniqueSamples);
    SetUniqueProbability(profile, 12, 147372, totalUniqueSamples);
    SetUniqueProbability(profile, 13, 140864, totalUniqueSamples);
    SetUniqueProbability(profile, 14, 72270, totalUniqueSamples);
    SetUniqueProbability(profile, 15, 17921, totalUniqueSamples);
    SetUniqueProbability(profile, 16, 1609, totalUniqueSamples);

    const uint32_t totalPrefixSamples = 500000;
    SetBucketHitProbability(profile, 1, 0, 84, totalPrefixSamples);
    SetBucketHitProbability(profile, 1, 1, 2870, totalPrefixSamples);
    SetBucketHitProbability(profile, 1, 2, 29967, totalPrefixSamples);
    SetBucketHitProbability(profile, 1, 3, 124815, totalPrefixSamples);
    SetBucketHitProbability(profile, 1, 4, 216039, totalPrefixSamples);
    SetBucketHitProbability(profile, 1, 5, 126225, totalPrefixSamples);

    SetBucketHitProbability(profile, 2, 0, 4152, totalPrefixSamples);
    SetBucketHitProbability(profile, 2, 1, 56935, totalPrefixSamples);
    SetBucketHitProbability(profile, 2, 2, 213572, totalPrefixSamples);
    SetBucketHitProbability(profile, 2, 3, 225341, totalPrefixSamples);

    SetBucketHitProbability(profile, 3, 0, 669, totalPrefixSamples);
    SetBucketHitProbability(profile, 3, 1, 14322, totalPrefixSamples);
    SetBucketHitProbability(profile, 3, 2, 93051, totalPrefixSamples);
    SetBucketHitProbability(profile, 3, 3, 222424, totalPrefixSamples);
    SetBucketHitProbability(profile, 3, 4, 169534, totalPrefixSamples);

    SetBucketHitProbability(profile, 4, 0, 23365, totalPrefixSamples);
    SetBucketHitProbability(profile, 4, 1, 180315, totalPrefixSamples);
    SetBucketHitProbability(profile, 4, 2, 296320, totalPrefixSamples);

    SetBucketHitProbability(profile, 5, 0, 23055, totalPrefixSamples);
    SetBucketHitProbability(profile, 5, 1, 180391, totalPrefixSamples);
    SetBucketHitProbability(profile, 5, 2, 296554, totalPrefixSamples);

    AddTailMode(profile, "0:4 1:5 2:4 3:2 4:1", 5.720);
    AddTailMode(profile, "0:3 1:6 2:4 3:3", 5.043);
    AddTailMode(profile, "0:3 1:5 2:6 3:2", 4.615);
    AddTailMode(profile, "0:3 1:6 2:5 3:1 4:1", 4.542);
    AddTailMode(profile, "0:3 1:7 2:3 3:2 4:1", 4.338);
    AddTailMode(profile, "0:2 1:7 2:5 3:2", 3.883);
    AddTailMode(profile, "0:4 1:4 2:5 3:3", 3.845);
    AddTailMode(profile, "0:4 1:4 2:6 3:1 4:1", 2.838);
    AddTailMode(profile, "0:4 1:6 2:2 3:3 4:1", 2.600);
    AddTailMode(profile, "0:4 1:5 2:3 3:4", 2.538);
    AddTailMode(profile, "0:5 1:4 2:3 3:3 4:1", 2.517);
    AddTailMode(profile, "0:2 1:8 2:4 3:1 4:1", 2.456);
    AddTailMode(profile, "0:5 1:3 2:5 3:2 4:1", 2.338);
    AddTailMode(profile, "0:2 1:8 2:3 3:3", 2.161);
    AddTailMode(profile, "0:2 1:6 2:7 3:1", 1.930);
    AddTailMode(profile, "0:4 1:6 2:3 3:1 4:2", 1.887);
    AddTailMode(profile, "0:4 1:3 2:7 3:2", 1.650);
    AddTailMode(profile, "0:4 1:6 2:3 3:2 5:1", 1.519);
    AddTailMode(profile, "0:3 1:7 2:2 3:4", 1.452);
    AddTailMode(profile, "0:5 1:4 2:4 3:1 4:2", 1.424);

    profile.unknownTailModeProb = 0.0010;
    return profile;
}

struct PositionChoice {
    char hex;
    uint8_t value;
    bool isDigit;
    Point point;
};

struct TailState {
    int digits;
    uint16_t usedMask;
    uint16_t prefixSeenMask;
    std::array<uint8_t, 16> counts;
    int uniqueCount;
    int last1;
    int last2;

    TailState()
        : digits(0)
        , usedMask(0)
        , prefixSeenMask(0)
        , uniqueCount(0)
        , last1(-1)
        , last2(-1) {
        counts.fill(0);
    }
};

std::string BuildTailModeKey(const std::array<uint8_t, 16>& tailCounts) {
    std::array<int, 24> histogram;
    histogram.fill(0);
    for (size_t i = 0; i < tailCounts.size(); i++) {
        histogram[tailCounts[i]]++;
    }

    std::ostringstream oss;
    bool first = true;
    for (size_t freq = 0; freq < histogram.size(); freq++) {
        if (histogram[freq] == 0) continue;
        if (!first) oss << ' ';
        first = false;
        oss << freq << ':' << histogram[freq];
    }
    return oss.str();
}

struct HeuristicEvaluation {
    bool enabled;
    bool valid;
    int tailDigits;
    int tailLetters;
    int uniqueChars;
    std::string tailModeKey;
    double ratioProb;
    double uniqueProb;
    double tailModeProb;
    std::array<int, 6> bucketHits;
    std::array<double, 6> bucketHitProb;
    double logScore;
    std::string profileName;

    HeuristicEvaluation()
        : enabled(false)
        , valid(true)
        , tailDigits(0)
        , tailLetters(0)
        , uniqueChars(0)
        , ratioProb(0.0)
        , uniqueProb(0.0)
        , tailModeProb(0.0)
        , logScore(0.0) {
        bucketHits.fill(0);
        bucketHitProb.fill(0.0);
    }
};

class MaskedSearchRunner {
public:
    MaskedSearchRunner(const std::string& targetLabel,
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
        bool& should_exit)
        : targetLabel_(targetLabel)
        , targetBytes_(targetBytes)
        , compMode_(compMode)
        , searchMode_(searchMode)
        , coinType_(coinType)
        , outputFile_(outputFile)
        , prefixRaw_(prefix)
        , suffixSetsSpec_(suffixSetsSpec)
        , threadCount_(threadCount > 0 ? threadCount : 1)
        , display_(display)
        , config_(config)
        , should_exit_(should_exit)
        , prefixLen_(0)
        , suffixLen_(0)
        , prefixPointSet_(false)
        , prefixHasNonZero_(false)
        , prefixCmpState_(0)
        , taskDepth_(0)
        , taskCount_(1)
        , keyspaceLog2_(0.0L)
        , profileActive_(false)
        , minSupportedDigits_(0)
        , maxSupportedDigits_(23)
        , minSupportedUnique_(0)
        , maxSupportedUnique_(16)
        , gpuProfileIgnored_(false)
        , gpuStrictTailModeIgnored_(false)
        , gpuResidualCount_(1)
        , taskCountClipped_(false)
        , found_(false)
        , stop_(false)
        , attempts_(0)
        , nextTask_(0)
        , foundCompressed_(false) {
    }

    int Run() {
        Prepare();

#ifdef WITHGPU
        MaskedGPUEngine* gpuEngine = NULL;
        if (config_.useGpu) {
            gpuEngine = CreateGpuEngine();
        }
#else
        if (config_.useGpu) {
            throw std::runtime_error("Masked GPU search requested but this build was compiled without WITHGPU");
        }
#endif

        PrintStartInfo();

        std::thread reporter;
        if (display_ > 0) {
            reporter = std::thread(&MaskedSearchRunner::ProgressLoop, this);
        }

        if (config_.useGpu) {
#ifdef WITHGPU
            RunGPU(*gpuEngine);
            delete gpuEngine;
            gpuEngine = NULL;
#endif
        }
        else {
            RunCPU();
        }

        stop_.store(true);
        if (reporter.joinable()) {
            reporter.join();
        }

        if (display_ > 0) {
            std::printf("\n");
        }

        if (found_.load()) {
            return 0;
        }

        if (should_exit_) {
            std::printf("  Masked search interrupted by user\n");
            return 1;
        }

        std::printf("  Masked search finished: no key matched the target\n");
        return 1;
    }

private:
    void Prepare() {
        if (searchMode_ != SEARCH_MODE_SA) {
            throw std::runtime_error("Masked suffix search currently supports only '-m address'");
        }
        if (coinType_ != COIN_BTC && coinType_ != COIN_ETH) {
            throw std::runtime_error("Unsupported coin type for masked suffix search");
        }
        if (targetBytes_.size() != 20) {
            throw std::runtime_error("Masked suffix search expects a single 20-byte address hash target");
        }

        prefix_ = NormalizePrefixHex(prefixRaw_);
        if (prefix_.size() > 64) {
            throw std::runtime_error("--prefix must not exceed 64 hex characters");
        }

        std::vector<std::string> rawSets = SplitSets(suffixSetsSpec_);
        if (rawSets.empty()) {
            throw std::runtime_error("--suffixsets must contain at least one character set");
        }

        sets_.clear();
        sets_.reserve(rawSets.size());
        for (size_t i = 0; i < rawSets.size(); i++) {
            sets_.push_back(ParseCharsetSpec(rawSets[i]));
        }

        prefixLen_ = prefix_.size();
        suffixLen_ = sets_.size();

        if (prefixLen_ + suffixLen_ != 64) {
            std::ostringstream oss;
            oss << "--prefix length (" << prefixLen_ << ") + suffix set count (" << suffixLen_
                << ") must equal 64 hex nibbles";
            throw std::runtime_error(oss.str());
        }
        if (config_.useGpu && suffixLen_ > 23) {
            throw std::runtime_error("Masked GPU V1 currently supports at most 23 suffix hex chars");
        }

        prefixInfo_ = AnalyzePrefix(prefix_);
        ResolveProbabilityProfile();

        secp_.Init();

        Int maxValid(secp_.order);
        maxValid.SubOne();
        maxValidHex_ = PadHex(maxValid.GetBase16(), 64);

        prefixCmpState_ = ComparePrefix(prefix_, maxValidHex_);
        if (prefixCmpState_ > 0) {
            throw std::runtime_error("Known prefix is already greater than secp256k1 order - 1; no valid private key exists in this suffix space");
        }

        prefixHasNonZero_ = prefix_.find_first_not_of('0') != std::string::npos;
        if (prefixHasNonZero_) {
            std::string baseKeyHex = prefix_ + std::string(suffixLen_, '0');
            Int baseKey;
            baseKey.SetBase16(baseKeyHex.c_str());
            prefixPoint_ = secp_.ComputePublicKey(&baseKey);
            prefixPointSet_ = true;
        }

        BuildChoices();
        if (config_.useGpu) {
            BuildGpuTaskPlan();
        }
        else {
            BuildTaskPlan();
        }
    }

    void ResolveProbabilityProfile() {
        std::string requested = ToLower(Trim(config_.probabilityProfile));
        ProbabilityProfile sample41 = BuildSample41Profile();

        profileActive_ = false;
        profileReason_.clear();
        minSupportedDigits_ = 0;
        maxSupportedDigits_ = static_cast<int>(suffixLen_);
        minSupportedUnique_ = 0;
        maxSupportedUnique_ = 16;
        gpuProfileIgnored_ = false;
        gpuStrictTailModeIgnored_ = false;

        if (config_.useGpu) {
            if (requested.empty()) {
                requested = "auto";
            }
            if (requested != "auto" && requested != "none" && requested != "sample41") {
                throw std::runtime_error("Unknown probability profile: " + config_.probabilityProfile);
            }
            gpuProfileIgnored_ = (requested != "none");
            gpuStrictTailModeIgnored_ = config_.strictTailModes;
            return;
        }

        bool shapeMatch = (prefixLen_ == 41 && suffixLen_ == 23);
        bool exactMatch = (prefixInfo_.normalizedHex == sample41.exactPrefixHex);
        bool featureMatch = (prefixInfo_.digitCount == sample41.prefixDigitCount &&
            prefixInfo_.letterCount == sample41.prefixLetterCount &&
            prefixInfo_.patternKey == sample41.prefixPatternKey);

        if (requested.empty()) {
            requested = "auto";
        }

        if (requested == "none") {
            return;
        }

        if (!shapeMatch) {
            if (requested == "sample41") {
                throw std::runtime_error("Probability profile 'sample41' requires a 41-char prefix and 23-char suffix");
            }
            return;
        }

        if (requested == "auto") {
            if (exactMatch) {
                profile_ = sample41;
                profileActive_ = true;
                profileReason_ = "auto(exact-prefix)";
            }
            else if (featureMatch) {
                profile_ = sample41;
                profileActive_ = true;
                profileReason_ = "auto(prefix-features)";
            }
        }
        else if (requested == "sample41") {
            profile_ = sample41;
            profileActive_ = true;
            profileReason_ = exactMatch ? "forced(exact-prefix)" : (featureMatch ? "forced(prefix-features)" : "forced(mismatch-warning)");
        }
        else {
            throw std::runtime_error("Unknown probability profile: " + config_.probabilityProfile);
        }

        if (!profileActive_) {
            return;
        }

        minSupportedDigits_ = static_cast<int>(suffixLen_);
        maxSupportedDigits_ = 0;
        for (size_t i = 0; i < profile_.tailDigitProb.size(); i++) {
            if (profile_.tailDigitProb[i] > 0.0) {
                minSupportedDigits_ = std::min(minSupportedDigits_, static_cast<int>(i));
                maxSupportedDigits_ = std::max(maxSupportedDigits_, static_cast<int>(i));
            }
        }
        minSupportedUnique_ = 16;
        maxSupportedUnique_ = 0;
        for (size_t i = 0; i < profile_.uniqueCountProb.size(); i++) {
            if (profile_.uniqueCountProb[i] > 0.0) {
                minSupportedUnique_ = std::min(minSupportedUnique_, static_cast<int>(i));
                maxSupportedUnique_ = std::max(maxSupportedUnique_, static_cast<int>(i));
            }
        }
    }

    void BuildChoices() {
        remMinDigits_.assign(suffixLen_ + 1, 0);
        remMaxDigits_.assign(suffixLen_ + 1, 0);
        unionMaskFromPos_.assign(suffixLen_ + 1, 0);

        choices_.clear();
        choices_.resize(suffixLen_);
        keyspaceLog2_ = 0.0L;

        for (size_t pos = 0; pos < suffixLen_; pos++) {
            keyspaceLog2_ += std::log(static_cast<long double>(sets_[pos].size())) / std::log(2.0L);
            choices_[pos].reserve(sets_[pos].size());
            for (size_t j = 0; j < sets_[pos].size(); j++) {
                PositionChoice choice;
                choice.hex = sets_[pos][j];
                choice.value = static_cast<uint8_t>(HexValue(choice.hex));
                choice.isDigit = choice.value < 10;
                if (choice.value != 0) {
                    std::string scalarHex(64, '0');
                    scalarHex[prefixLen_ + pos] = choice.hex;
                    Int scalar;
                    scalar.SetBase16(scalarHex.c_str());
                    choice.point = secp_.ComputePublicKey(&scalar);
                }
                else {
                    choice.point.Clear();
                }
                choices_[pos].push_back(choice);
            }
        }

        for (int pos = static_cast<int>(suffixLen_) - 1; pos >= 0; pos--) {
            bool hasDigit = false;
            bool hasLetter = false;
            uint16_t unionMask = unionMaskFromPos_[static_cast<size_t>(pos + 1)];
            for (size_t j = 0; j < choices_[static_cast<size_t>(pos)].size(); j++) {
                if (choices_[static_cast<size_t>(pos)][j].isDigit) hasDigit = true;
                else hasLetter = true;
                unionMask |= static_cast<uint16_t>(1U << choices_[static_cast<size_t>(pos)][j].value);
            }
            remMinDigits_[static_cast<size_t>(pos)] = remMinDigits_[static_cast<size_t>(pos + 1)] + (hasDigit ? 0 : 1);
            remMaxDigits_[static_cast<size_t>(pos)] = remMaxDigits_[static_cast<size_t>(pos + 1)] + (hasDigit ? 1 : 0);
            unionMaskFromPos_[static_cast<size_t>(pos)] = unionMask;
        }
    }

    void BuildTaskPlan() {
        uint64_t targetTasks = std::max<uint64_t>(1024ULL, static_cast<uint64_t>(threadCount_) * 256ULL);
        taskDepth_ = 0;
        taskCount_ = 1;
        taskCountClipped_ = false;

        while (taskDepth_ < suffixLen_) {
            uint64_t radix = static_cast<uint64_t>(choices_[taskDepth_].size());
            if (taskDepth_ > 0 && taskCount_ >= targetTasks) {
                break;
            }
            if (taskCount_ > (kTaskHardCap / radix)) {
                break;
            }
            taskCount_ *= radix;
            taskDepth_++;
        }

        if (taskDepth_ == 0 && suffixLen_ > 0) {
            taskDepth_ = 1;
            taskCount_ = static_cast<uint64_t>(choices_[0].size());
        }
    }

    void BuildGpuTaskPlan() {
        gpuResidualCount_ = 1;
        taskDepth_ = suffixLen_;
        taskCount_ = 1;
        taskCountClipped_ = false;

        size_t startPos = suffixLen_;
        while (startPos > 0) {
            uint64_t radix = static_cast<uint64_t>(choices_[startPos - 1].size());
            if (gpuResidualCount_ > (std::numeric_limits<uint64_t>::max() / radix)) {
                break;
            }
            gpuResidualCount_ *= radix;
            startPos--;
        }
        taskDepth_ = startPos;

        for (size_t pos = 0; pos < taskDepth_; pos++) {
            uint64_t radix = static_cast<uint64_t>(choices_[pos].size());
            if (taskCount_ > (std::numeric_limits<uint64_t>::max() / radix)) {
                taskCount_ = std::numeric_limits<uint64_t>::max();
                taskCountClipped_ = true;
                break;
            }
            taskCount_ *= radix;
        }
    }

    static int ComparePrefix(const std::string& prefix, const std::string& boundHex) {
        for (size_t i = 0; i < prefix.size(); i++) {
            if (prefix[i] < boundHex[i]) return -1;
            if (prefix[i] > boundHex[i]) return 1;
        }
        return 0;
    }

    int AdvanceCmpState(int cmpState, size_t fullPos, char hex) const {
        if (cmpState < 0) {
            return -1;
        }
        char bound = maxValidHex_[fullPos];
        if (hex < bound) return -1;
        if (hex == bound) return 0;
        return 1;
    }

    bool ShouldStop() const {
        return stop_.load() || found_.load() || should_exit_;
    }

    void FlushAttempts(uint64_t& localAttempts) {
        if (localAttempts > 0) {
            attempts_.fetch_add(localAttempts);
            localAttempts = 0;
        }
    }

    bool ApplyChoice(TailState& state, const PositionChoice& choice) const {
        int value = static_cast<int>(choice.value);

        if (config_.forbidTripleSame && state.last2 == value && state.last1 == value) {
            return false;
        }

        if (config_.forbidTripleRun && state.last2 >= 0 && state.last1 >= 0) {
            if (state.last2 + 1 == state.last1 && state.last1 + 1 == value) {
                return false;
            }
            if (state.last2 - 1 == state.last1 && state.last1 - 1 == value) {
                return false;
            }
        }

        state.last2 = state.last1;
        state.last1 = value;

        if (choice.isDigit) {
            state.digits++;
        }

        uint16_t bit = static_cast<uint16_t>(1U << value);
        if ((state.usedMask & bit) == 0) {
            state.usedMask |= bit;
            state.uniqueCount++;
        }
        state.counts[static_cast<size_t>(value)]++;

        if (prefixInfo_.counts[static_cast<size_t>(value)] > 0) {
            state.prefixSeenMask |= bit;
        }

        return true;
    }

    bool PassBounds(const TailState& state, size_t nextPos) const {
        if (!profileActive_) {
            return true;
        }

        int minDigits = state.digits + remMinDigits_[nextPos];
        int maxDigits = state.digits + remMaxDigits_[nextPos];
        if (maxDigits < minSupportedDigits_ || minDigits > maxSupportedDigits_) {
            return false;
        }

        bool ratioPossible = false;
        for (int digits = std::max(minDigits, minSupportedDigits_); digits <= std::min(maxDigits, maxSupportedDigits_); digits++) {
            if (profile_.tailDigitProb[static_cast<size_t>(digits)] > 0.0) {
                ratioPossible = true;
                break;
            }
        }
        if (!ratioPossible) {
            return false;
        }

        if (state.uniqueCount > maxSupportedUnique_) {
            return false;
        }
        int possibleNew = Popcount16(static_cast<uint16_t>(unionMaskFromPos_[nextPos] & static_cast<uint16_t>(~state.usedMask)));
        if (state.uniqueCount + possibleNew < minSupportedUnique_) {
            return false;
        }

        return true;
    }

    HeuristicEvaluation EvaluateState(const TailState& state) const {
        HeuristicEvaluation eval;
        eval.enabled = profileActive_;
        eval.valid = true;
        eval.tailDigits = state.digits;
        eval.tailLetters = static_cast<int>(suffixLen_) - state.digits;
        eval.uniqueChars = state.uniqueCount;
        eval.profileName = profile_.name;

        if (!profileActive_) {
            return eval;
        }

        eval.ratioProb = (eval.tailDigits >= 0 && eval.tailDigits < static_cast<int>(profile_.tailDigitProb.size()))
            ? profile_.tailDigitProb[static_cast<size_t>(eval.tailDigits)]
            : 0.0;
        if (eval.ratioProb <= 0.0) {
            eval.valid = false;
            return eval;
        }

        eval.uniqueProb = (eval.uniqueChars >= 0 && eval.uniqueChars < static_cast<int>(profile_.uniqueCountProb.size()))
            ? profile_.uniqueCountProb[static_cast<size_t>(eval.uniqueChars)]
            : 0.0;
        if (eval.uniqueProb <= 0.0) {
            eval.valid = false;
            return eval;
        }

        double logScore = std::log(eval.ratioProb) + std::log(eval.uniqueProb);

        for (int bucketFreq = 1; bucketFreq <= 5; bucketFreq++) {
            if (prefixInfo_.bucketSizes[static_cast<size_t>(bucketFreq)] <= 0) {
                continue;
            }
            eval.bucketHits[static_cast<size_t>(bucketFreq)] = Popcount16(static_cast<uint16_t>(state.prefixSeenMask & prefixInfo_.bucketMask[static_cast<size_t>(bucketFreq)]));
            const std::vector<double>& probs = profile_.prefixBucketHitProb[static_cast<size_t>(bucketFreq)];
            if (eval.bucketHits[static_cast<size_t>(bucketFreq)] >= static_cast<int>(probs.size())) {
                eval.valid = false;
                return eval;
            }
            eval.bucketHitProb[static_cast<size_t>(bucketFreq)] = probs[static_cast<size_t>(eval.bucketHits[static_cast<size_t>(bucketFreq)])];
            if (eval.bucketHitProb[static_cast<size_t>(bucketFreq)] <= 0.0) {
                eval.valid = false;
                return eval;
            }
            logScore += std::log(eval.bucketHitProb[static_cast<size_t>(bucketFreq)]);
        }

        eval.tailModeKey = BuildTailModeKey(state.counts);
        std::unordered_map<std::string, double>::const_iterator modeIt = profile_.tailModeProb.find(eval.tailModeKey);
        if (modeIt != profile_.tailModeProb.end()) {
            eval.tailModeProb = modeIt->second;
        }
        else if (config_.strictTailModes) {
            eval.valid = false;
            return eval;
        }
        else {
            eval.tailModeProb = profile_.unknownTailModeProb;
        }

        logScore += std::log(eval.tailModeProb);
        eval.logScore = logScore;
        return eval;
    }

    bool Explore(size_t pos,
        std::string& keyHex,
        Point current,
        bool pointSet,
        bool hasNonZero,
        int cmpState,
        const TailState& state,
        uint64_t& localAttempts) {

        if (ShouldStop()) {
            return true;
        }

        if (pos == suffixLen_) {
            if (!pointSet || !hasNonZero) {
                return false;
            }

            localAttempts++;
            if (localAttempts >= kAttemptFlushThreshold) {
                FlushAttempts(localAttempts);
                if (ShouldStop()) {
                    return true;
                }
            }

            HeuristicEvaluation eval = EvaluateState(state);
            if (!eval.valid) {
                return false;
            }

            return CheckCandidate(keyHex, current, eval);
        }

        const size_t fullPos = prefixLen_ + pos;
        for (size_t i = 0; i < choices_[pos].size(); i++) {
            if (ShouldStop()) {
                return true;
            }

            const PositionChoice& choice = choices_[pos][i];
            int nextCmp = AdvanceCmpState(cmpState, fullPos, choice.hex);
            if (nextCmp > 0) {
                continue;
            }

            TailState nextState = state;
            if (!ApplyChoice(nextState, choice)) {
                continue;
            }
            if (!PassBounds(nextState, pos + 1)) {
                continue;
            }

            keyHex[fullPos] = choice.hex;
            bool nextHasNonZero = hasNonZero || (choice.value != 0);

            if (choice.value == 0) {
                if (Explore(pos + 1, keyHex, current, pointSet, nextHasNonZero, nextCmp, nextState, localAttempts)) {
                    return true;
                }
            }
            else {
                Point nextPoint;
                bool nextPointSet = true;
                if (pointSet) {
                    Point addPoint = choice.point;
                    nextPoint = secp_.Add2(current, addPoint);
                }
                else {
                    nextPoint = choice.point;
                }

                if (Explore(pos + 1, keyHex, nextPoint, nextPointSet, nextHasNonZero, nextCmp, nextState, localAttempts)) {
                    return true;
                }
            }
        }

        return false;
    }

    bool ApplyTaskPrefix(uint64_t taskId,
        std::string& keyHex,
        Point& current,
        bool& pointSet,
        bool& hasNonZero,
        int& cmpState,
        TailState& state) {

        if (taskDepth_ == 0) {
            return PassBounds(state, 0);
        }

        std::vector<size_t> indexes(taskDepth_, 0);
        for (size_t pos = taskDepth_; pos-- > 0;) {
            size_t radix = choices_[pos].size();
            indexes[pos] = static_cast<size_t>(taskId % radix);
            taskId /= radix;
        }

        for (size_t pos = 0; pos < taskDepth_; pos++) {
            const PositionChoice& choice = choices_[pos][indexes[pos]];
            size_t fullPos = prefixLen_ + pos;
            int nextCmp = AdvanceCmpState(cmpState, fullPos, choice.hex);
            if (nextCmp > 0) {
                return false;
            }
            if (!ApplyChoice(state, choice)) {
                return false;
            }

            keyHex[fullPos] = choice.hex;
            cmpState = nextCmp;
            hasNonZero = hasNonZero || (choice.value != 0);

            if (choice.value != 0) {
                if (pointSet) {
                    Point addPoint = choice.point;
                    current = secp_.Add2(current, addPoint);
                }
                else {
                    current = choice.point;
                    pointSet = true;
                }
            }
        }

        return PassBounds(state, taskDepth_);
    }

    void RunCPU() {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(threadCount_));
        for (int i = 0; i < threadCount_; i++) {
            workers.push_back(std::thread(&MaskedSearchRunner::WorkerLoop, this));
        }
        for (size_t i = 0; i < workers.size(); i++) {
            workers[i].join();
        }
    }

#ifdef WITHGPU
    MaskedGPUCharsetConfig BuildGpuCharsetConfig() const {
        MaskedGPUCharsetConfig cfg;
        cfg.suffixLen = static_cast<uint32_t>(suffixLen_);
        cfg.compMode = static_cast<uint32_t>(compMode_);
        cfg.coinType = static_cast<uint32_t>(coinType_);
        cfg.forbidTripleSame = config_.forbidTripleSame ? 1 : 0;
        cfg.forbidTripleRun = config_.forbidTripleRun ? 1 : 0;
        std::memcpy(cfg.target, &targetBytes_[0], 20);

        for (size_t pos = 0; pos < suffixLen_; pos++) {
            cfg.radices[pos] = static_cast<uint8_t>(choices_[pos].size());
            cfg.bound[pos] = static_cast<uint8_t>(HexValue(maxValidHex_[prefixLen_ + pos]));
            for (size_t j = 0; j < choices_[pos].size(); j++) {
                cfg.values[pos][j] = choices_[pos][j].value;
                if (choices_[pos][j].value != 0) {
                    cfg.pointPresent[pos][j] = 1;
                    for (int k = 0; k < 4; k++) {
                        cfg.pointX[pos][j][k] = choices_[pos][j].point.x.bits64[k];
                        cfg.pointY[pos][j][k] = choices_[pos][j].point.y.bits64[k];
                    }
                }
            }
        }
        return cfg;
    }

    MaskedGPUTask BuildGpuTask(const Point& current, bool pointSet, bool hasNonZero, int cmpState, const TailState& state) const {
        MaskedGPUTask task;
        task.pointSet = pointSet ? 1 : 0;
        task.hasNonZero = hasNonZero ? 1 : 0;
        task.last1 = static_cast<int8_t>(state.last1);
        task.last2 = static_cast<int8_t>(state.last2);
        task.cmpState = static_cast<int8_t>(cmpState);
        task.startPos = static_cast<uint8_t>(taskDepth_);
        if (pointSet) {
            for (int i = 0; i < 4; i++) {
                task.baseX[i] = current.x.bits64[i];
                task.baseY[i] = current.y.bits64[i];
                task.baseZ[i] = current.z.bits64[i];
            }
        }
        return task;
    }

    void FillKeyFromIndex(std::string& keyHex, size_t startPos, uint64_t index) const {
        for (size_t pos = suffixLen_; pos-- > startPos;) {
            const std::vector<PositionChoice>& posChoices = choices_[pos];
            uint64_t radix = static_cast<uint64_t>(posChoices.size());
            uint64_t choiceIndex = index % radix;
            index /= radix;
            keyHex[prefixLen_ + pos] = posChoices[static_cast<size_t>(choiceIndex)].hex;
        }
    }

    bool VerifyGpuCandidate(const std::string& keyHex) {
        Int privKey;
        privKey.SetBase16(keyHex.c_str());
        Point pub = secp_.ComputePublicKey(&privKey);
        HeuristicEvaluation eval;
        eval.enabled = false;
        eval.valid = true;
        return CheckCandidate(keyHex, pub, eval);
    }

    MaskedGPUEngine* CreateGpuEngine() {
        if (config_.gpuIds.empty()) {
            throw std::runtime_error("Masked GPU V1 requires exactly one gpu id");
        }

        int gridX = -1;
        int gridY = 128;
        if (config_.gridSize.size() >= 2) {
            gridX = config_.gridSize[0];
            gridY = config_.gridSize[1];
        }

        MaskedGPUCharsetConfig cfg = BuildGpuCharsetConfig();
        MaskedGPUEngine* engine = new MaskedGPUEngine(config_.gpuIds[0], gridX, gridY, 256, cfg);
        if (!engine->IsInitialised()) {
            delete engine;
            throw std::runtime_error("Unable to initialize masked GPU engine");
        }
        gpuDeviceName_ = engine->GetDeviceName();
        return engine;
    }

    bool LaunchGpuTask(const std::string& keyHex,
                       const Point& current,
                       bool pointSet,
                       bool hasNonZero,
                       int cmpState,
                       const TailState& state,
                       MaskedGPUEngine& engine) {
        MaskedGPUTask task = BuildGpuTask(current, pointSet, hasNonZero, cmpState, state);
        std::vector<MaskedGPUHit> hits;
        uint64_t launched = 0;
        uint64_t batchSpan = static_cast<uint64_t>(engine.GetNbThread());
        if (batchSpan == 0) {
            throw std::runtime_error("Masked GPU engine returned zero thread capacity");
        }

        nextTask_.fetch_add(1);

        while (launched < gpuResidualCount_ && !ShouldStop()) {
            task.batchStart = launched;
            task.batchCount = std::min<uint64_t>(batchSpan, gpuResidualCount_ - launched);
            if (!engine.SearchBatch(task, hits)) {
                throw std::runtime_error("Masked GPU batch launch failed");
            }
            attempts_.fetch_add(task.batchCount);

            for (size_t i = 0; i < hits.size() && !ShouldStop(); i++) {
                std::string candidateHex = keyHex;
                FillKeyFromIndex(candidateHex, taskDepth_, launched + hits[i].localIndex);
                if (VerifyGpuCandidate(candidateHex)) {
                    return true;
                }
            }
            launched += task.batchCount;
        }

        return ShouldStop();
    }

    bool ExploreGpuTasks(size_t pos,
                         std::string& keyHex,
                         Point current,
                         bool pointSet,
                         bool hasNonZero,
                         int cmpState,
                         const TailState& state,
                         MaskedGPUEngine& engine) {
        if (ShouldStop()) {
            return true;
        }

        if (pos == taskDepth_) {
            return LaunchGpuTask(keyHex, current, pointSet, hasNonZero, cmpState, state, engine);
        }

        const size_t fullPos = prefixLen_ + pos;
        for (size_t i = 0; i < choices_[pos].size(); i++) {
            if (ShouldStop()) {
                return true;
            }

            const PositionChoice& choice = choices_[pos][i];
            int nextCmp = AdvanceCmpState(cmpState, fullPos, choice.hex);
            if (nextCmp > 0) {
                continue;
            }

            TailState nextState = state;
            if (!ApplyChoice(nextState, choice)) {
                continue;
            }

            keyHex[fullPos] = choice.hex;
            bool nextHasNonZero = hasNonZero || (choice.value != 0);

            if (choice.value == 0) {
                if (ExploreGpuTasks(pos + 1, keyHex, current, pointSet, nextHasNonZero, nextCmp, nextState, engine)) {
                    return true;
                }
            }
            else {
                Point nextPoint;
                bool nextPointSet = true;
                if (pointSet) {
                    Point addPoint = choice.point;
                    nextPoint = secp_.Add2(current, addPoint);
                }
                else {
                    nextPoint = choice.point;
                }

                if (ExploreGpuTasks(pos + 1, keyHex, nextPoint, nextPointSet, nextHasNonZero, nextCmp, nextState, engine)) {
                    return true;
                }
            }
        }

        return false;
    }

    void RunGPU(MaskedGPUEngine& engine) {
        std::string keyHex = prefix_ + std::string(suffixLen_, '0');
        Point current;
        current.Clear();
        bool pointSet = false;
        bool hasNonZero = prefixHasNonZero_;
        int cmpState = prefixCmpState_;
        TailState state;

        if (prefixPointSet_) {
            current = prefixPoint_;
            pointSet = true;
        }

        ExploreGpuTasks(0, keyHex, current, pointSet, hasNonZero, cmpState, state, engine);
    }
#endif

    void WorkerLoop() {
        uint64_t localAttempts = 0;
        std::string keyHex = prefix_ + std::string(suffixLen_, '0');

        while (!ShouldStop()) {
            uint64_t taskId = nextTask_.fetch_add(1);
            if (taskId >= taskCount_) {
                break;
            }

            Point current;
            current.Clear();
            bool pointSet = false;
            bool hasNonZero = prefixHasNonZero_;
            int cmpState = prefixCmpState_;
            TailState state;
            keyHex.assign(prefix_);
            keyHex.append(suffixLen_, '0');

            if (prefixPointSet_) {
                current = prefixPoint_;
                pointSet = true;
            }
            if (!ApplyTaskPrefix(taskId, keyHex, current, pointSet, hasNonZero, cmpState, state)) {
                continue;
            }

            if (Explore(taskDepth_, keyHex, current, pointSet, hasNonZero, cmpState, state, localAttempts)) {
                break;
            }
        }

        FlushAttempts(localAttempts);
    }

    bool CheckCandidate(const std::string& keyHex, Point& candidatePoint, const HeuristicEvaluation& eval) {
        Point reduced(candidatePoint);
        reduced.Reduce();

        unsigned char hash[20];
        bool matched = false;
        bool matchedCompressed = false;

        if (coinType_ == COIN_BTC) {
            if (compMode_ == SEARCH_COMPRESSED || compMode_ == SEARCH_BOTH) {
                secp_.GetHash160(true, reduced, hash);
                if (std::memcmp(hash, &targetBytes_[0], 20) == 0) {
                    matched = true;
                    matchedCompressed = true;
                }
            }
            if (!matched && (compMode_ == SEARCH_UNCOMPRESSED || compMode_ == SEARCH_BOTH)) {
                secp_.GetHash160(false, reduced, hash);
                if (std::memcmp(hash, &targetBytes_[0], 20) == 0) {
                    matched = true;
                    matchedCompressed = false;
                }
            }
        }
        else {
            secp_.GetHashETH(reduced, hash);
            matched = (std::memcmp(hash, &targetBytes_[0], 20) == 0);
        }

        if (!matched) {
            return false;
        }

        bool expected = false;
        if (!found_.compare_exchange_strong(expected, true)) {
            stop_.store(true);
            return true;
        }

        stop_.store(true);
        FlushFoundResult(keyHex, reduced, matchedCompressed, eval);
        return true;
    }

    void FlushFoundResult(const std::string& keyHex, Point& reducedPoint, bool matchedCompressed, const HeuristicEvaluation& eval) {
        std::lock_guard<std::mutex> guard(foundMutex_);

        Int privKey;
        privKey.SetBase16(keyHex.c_str());

        foundHex_ = keyHex;
        foundCompressed_ = matchedCompressed;
        foundAddress_ = targetLabel_;
        foundPubHex_ = (coinType_ == COIN_BTC)
            ? secp_.GetPublicKeyHex(matchedCompressed, reducedPoint)
            : secp_.GetPublicKeyHexETH(reducedPoint);
        if (coinType_ == COIN_BTC) {
            foundWif_ = secp_.GetPrivAddress(matchedCompressed, privKey);
        }
        foundEval_ = eval;
        WriteFoundOutput();
    }

    void WriteFoundOutput() {
        FILE* file = stdout;
        bool closeFile = false;

        if (!outputFile_.empty()) {
            file = std::fopen(outputFile_.c_str(), "a");
            if (file == NULL) {
                std::printf("  Cannot open %s for writing\n", outputFile_.c_str());
                file = stdout;
            }
            else {
                closeFile = true;
            }
        }

        std::fprintf(stdout, "\n  ================================================================================================\n");
        std::fprintf(stdout, "  PubAddress: %s\n", foundAddress_.c_str());
        if (coinType_ == COIN_BTC) {
            std::fprintf(stdout, "  Priv (WIF): %s\n", foundWif_.c_str());
        }
        std::fprintf(stdout, "  Priv (HEX): %s\n", foundHex_.c_str());
        std::fprintf(stdout, "  PubK (HEX): %s\n", foundPubHex_.c_str());
        std::fprintf(stdout, "  Mask prefix: %s\n", prefix_.c_str());
        std::fprintf(stdout, "  Mask sets  : %s\n", suffixSetsSpec_.c_str());
        std::fprintf(stdout, "  Prefix stat: digit=%d letter=%d pattern=[%s] adjpair=%d\n",
            prefixInfo_.digitCount, prefixInfo_.letterCount, prefixInfo_.patternKey.c_str(), prefixInfo_.adjacentRepeatPairs);
        if (foundEval_.enabled) {
            std::fprintf(stdout, "  Heuristic   : profile=%s/%s score(log)=%.4f\n",
                foundEval_.profileName.c_str(), profileReason_.c_str(), foundEval_.logScore);
            std::fprintf(stdout, "  Tail ratio  : %d:%d (p=%s)\n",
                foundEval_.tailDigits, foundEval_.tailLetters, FormatPercent(foundEval_.ratioProb).c_str());
            std::fprintf(stdout, "  Tail unique : %d (p=%s)\n",
                foundEval_.uniqueChars, FormatPercent(foundEval_.uniqueProb).c_str());
            std::fprintf(stdout, "  Tail mode   : %s (p=%s)\n",
                foundEval_.tailModeKey.c_str(), FormatPercent(foundEval_.tailModeProb).c_str());
            for (int bucketFreq = 1; bucketFreq <= 5; bucketFreq++) {
                if (prefixInfo_.bucketSizes[static_cast<size_t>(bucketFreq)] <= 0) continue;
                std::fprintf(stdout, "  Prefix %dx   : %d/%d chars hit in tail (p=%s)\n",
                    bucketFreq,
                    foundEval_.bucketHits[static_cast<size_t>(bucketFreq)],
                    prefixInfo_.bucketSizes[static_cast<size_t>(bucketFreq)],
                    FormatPercent(foundEval_.bucketHitProb[static_cast<size_t>(bucketFreq)]).c_str());
            }
        }
        std::fprintf(stdout, "\n");

        if (file != stdout) {
            std::fprintf(file, "PubAddress: %s\n", foundAddress_.c_str());
            if (coinType_ == COIN_BTC) {
                std::fprintf(file, "Priv (WIF): %s\n", foundWif_.c_str());
            }
            std::fprintf(file, "Priv (HEX): %s\n", foundHex_.c_str());
            std::fprintf(file, "PubK (HEX): %s\n", foundPubHex_.c_str());
            std::fprintf(file, "Mask prefix: %s\n", prefix_.c_str());
            std::fprintf(file, "Mask sets  : %s\n", suffixSetsSpec_.c_str());
            std::fprintf(file, "Prefix stat: digit=%d letter=%d pattern=[%s] adjpair=%d\n",
                prefixInfo_.digitCount, prefixInfo_.letterCount, prefixInfo_.patternKey.c_str(), prefixInfo_.adjacentRepeatPairs);
            if (foundEval_.enabled) {
                std::fprintf(file, "Heuristic: profile=%s/%s score(log)=%.4f\n",
                    foundEval_.profileName.c_str(), profileReason_.c_str(), foundEval_.logScore);
                std::fprintf(file, "Tail ratio: %d:%d (p=%s)\n",
                    foundEval_.tailDigits, foundEval_.tailLetters, FormatPercent(foundEval_.ratioProb).c_str());
                std::fprintf(file, "Tail unique: %d (p=%s)\n",
                    foundEval_.uniqueChars, FormatPercent(foundEval_.uniqueProb).c_str());
                std::fprintf(file, "Tail mode: %s (p=%s)\n",
                    foundEval_.tailModeKey.c_str(), FormatPercent(foundEval_.tailModeProb).c_str());
                for (int bucketFreq = 1; bucketFreq <= 5; bucketFreq++) {
                    if (prefixInfo_.bucketSizes[static_cast<size_t>(bucketFreq)] <= 0) continue;
                    std::fprintf(file, "Prefix %dx: %d/%d chars hit in tail (p=%s)\n",
                        bucketFreq,
                        foundEval_.bucketHits[static_cast<size_t>(bucketFreq)],
                        prefixInfo_.bucketSizes[static_cast<size_t>(bucketFreq)],
                        FormatPercent(foundEval_.bucketHitProb[static_cast<size_t>(bucketFreq)]).c_str());
                }
            }
            std::fprintf(file, "======================================================================================\n");
        }

        if (closeFile) {
            std::fclose(file);
        }
    }

    void PrintProbabilityTables() const {
        if (!config_.showProbabilities) {
            return;
        }
        std::printf("  PROB TABLE   : %s (%s)\n", profile_.name.c_str(), profileReason_.c_str());
        std::printf("  Ratio table  : tail digits in [%d,%d] => %s\n",
            minSupportedDigits_, maxSupportedDigits_,
            BuildProbabilityEntryList(profile_.tailDigitProb, ", ").c_str());
        std::printf("  Unique table : tail unique in [%d,%d] => %s\n",
            minSupportedUnique_, maxSupportedUnique_,
            BuildProbabilityEntryList(profile_.uniqueCountProb, ", ").c_str());
        std::printf("  Tail modes   : %s\n",
            BuildTailModeSummary(profile_.tailModeProb).c_str());
        std::printf("  Tail fallback: %s%s\n",
            FormatPercent(profile_.unknownTailModeProb).c_str(),
            config_.strictTailModes ? " (strict mode: fallback disabled)" : "");
        for (int bucketFreq = 1; bucketFreq <= 5; bucketFreq++) {
            if (prefixInfo_.bucketSizes[static_cast<size_t>(bucketFreq)] <= 0) continue;
            const std::vector<double>& probs = profile_.prefixBucketHitProb[static_cast<size_t>(bucketFreq)];
            std::printf("  Prefix %dx hit: %s\n", bucketFreq, BuildProbabilityEntryList(probs, ", ").c_str());
        }
    }

    void PrintStartInfo() const {
        std::printf("  MASK MODE    : SUFFIX CHARSETS (%s)\n", config_.useGpu ? "GPU V1" : "CPU");
        std::printf("  COIN TYPE    : %s\n", CoinName(coinType_));
        if (coinType_ == COIN_BTC) {
            std::printf("  COMP MODE    : %s\n", CompModeName(compMode_));
        }
        std::printf("  TARGET       : %s\n", targetLabel_.c_str());
        std::printf("  KNOWN PREFIX : %s\n", prefix_.c_str());
        std::printf("  PREFIX INFO  : len=%zu digit=%d letter=%d pattern=[%s] adjpair=%d missing=%zu [%s]\n",
            prefix_.size(), prefixInfo_.digitCount, prefixInfo_.letterCount,
            prefixInfo_.patternKey.c_str(), prefixInfo_.adjacentRepeatPairs, prefixInfo_.missingChars.size(),
            prefixInfo_.missingChars.empty() ? "-" : prefixInfo_.missingChars.c_str());
        std::printf("  UNKNOWN TAIL : %u hex chars\n", static_cast<unsigned>(suffixLen_));
        std::printf("  SUFFIX SETS  : %s\n", suffixSetsSpec_.c_str());
        std::printf("  RULES        : no3same=%s no3seq=%s\n",
            config_.forbidTripleSame ? "ON" : "OFF",
            config_.forbidTripleRun ? "ON" : "OFF");
        if (config_.useGpu) {
            long double residualLog2 = 0.0L;
            for (size_t pos = taskDepth_; pos < suffixLen_; pos++) {
                residualLog2 += std::log(static_cast<long double>(choices_[pos].size())) / std::log(2.0L);
            }
            std::printf("  HOST THREADS : %d\n", threadCount_);
            std::printf("  GPU DEVICE   : %s\n", gpuDeviceName_.empty() ? "(initializing)" : gpuDeviceName_.c_str());
            std::printf("  TASK SPLIT   : host-prefix=%u chars (%s tasks), gpu-tail=%u chars (%s per task)%s\n",
                static_cast<unsigned>(taskDepth_),
                FormatThousands(taskCount_).c_str(),
                static_cast<unsigned>(suffixLen_ - taskDepth_),
                FormatKeyspace(residualLog2).c_str(),
                taskCountClipped_ ? " [task-count clipped]" : "");
        }
        else {
            std::printf("  THREADS      : %d\n", threadCount_);
            std::printf("  TASK DEPTH   : %u leading masked chars (%s tasks)\n",
                static_cast<unsigned>(taskDepth_),
                FormatThousands(taskCount_).c_str());
        }
        std::printf("  KEYSPACE     : %s\n", FormatKeyspace(keyspaceLog2_).c_str());
        std::printf("  OUTPUT FILE  : %s\n", outputFile_.c_str());
        if (profileActive_) {
            PrintProbabilityTables();
        }
        else if (config_.showProbabilities) {
            if (config_.useGpu) {
                std::ostringstream oss;
                if (gpuProfileIgnored_) {
                    oss << "probability pruning disabled in GPU V1 (requested=" << config_.probabilityProfile << ")";
                }
                else {
                    oss << "disabled (--prob-profile none)";
                }
                if (gpuStrictTailModeIgnored_) {
                    oss << "; strict tailmode ignored in GPU V1";
                }
                std::printf("  PROB TABLE   : %s\n", oss.str().c_str());
            }
            else {
                std::printf("  PROB TABLE   : no built-in profile matched this prefix (requested=%s)\n", config_.probabilityProfile.c_str());
            }
        }
        std::printf("\n");
    }

    void ProgressLoop() {
        using clock = std::chrono::steady_clock;
        clock::time_point lastTs = clock::now();
        uint64_t lastAttempts = 0;

        while (!stop_.load() && !should_exit_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            uint64_t currentAttempts = attempts_.load();
            clock::time_point now = clock::now();
            double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(now - lastTs).count();
            if (seconds <= 0.0) {
                continue;
            }

            double rate = static_cast<double>(currentAttempts - lastAttempts) / seconds;
            std::printf("\r  [MASK] [Tried: %s] [Rate: %s] [Tasks: %s/%s]    ",
                FormatThousands(currentAttempts).c_str(),
                FormatRate(rate).c_str(),
                FormatThousands(std::min<uint64_t>(nextTask_.load(), taskCount_)).c_str(),
                FormatThousands(taskCount_).c_str());
            std::fflush(stdout);

            lastAttempts = currentAttempts;
            lastTs = now;
        }
    }

private:
    std::string targetLabel_;
    std::vector<unsigned char> targetBytes_;
    int compMode_;
    int searchMode_;
    int coinType_;
    std::string outputFile_;
    std::string prefixRaw_;
    std::string suffixSetsSpec_;
    int threadCount_;
    int display_;
    MaskedSearchConfig config_;
    bool& should_exit_;

    Secp256K1 secp_;
    std::string prefix_;
    std::vector<std::string> sets_;
    std::vector< std::vector<PositionChoice> > choices_;
    PrefixInfo prefixInfo_;
    size_t prefixLen_;
    size_t suffixLen_;
    Point prefixPoint_;
    bool prefixPointSet_;
    bool prefixHasNonZero_;
    int prefixCmpState_;
    std::string maxValidHex_;
    size_t taskDepth_;
    uint64_t taskCount_;
    long double keyspaceLog2_;
    std::vector<int> remMinDigits_;
    std::vector<int> remMaxDigits_;
    std::vector<uint16_t> unionMaskFromPos_;

    ProbabilityProfile profile_;
    bool profileActive_;
    std::string profileReason_;
    int minSupportedDigits_;
    int maxSupportedDigits_;
    int minSupportedUnique_;
    int maxSupportedUnique_;
    bool gpuProfileIgnored_;
    bool gpuStrictTailModeIgnored_;
    std::string gpuDeviceName_;
    uint64_t gpuResidualCount_;
    bool taskCountClipped_;

    std::atomic<bool> found_;
    std::atomic<bool> stop_;
    std::atomic<uint64_t> attempts_;
    std::atomic<uint64_t> nextTask_;
    std::mutex foundMutex_;

    std::string foundHex_;
    std::string foundAddress_;
    std::string foundWif_;
    std::string foundPubHex_;
    bool foundCompressed_;
    HeuristicEvaluation foundEval_;
};

} // namespace

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
    bool& should_exit) {

    try {
        MaskedSearchRunner runner(targetLabel, targetBytes, compMode, searchMode, coinType,
            outputFile, prefix, suffixSetsSpec, threadCount, display, config, should_exit);
        return runner.Run();
    }
    catch (const std::exception& ex) {
        std::printf("  Masked search error: %s\n", ex.what());
        return -1;
    }
}
