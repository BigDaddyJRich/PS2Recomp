#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <array>
#include <cstdint>
#include <vector>

class GS;
class PS2Memory;
class VU1Interpreter;

// VU1 code recompiled ahead of time (runtime/ps2_vu1_compiled.h)
enum class Vu1CompiledResult : uint8_t;
using Vu1CompiledFn = Vu1CompiledResult (*)(VU1Interpreter &vu, uint64_t budgetEnd, uint64_t &pairsRun);
struct Vu1Compiled;
struct Vu1CompiledPiece;

struct VU1State
{
    float vf[32][4];
    int32_t vi[16];
    float acc[4];
    float q;
    float p;
    float i;
    uint32_t r;
    uint32_t pc;
    uint32_t mac;
    uint32_t clip;
    uint32_t status;
    uint64_t cycles;
    bool ebit;
    bool haltAfterDelaySlot;
    bool dBitEnabled;
    bool tBitEnabled;
    bool stoppedByD;
    bool stoppedByT;
    uint32_t top;  // VIF TOP visible to XTOP
    uint32_t itop; // VIF ITOP visible to XITOP

    bool branchPending;
    uint32_t branchTarget;
    uint32_t branchDelay;
};

class VU1Interpreter
{
    friend struct Vu1Compiled; // the compiled VU1 code's access (runtime/ps2_vu1_compiled.h)

public:
    enum class Unit : uint8_t
    {
        VU0,
        VU1
    };

    explicit VU1Interpreter(Unit unit = Unit::VU1);

    void reset();

    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t top = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t top = 0, uint32_t itop = 0, uint32_t maxCycles = 65536);

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }

private:
    enum Pipeline : uint8_t
    {
        PipelineNone = 0,
        PipelineFmac,
        PipelineLsu,
        PipelineFdiv,
        PipelineEfu,
        PipelineIalu,
        PipelineBranch,
        PipelineXgkick
    };

    struct VfAccess
    {
        uint8_t reg = 0;
        uint8_t lanes = 0;
    };

    struct InstructionUsage
    {
        std::array<VfAccess, 2> vfRead{};
        VfAccess vfWrite{};
        uint8_t vfReadCount = 0;
        uint16_t viRead = 0;
        uint16_t viWrite = 0;
        uint8_t accRead = 0;
        uint8_t accWrite = 0;
        uint8_t latency = 0;
        uint8_t vfLatency = 0;
        uint8_t viLatency = 0;
        Pipeline pipeline = PipelineNone;
        bool waitQ = false;
        bool waitP = false;
        bool readsClip = false;
        bool writesClip = false;
        bool delaysNextBranchRead = false;
        bool reserved = false;
    };

    struct DecodedInstructionPair
    {
        uint32_t lower = 0;
        uint32_t upper = 0;
        InstructionUsage lowerUsage{};
        InstructionUsage upperUsage{};
        bool iBit = false;
        bool eBit = false;
        bool mBit = false;
        bool dBit = false;
        bool tBit = false;
        uint8_t upperVfShadowReg = 0;
        uint8_t suppressedLowerVf = 0;
        uint8_t lowerViWriteReg = 0; // the lowest VI register the lower instruction writes (0 = none)

        // Precomputed from the two usages for the stall check and the write marking (same rules, no loops over
        // every register / lane per pair).
        std::array<VfAccess, 4> stallReads{}; // VF reads of both halves (vf0 left out: it is never written)
        uint8_t stallReadCount = 0;
        uint16_t stallViReads = 0;   // VI registers 1-15 read by either half
        uint8_t stallAccReads = 0;   // ACC lanes read by either half
        VfAccess markLowerVf{};      // the lower VF write (not when the upper's write to the same register wins)
        uint8_t markLowerVfLatency = 0;
        VfAccess markUpperVf{};
        uint8_t markUpperVfLatency = 0;
        uint16_t markViWrites = 0;   // VI registers 1-15 the lower writes
        uint8_t markViLatency = 0;
        uint8_t markAccWrites = 0;   // ACC lanes the upper writes
    };

    // The pipeline entries are cleared field by field with clear(): `entry = {}` builds a zeroed temporary and copies
    // it, and the copy's wide reads of fresh narrow stores stall (for XgkickPipeline it was a 64 KB memset + memcpy on
    // every XGKICK).
    struct FlagPipelineEntry
    {
        uint64_t readyCycle = 0;
        uint64_t issueCycle = 0;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t extraSticky = 0;
        uint32_t clip = 0;
        bool valid = false;
        bool writesMac = false;
        bool writesStatus = false;
        bool writesSticky = false;
        bool writesClip = false;

        void clear() noexcept
        {
            readyCycle = 0;
            issueCycle = 0;
            mac = 0;
            status = 0;
            extraSticky = 0;
            clip = 0;
            valid = false;
            writesMac = false;
            writesStatus = false;
            writesSticky = false;
            writesClip = false;
        }
    };

    struct ScalarPipelineEntry
    {
        uint64_t readyCycle = 0;
        float value = 0.0f;
        uint32_t statusDi = 0;
        bool valid = false;

        void clear() noexcept
        {
            readyCycle = 0;
            value = 0.0f;
            statusDi = 0;
            valid = false;
        }
    };

    struct PendingStore
    {
        uint64_t readyCycle = 0;
        uint32_t address = 0;
        std::array<uint32_t, 4> words{};
        uint8_t laneMask = 0;
        bool valid = false;

        void clear() noexcept
        {
            readyCycle = 0;
            address = 0;
            words[0] = words[1] = words[2] = words[3] = 0u;
            laneMask = 0;
            valid = false;
        }
    };

    struct PendingVfWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        std::array<float, 4> value{};
        uint8_t reg = 0;
        uint8_t laneMask = 0;
        bool valid = false;

        void clear() noexcept
        {
            readyCycle = 0;
            sequence = 0;
            value[0] = value[1] = value[2] = value[3] = 0.0f;
            reg = 0;
            laneMask = 0;
            valid = false;
        }
    };

    struct PendingViWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        int32_t value = 0;
        uint8_t reg = 0;
        bool valid = false;

        void clear() noexcept
        {
            readyCycle = 0;
            sequence = 0;
            value = 0;
            reg = 0;
            valid = false;
        }
    };

    struct PendingAccWrite
    {
        uint64_t readyCycle = 0;
        uint64_t sequence = 0;
        std::array<float, 4> value{};
        uint8_t laneMask = 0;
        bool valid = false;

        void clear() noexcept
        {
            readyCycle = 0;
            sequence = 0;
            value[0] = value[1] = value[2] = value[3] = 0.0f;
            laneMask = 0;
            valid = false;
        }
    };

    struct XgkickPipeline
    {
        static constexpr uint32_t kBufferSize = 0x10000u;
        std::array<uint8_t, kBufferSize> packet{};
        uint32_t sourceAddress = 0;
        uint32_t totalBytes = 0;
        uint32_t copiedBytes = 0;
        uint32_t currentTagEnd = 0;
        uint32_t cycleCredit = 0;
        uint64_t issueCycle = 0;
        bool active = false;
        bool currentTagEop = false;

        // the packet bytes are left as they are: only the bytes copied in since the XGKICK are ever read
        void clear() noexcept
        {
            sourceAddress = 0;
            totalBytes = 0;
            copiedBytes = 0;
            currentTagEnd = 0;
            cycleCredit = 0;
            issueCycle = 0;
            active = false;
            currentTagEop = false;
        }
    };

    static constexpr uint32_t kFmacLatency = 4u;
    static constexpr uint32_t kAccForwardLatency = 1u;
    static constexpr uint32_t kMaxFlagEntries = 8u;
    static constexpr uint32_t kMaxPendingStores = 8u;
    static constexpr uint32_t kMaxPendingVfWrites = 16u;
    static constexpr uint32_t kMaxPendingViWrites = 8u;
    static constexpr uint32_t kMaxPendingAccWrites = 8u;
    static constexpr uint32_t kMaxDecodedPairs = 0x4000u / 8u;

    Unit m_unit;
    VU1State m_state;
    std::array<DecodedInstructionPair, kMaxDecodedPairs> m_decodedCodeCache{};
    const uint8_t *m_cachedVuCode = nullptr;
    const PS2Memory *m_cachedMemory = nullptr;
    uint32_t m_cachedCodeSize = 0;
    uint64_t m_cachedCodeGeneration = 0;
    bool m_decodedCodeCacheValid = false;
    bool m_codeReadsMacStatus = true; // the decoded code has FSEQ / FSAND / FSOR / FMEQ / FMAND / FMOR
    // Compiled VU1 code (runtime/ps2_vu1_compiled.h): per pair, the compiled function to enter there for the code
    // now in VU memory (nullptr = interpret); refreshed with the decode cache.
    std::array<Vu1CompiledFn, kMaxDecodedPairs> m_compiledFnAt{};
    bool m_compiledEnabled = true; // VENDETTA_VU1_COMPILED=0 turns the compiled VU1 code off
    void refreshCompiledEntries(const uint8_t *vuCode, uint32_t codeSize);
    bool compiledShapesMatch(const Vu1CompiledPiece &piece, uint32_t &badPair) const;
    // the decode cache for the code in VU memory, rebuilt if the code changed; false = untracked code memory
    bool ensureDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory);

    std::array<FlagPipelineEntry, kMaxFlagEntries> m_flagPipeline{};
    ScalarPipelineEntry m_fdiv{};
    std::array<ScalarPipelineEntry, 2> m_efu{};
    std::array<PendingStore, kMaxPendingStores> m_storePipeline{};
    // Which slots of m_flagPipeline / m_storePipeline hold an entry (bit i = slot i, kept in every mode). Fast mode
    // allocates and commits by them; the original model scans every slot as before (VENDETTA_VU1_VERIFY compares
    // the two).
    uint8_t m_flagSlotsUsed = 0;
    uint8_t m_storeSlotsUsed = 0;
    static_assert(kMaxFlagEntries <= 8u && kMaxPendingStores <= 8u);
    template <typename Entry, size_t N>
    Entry *takeFreeSlot(std::array<Entry, N> &slots, uint8_t &used);
    std::array<PendingVfWrite, kMaxPendingVfWrites> m_vfWritePipeline{};
    std::array<PendingViWrite, kMaxPendingViWrites> m_viWritePipeline{};
    std::array<PendingAccWrite, kMaxPendingAccWrites> m_accWritePipeline{};
    XgkickPipeline m_xgkick{};
    std::array<std::array<uint64_t, 4>, 32> m_vfReady{};
    std::array<uint64_t, 16> m_viReady{};
    std::array<uint64_t, 4> m_accReady{};
    std::array<std::array<uint64_t, 4>, 32> m_vfLatestWrite{};
    std::array<uint64_t, 16> m_viLatestWrite{};
    std::array<uint64_t, 4> m_accLatestWrite{};

    uint64_t m_cycle = 0;
    uint64_t m_nextWriteSequence = 0;
    uint64_t m_efuResourceReady = 0;
    // The earliest readyCycle of any queued pipeline result (0 = unknown, scan next cycle). Lets
    // commitReadyPipelines() skip its scan of every pipeline slot on cycles where nothing is due.
    uint64_t m_nextCommitCycle = 0;
    void noteReadyCycle(uint64_t cycle) noexcept
    {
        if (cycle < m_nextCommitCycle)
            m_nextCommitCycle = cycle;
    }
    // an address in the active VU data memory (a power-of-two size wraps with a mask, not a division)
    uint32_t wrapVuDataAddress(uint32_t address) const noexcept
    {
        const uint32_t size = m_activeVuDataSize;
        return (size & (size - 1u)) == 0u ? (address & (size - 1u)) : (address % size);
    }
    uint32_t m_workingClip = 0;
    uint32_t m_currentUpperInstruction = 0;
    int32_t m_viBranchBackupValue = 0;
    uint8_t m_viBranchBackupReg = 0;
    bool m_viBranchBackupValid = false;
    uint8_t *m_activeVuData = nullptr;
    uint32_t m_activeVuDataSize = 0;
    GS *m_activeGs = nullptr;
    PS2Memory *m_activeMemory = nullptr;
    bool m_stopRequested = false;
    bool m_pendingHaltD = false;
    bool m_pendingHaltT = false;

    void run(uint8_t *vuCode, uint32_t codeSize,
             uint8_t *vuData, uint32_t dataSize,
             GS &gs, PS2Memory *memory, uint32_t maxCycles);
    void runLoop(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory, uint32_t maxCycles);
    void runVerified(uint8_t *vuCode, uint32_t codeSize,
                     uint8_t *vuData, uint32_t dataSize,
                     GS &gs, PS2Memory *memory, uint32_t maxCycles);
    void drainRegisterWrites();

    // Fast mode (default; VENDETTA_VU1_FAST=0 = off): results are written to the registers when the instruction
    // runs, not queued and replayed when their latency is up. Equivalent because every reader of a pending result
    // stalls until it is ready (m_vfReady / m_viReady / m_accReady are still kept); Q, P, the flags, stores and
    // XGKICK keep their timed pipelines. VENDETTA_VU1_VERIFY=1 runs every program both ways and compares.
    bool m_fastMode = true;
    bool m_verifying = false;
    uint64_t m_registerWritesSettleCycle = 0; // fast mode: when the last register write would have landed
    std::vector<std::vector<uint8_t>> *m_kickCapture = nullptr; // XGKICK packets go here instead of to the GS
    DecodedInstructionPair m_decodeScratch{};

    InstructionUsage decodeUpperUsage(uint32_t upper) const;
    InstructionUsage decodeLowerUsage(uint32_t lower) const;
    static void addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static void addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes);
    static uint8_t vfReadLanes(const InstructionUsage &usage, uint8_t reg);
    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    const DecodedInstructionPair &getDecodedInstructionPairForPc(const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc);
    void rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize, const PS2Memory *memory, uint64_t generation);

    void execUpper(uint32_t instr);
    // Fast mode: the common upper instructions without the MAC / status flag work (only when the code in VU
    // memory never reads those flags) and with only the lanes they use; false = not handled, use execUpper.
    bool execUpperFast(uint32_t instr);
    void runUpper(uint32_t instr)
    {
        if (!(m_fastMode && m_decodedCodeCacheValid && !m_codeReadsMacStatus && execUpperFast(instr)))
            execUpper(instr);
    }
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    void applyFmacDest(float *dst, float *result, uint8_t dest);
    void applyFmacDestAcc(float *result, uint8_t dest);
    void normalizeFmacResult(float *result, uint8_t dest, uint8_t laneFlags[4]);
    bool calculateFmacExactResult(uint32_t component, long double &result) const;
    uint8_t normalizeFmacExactResult(float &value, long double exactResult) const;
    uint32_t calculateFmacProductSticky(uint8_t dest) const;
    void updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest, uint32_t extraSticky);
    void queueFsset(uint16_t immediate);
    void queueClip(uint32_t clip);
    void queueFcset(uint32_t clip);
    void queueQ(float value, uint32_t latency, uint32_t statusDi);
    void queueP(float value, uint32_t latency);
    void queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask);
    void queueVfWrite(uint8_t reg, uint8_t laneMask, const float value[4], uint32_t latency);
    void queueViWrite(uint8_t reg, int32_t value, uint32_t latency);
    void queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency);
    void startXgkick(uint32_t qwordAddress);

    void resetScheduler();
    void commitReadyPipelines();
    void commitReadyPipelinesScan(); // the original model's commit: every slot scanned
    void advanceOneCycle();
    void advanceTo(uint64_t targetCycle);
    void flushPipelines();
    void progressXgkick();
    void finishXgkick();
    uint64_t calculatePairReadyCycle(const DecodedInstructionPair &decoded) const;
    void markPairWrites(const DecodedInstructionPair &decoded);
    bool pipelinesPending() const;

    float normalizeOperand(float value) const;
    float normalizeResult(float value, uint32_t &laneFlags) const;
    uint32_t microAddressMask() const;
    int32_t readBranchVi(uint8_t reg) const;
    void recordViWriteForBranch(uint8_t reg, int32_t oldValue);
    void reportReservedInstruction(bool upper, uint32_t instruction);
    float broadcast(const float *vf, uint8_t bc);
};

#endif
