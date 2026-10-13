#pragma once

#include "runtime/gs/gs_presentation.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

class PS2Runtime;

// Why the game thread waits for the chain worker (counted).
enum class ChainFence : uint8_t
{
    Csr,      // a CSR read / write (SIGNAL / FINISH are set by the GS front end)
    Siglblid, // SIGLBLID (SIGNAL / LABEL)
    GifStat,  // GIF_STAT's path-3 mask bit
    VuMemory, // the EE reading / writing VU1 code or data memory
    GsCall,   // a GS:: call from outside the worker (HLE stubs)
    DtKick,   // a VIF1 kick with VU1 D / T bits enabled (VPU_STAT must be exact at once)
    Sync,     // VENDETTA_CHAIN_SYNC=1
};

// The VIF1 -> VU1 -> GS front end chain on its own thread. The game thread keeps
// everything the game reads — DMA registers, D_STAT and the DMA-end interrupts, CSR.FIELD, the display registers —
// and completes every DMA at the kick, as before; it hands the chain the bytes the transfer carries (copied at the
// kick) as records in one ordered ring. The worker owns VIF1, VU1 and its memory, the GIF arbiter and the GS front
// end (it is the GS renderer's producer); it runs the records in order with the same code. The game thread reads
// worker-owned state only after a fence (wait until the worker has run everything handed to it).
// The worker is the default; VENDETTA_CHAIN_ASYNC=0 runs every record at once on the game thread instead (the
// earlier behaviour).
class ChainWorker
{
public:
    explicit ChainWorker(PS2Runtime &runtime);
    ~ChainWorker();
    ChainWorker(const ChainWorker &) = delete;
    ChainWorker &operator=(const ChainWorker &) = delete;

    // Before the game thread starts: reads the environment, starts ChainThread when asynchronous.
    void start();
    // After the game thread has ended (idempotent): the worker finishes its record and stops.
    void shutdown();
    // requestStop: releases the game thread's waits.
    void wakeWaiters();
    bool async() const noexcept { return m_async; }
    static bool onWorker() noexcept;
    // The game thread (the ring's one producer).
    static void markProducerThread() noexcept;

    // The producer: what a DMA kick (or a register write, or the vblank settle point) hands the chain, in order.
    void submitGif(const uint8_t *data, uint32_t sizeBytes);  // PATH3 data (the GIF channel), drained later
    void submitVif1(const uint8_t *data, uint32_t sizeBytes); // a VIF1 stream piece (the VIF1 channel / FIFO)
    void drainArbiter();                                      // the end of a kick: the GIF arbiter's queue to the GS
    void writeVif1Register(uint32_t address, uint32_t value); // VIF1_FBRST ... VIF1_TOP (0x10003C10–0x10003CE0)
    void present(const GSDisplayRegs &regs, uint64_t vsyncTick);
    // Wait until the worker has run everything handed to it (the producer only; nothing to do when inline).
    void fence(ChainFence reason);
    // The chain throttle (the producer, holding no lock): wait while more than maxAhead presentations are still to
    // run; false = stopped.
    bool waitBacklog(uint32_t maxAhead, const std::atomic<bool> &stop);

    // For the VU1 MSCAL / MSCNT callbacks (they run where the VIF1 stream runs): VIF1's FBRST bits (DE1 / TE1)
    // captured at the kick, and the stop bits a run leaves for VPU_STAT.
    uint32_t kickFbrst() const noexcept { return m_kickFbrst; }
    void noteVu1Run(uint32_t vpuStatStopBits);

    // A game-specific CSR read site that uses only CSR.FIELD (bit 13, the game thread's own): it reads without a fence
    // (before the game starts; VENDETTA_CHAIN_CSR_EXEMPT=0 ignores these). Up to 8 sites.
    void allowUnfencedCsrRead(uint32_t pc) noexcept;
    bool csrReadExempt(uint32_t pc) const noexcept;
    // VENDETTA_CHAIN_VERIFY=1: an exempt read is checked against a fenced read of the same register.
    bool verifyExemptReads() const noexcept { return m_verifyExempt; }

private:
    enum class Cmd : uint16_t
    {
        Wrap = 0,
        Gif,
        GifHeap,
        Vif1,
        Vif1Heap,
        DrainArbiter,
        Vif1Register,
        Present,
    };
    struct alignas(16) Header
    {
        uint16_t cmd = 0;
        uint16_t flags = 0;
        uint32_t bytes = 0; // payload bytes (the record is padded to 16)
        uint32_t aux = 0;   // Vif1: the FBRST bits
        uint32_t mxcsr = 0; // the producer's rounding mode at the push
    };
    static_assert(sizeof(Header) == 16u);

    // Records run the same code in both modes.
    void runGif(const uint8_t *data, uint32_t sizeBytes);
    void runVif1(const uint8_t *data, uint32_t sizeBytes, uint32_t fbrst);
    void runPresent(const uint8_t *payload);

    // producer
    uint8_t *reserve(Cmd cmd, uint32_t payloadBytes);
    void commit();
    void publish();
    void pushData(Cmd inlineCmd, Cmd heapCmd, const uint8_t *data, uint32_t sizeBytes, uint32_t aux);
    template <typename Pred>
    void waitProgress(const char *what, Pred pred);
    uint32_t currentFbrst() const;
    bool needsDtSync() const;

    // worker
    void threadMain();
    void execute(const Header &header, const uint8_t *payload);
    void publishProgress(uint64_t readPos);

    PS2Runtime &m_runtime;
    uint32_t m_exemptCsrPcs[8] = {};
    uint32_t m_exemptCsrCount = 0u;
    bool m_verifyExempt = false;
    bool m_async = false;
    bool m_sync = false;
    uint32_t m_slowPresentUs = 0u;

    std::unique_ptr<uint8_t[]> m_ringStorage;
    uint8_t *m_ring = nullptr;
    uint64_t m_size = 0u;
    uint64_t m_mask = 0u;
    alignas(64) std::atomic<uint64_t> m_writePos{0u};
    alignas(64) std::atomic<uint64_t> m_readPos{0u};
    // producer only
    alignas(64) uint64_t m_localWrite = 0u;
    uint64_t m_published = 0u;
    uint64_t m_cachedRead = 0u;
    uint64_t m_pendingRecordBytes = 0u;
    uint64_t m_presentsPushed = 0u;
    // worker only (and the producer when inline)
    alignas(64) uint32_t m_kickFbrst = 0u;
    uint32_t m_runStopBits = 0u;
    bool m_runReported = false;
    bool m_reportedException = false;

    alignas(64) std::atomic<uint32_t> m_consumerSleeping{0u};
    std::atomic<uint32_t> m_wakeEpoch{0u};
    std::atomic<uint32_t> m_producerWaiting{0u};
    std::mutex m_progressMutex;
    std::condition_variable m_progressCv;
    std::atomic<uint64_t> m_presentsDone{0u};
    std::atomic<uint32_t> m_currentCmd{0u};
    std::atomic<bool> m_stop{false};
    bool m_shutDown = false;
    std::thread m_thread;
};
