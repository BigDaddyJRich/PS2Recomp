#pragma once

#include "runtime/gs/gs_cpu_backend.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// The GS renderer on its own thread (the default; VENDETTA_GS_ASYNC=0 draws on the game thread). The GS front end stays on the game thread (register state,
// GIF parsing, CSR / SIGNAL / FINISH: everything the game can read is produced as before); only the renderer moves.
// Its calls go, in order, into a single-producer / single-consumer ring of command records, and the GS thread runs
// them on the GSCpuBackend it owns. Calls that return something wait until every earlier command has run, then run
// on the calling thread (the GS thread is idle then; GSCpuBackend locks its own state).
class GSThreadedBackend final : public GSRasterBackend
{
public:
    struct Options
    {
        size_t ringBytes = size_t{8} << 20u; // VENDETTA_GS_RING_MB (a power of two, >= 1 MiB)
        bool sync = false;                   // VENDETTA_GS_SYNC=1: wait for the GS thread after every call (debug)
        uint32_t slowPresentUs = 0u;         // VENDETTA_GS_SLOW=us: sleep before each presentation (stress)
        bool verifyStream = false;           // VENDETTA_GS_VERIFY: hash the commands on both sides
    };

    GSThreadedBackend(std::unique_ptr<GSCpuBackend> inner, const Options &options);
    ~GSThreadedBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

    void PresentToMailbox(const GSPresentationRequest &request, GSPresentTicket ticket, GSPresentMailbox &box) override;
    void ResetPresentation(GSPresentMailbox &box, uint64_t seq) override;
    void VerifyPages(const std::array<uint64_t, 512> &expected, uint64_t check, uint64_t streamHash) override;
    void Kick() override;
    bool WaitFlipBacklog(uint32_t maxAhead, const std::atomic<bool> &stop) override;
    void WakeWaiters() override;
    void Shutdown() override;

private:
    enum class Cmd : uint16_t
    {
        Wrap = 0, // fills the rest of the ring; the next record starts at its beginning
        Submit,
        LoadClut,
        BeginTransfer,
        UploadInline,
        UploadHeap,
        Flush,
        TextureFlush,
        WriteVram,
        Clear,
        Reset,
        Present,
        ResetPresentation,
        Verify,
    };
    struct alignas(16) Header
    {
        uint16_t cmd = 0;
        uint16_t flags = 0;
        uint32_t bytes = 0; // payload bytes (the record is padded to 16)
        uint64_t aux = 0;   // Verify: the producer's command-stream hash
    };
    static_assert(sizeof(Header) == 16u);

    // producer (whoever holds GS::m_stateMutex)
    uint8_t *reserve(Cmd cmd, uint32_t payloadBytes);
    void commit();
    template <typename T>
    void push(Cmd cmd, const T &payload);
    void pushEmpty(Cmd cmd);
    void publish();
    void drain();
    void drain() const { const_cast<GSThreadedBackend *>(this)->drain(); }
    template <typename Pred>
    void waitProgress(const char *what, Pred pred);

    // GS thread
    void threadMain();
    void execute(const Header &header, const uint8_t *payload);
    void publishProgress(uint64_t readPos);

    std::unique_ptr<GSCpuBackend> m_inner;
    Options m_options;
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
    uint64_t m_pendingCmds = 0u;
    uint64_t m_pendingBytes = 0u;
    uint64_t m_producerStream = 0u;
    bool m_hasVram = false;
    std::atomic<uint64_t> m_flipsIssued{0u};

    // GS thread only
    alignas(64) uint64_t m_consumerStream = 0u;
    bool m_reportedException = false;

    // waking the GS thread (atomic wait) and the producer (condition variable, with a stall report)
    alignas(64) std::atomic<uint32_t> m_consumerSleeping{0u};
    std::atomic<uint32_t> m_wakeEpoch{0u};
    std::atomic<uint32_t> m_producerWaiting{0u};
    std::mutex m_progressMutex;
    std::condition_variable m_progressCv;

    std::atomic<uint64_t> m_flipsDone{0u};
    std::atomic<uint32_t> m_currentCmd{0u};
    std::atomic<bool> m_stop{false};
    bool m_shutDown = false;
    std::thread m_thread;
};

// VENDETTA_GS_VERIFY=1[,N]: every renderer call goes to the configured renderer and to a 1-band shadow GSCpuBackend
// with its own GS memory, on the producer. Values the renderer returns are compared, and at every presentation (and
// every N calls) the shadow's GS memory page hashes are checked against the renderer's at the same point in the order.
class GSTeeBackend final : public GSRasterBackend
{
public:
    GSTeeBackend(std::unique_ptr<GSRasterBackend> primary, uint32_t checkEvery);

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;
    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;
    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;
    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;
    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;
    void PresentToMailbox(const GSPresentationRequest &request, GSPresentTicket ticket, GSPresentMailbox &box) override;
    void ResetPresentation(GSPresentMailbox &box, uint64_t seq) override;
    void Kick() override;
    bool WaitFlipBacklog(uint32_t maxAhead, const std::atomic<bool> &stop) override;
    void WakeWaiters() override;
    void Shutdown() override;

private:
    void called();
    void check();
    void mismatch(const char *what) const;

    std::unique_ptr<GSRasterBackend> m_primary;
    std::unique_ptr<GSCpuBackend> m_shadow;
    std::vector<uint8_t> m_shadowVram;
    uint32_t m_checkEvery = 0u;
    uint32_t m_sinceCheck = 0u;
    uint64_t m_checks = 0u;
};

// The renderer the environment asks for: GSCpuBackend on its own thread (GSThreadedBackend), or on the calling thread
// with VENDETTA_GS_ASYNC=0; wrapped in GSTeeBackend with VENDETTA_GS_VERIFY.
std::unique_ptr<GSRasterBackend> makeRasterBackendFromEnvironment();
