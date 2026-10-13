#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"
#include "runtime/ps2_host_profile.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// The rounding mode every drawing thread draws in (round to nearest, exceptions masked, no flush-to-zero): set on
// entry when the caller's differs (counted as gs-fpenv-fixups), the caller's restored on exit. Out of line
// (gs_threaded_backend.cpp) so the compiler moves no floating-point code across them.
unsigned gsEnterDrawFpEnv() noexcept;
void gsLeaveDrawFpEnv(unsigned saved) noexcept;

class GSCpuBackend final : public GSRasterBackend
{
public:
    // drawingThreads: 0 = the default (VENDETTA_GS_THREADS, else half the cores, at most 8); 1 = no worker threads.
    // referenceMode: the pre-round-26 draw-queue rules, kept for the VENDETTA_GS_VERIFY shadow — drain the queue on
    // a palette change (instead of giving each queued primitive the palette it was submitted with), and the wider
    // page ranges for palette reads and uploads.
    explicit GSCpuBackend(uint32_t drawingThreads = 0u, bool referenceMode = false);
    ~GSCpuBackend() override;

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
    void VerifyPages(const std::array<uint64_t, 512> &expected, uint64_t check, uint64_t streamHash) override;

    // Draw what is queued, then a CRC of GS memory (and of each 8 KiB page when pages is given).
    uint64_t HashVram(std::array<uint64_t, 512> *pages);
    // What ClearFramebuffer returns, without drawing anything.
    static bool ClearFramebufferWritesPixels(const GSContext &context, bool hasVram);
    // The screen picture from a copy of GS memory (the presentation conversion; integer only, any thread).
    static PresentationFrame ConvertSnapshot(const uint8_t *vram, uint32_t vramSize, const GSPresentationRequest &request);

private:
    // Parallel drawing: primitives are queued and drawn in bands of rows, one band set per core (each pixel still
    // sees every primitive in order). A band owns rows y with (y >> kBandShift) % count == index.
    struct RowBand
    {
        uint32_t index = 0;
        uint32_t count = 1;
        bool owns(int y) const noexcept
        {
            return count <= 1u || ((static_cast<uint32_t>(y) >> kBandShift) % count) == index;
        }
    };
    struct PageRange
    {
        uint32_t begin = 0;
        uint32_t end = 0;
        bool overlaps(const PageRange &other) const noexcept { return begin < other.end && other.begin < end; }
    };
    static constexpr uint32_t kBandShift = 3u;              // 8-row bands
    static constexpr size_t kMaxQueuedBatches = 2048u;
    static constexpr uint64_t kParallelPixelThreshold = 2048u; // smaller queues are drawn on the calling thread

    using FlushCause = ps2_host_profile::Counter;
    void QueueBatchUnlocked(const GSPrimitiveBatch &batch);
    void FlushQueueUnlocked(FlushCause cause);
    // Drain the queue only if a write to these pages could change what a queued primitive reads or draws to.
    void FlushQueueIfTouchedUnlocked(const PageRange &pages, FlushCause cause);
    PageRange TransferDestinationPages() const;
    void FlushQueueUnlocked(FlushCause cause) const { const_cast<GSCpuBackend *>(this)->FlushQueueUnlocked(cause); }
    void DrawQueuedBand(const RowBand &band);
    void WorkerMain(uint32_t index);
    void StartWorkers();
    void StopWorkers();

    void ResetUnlocked();
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    void DrawPrimitive(const GSPrimitiveBatch &batch, const RowBand &band, const uint16_t *clut);
    void DrawSprite(const GSPrimitiveBatch &batch, const RowBand &band, const uint16_t *clut);
    void DrawTriangle(const GSPrimitiveBatch &batch, const RowBand &band, const uint16_t *clut);
    void DrawLine(const GSPrimitiveBatch &batch, const RowBand &band);
    void WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleTexture(const GSDrawState &state, const uint16_t *clut, float s, float t, float q, uint16_t u, uint16_t v);
    uint32_t LookupCLUT(const GSDrawState &state, const uint16_t *clut, uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    std::array<uint16_t, 512> m_clut{}; // the palette (CLUT buffer) now
    // Palettes of queued primitives: a palette change no longer drains the queue; each queued primitive draws with
    // the palette current when it was submitted (a snapshot taken when the palette changed since the last one).
    static constexpr uint16_t kNoClutVersion = 0xFFFFu;
    static constexpr size_t kMaxClutVersions = 1024u;
    std::vector<std::array<uint16_t, 512>> m_clutVersions;
    std::vector<uint16_t> m_queuedClut; // per queued primitive: its palette in m_clutVersions
    uint16_t m_currentClutVersion = kNoClutVersion; // m_clutVersions' copy of m_clut, if taken
    bool m_referenceMode = false;
    std::array<uint32_t, 2> m_clutCbp{};
    GSMem::TexturePageCache m_texturePageCache;

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;

    // the draw queue (guarded by m_mutex) and the drawing workers
    std::vector<GSPrimitiveBatch> m_queue;
    std::vector<PageRange> m_queuedTargets;  // frame / z pages the queued primitives write
    std::vector<PageRange> m_queuedTextures; // texture pages they read
    uint64_t m_queuedPixels = 0;
    uint32_t m_drawingThreads = 1u;
    std::vector<std::thread> m_workers;
    std::mutex m_workMutex;
    std::condition_variable m_workCv;
    std::condition_variable m_doneCv;
    uint64_t m_workGeneration = 0;
    uint32_t m_workersPending = 0;
    uint32_t m_workBandCount = 1;
    bool m_stopWorkers = false;
};
