#ifndef PS2_GS_FRONTEND_H
#define PS2_GS_FRONTEND_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_gpu.h"

struct GSDebugSnapshot
{
    GSContext ctx[2]{};
    GSPrimReg prim{};
    GSTexaReg texa{};
    GSTexClutReg texclut{};
    uint64_t scanmsk = 0;
    uint64_t dimx = 0;
    uint64_t dthe = 0;
    uint64_t colclamp = 0;
    GSBitBltBuf bitbltbuf{};
    GSTrxPos trxpos{};
    GSTrxReg trxreg{};
    uint32_t trxdir = 0;
    uint32_t transferX = 0;
    uint32_t transferY = 0;
    uint32_t transferTotalPixels = 0;
    uint32_t transferCopiedPixels = 0;
    uint32_t lastDisplayBaseBytes = 0;
    GSFrameReg preferredDisplaySourceFrame{};
    uint32_t preferredDisplayDestFbp = 0;
    bool hasPreferredDisplaySource = false;
    uint32_t hostPresentationWidth = 0;
    uint32_t hostPresentationHeight = 0;
    uint32_t hostPresentationDisplayFbp = 0;
    uint32_t hostPresentationSourceFbp = 0;
    bool hostPresentationUsedPreferred = false;
    bool hasHostPresentationFrame = false;
    size_t localToHostPendingBytes = 0;
};

enum class GSDebugEventKind : uint8_t
{
    GifTag = 0,
    Register = 1,
    Draw = 2,
    Transfer = 3,
    Present = 4,
};

struct GSDebugHistoryEntry
{
    uint64_t seq = 0;
    uint64_t vsyncTick = 0;
    uint32_t frameIndex = 0;
    GSDebugEventKind kind = GSDebugEventKind::Register;

    uint8_t reg = 0;
    uint64_t regValue = 0;

    uint32_t gifSizeBytes = 0;
    uint32_t gifNloop = 0;
    uint8_t gifFlg = 0;
    uint8_t gifNreg = 0;

    GSPrimReg prim{};
    GSFrameReg frame{};
    GSZbufReg zbuf{};
    GSTex0Reg tex0{};
    GSScissorReg scissor{};
    uint64_t test = 0;
    uint64_t alpha = 0;

    uint32_t vertexCount = 0;
    float xMin = 0.0f;
    float xMax = 0.0f;
    float yMin = 0.0f;
    float yMax = 0.0f;
    double zMin = 0.0;
    double zMax = 0.0;
    uint8_t aMin = 0;
    uint8_t aMax = 0;

    GSBitBltBuf bitbltbuf{};
    GSTrxPos trxpos{};
    GSTrxReg trxreg{};
    uint32_t trxdir = 0;
    uint32_t transferPixels = 0;

    uint32_t displayFbp = 0;
    uint32_t sourceFbp = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    bool usedPreferred = false;
};

class GS
{
public:
    GS();
    ~GS() = default;

    void init(uint8_t *vram, uint32_t vramSize, struct GSRegisters *privRegs = nullptr);
    void reset();
    // Before the game runs, or from the thread that feeds the GS.
    void setRasterBackend(std::unique_ptr<GSRasterBackend> backend);

    // path: the GIF path the packet came by (1 = VU1 XGKICK, 2 = VIF1, 3 = GIF DMA) — an unfinished tag continues
    // only with more data from the same path.
    void processGIFPacket(const uint8_t *data, uint32_t sizeBytes, uint8_t path = 3u);
    bool processNativePackedGIFPacket(const uint8_t *data, uint32_t sizeBytes);
    void uploadImageNative(uint64_t bitbltbuf,
                           uint64_t trxpos,
                           uint64_t trxreg,
                           uint64_t trxdir,
                           const uint8_t *data,
                           uint32_t sizeBytes);
    void writeRegister(uint8_t regAddr, uint64_t value);

    // Widescreen (runtime/ps2_widescreen.h): the game's rule for its 2-D layer (a HUD drawn over the 3-D scene). In a
    // wide frame those primitives are held until the picture is shown, grouped into elements (primitives that touch)
    // and each squeezed back to its 4:3 shape around the screen's left edge, right edge or centre, by the third of the
    // screen it sits in - so the HUD keeps its shape and its corners. The graphics card's renderer only (the held GIF
    // packets are patched where each vertex's X sits).
    using OverlayRule = bool (*)(const struct GSDrawState &state);
    void setWidescreenOverlayRule(OverlayRule rule) { m_overlayRule = rule; }

    const uint8_t *lockDisplaySnapshot(uint32_t &outSize);
    void unlockDisplaySnapshot();
    uint32_t getLastDisplayBaseBytes() const;
    const GSFrameReg &getContextFrame(int index) const
    {
        return m_ctx[(index != 0) ? 1 : 0].frame;
    }
    GSDebugSnapshot getDebugSnapshot() const;
    std::vector<GSDebugHistoryEntry> getDebugHistory() const;
    void clearDebugHistory();
    bool isDebugHistoryPaused() const;
    void setDebugHistoryPaused(bool paused);
    bool getPreferredDisplaySource(GSFrameReg &outSource, uint32_t &outDestFbp) const;

    // Presentation. The game thread hands the renderer "copy GS memory for the screen" at the EE scheduler's settle
    // point after each vblank's handlers (skipped when the picture cannot have changed); the renderer publishes the
    // copy in a mailbox, and the main thread converts the newest one (takePresentedFrame). The renderer may be on its
    // own thread (the default; VENDETTA_GS_ASYNC=0 = inline): the main thread never calls it.
    void presentAtSettlePoint(uint64_t vsyncTick);
    // The same with display registers captured earlier (the chain worker presents in order with the drawing it was
    // given; the game thread captured the registers at the settle point).
    void presentCaptured(const GSDisplayRegs &regs, uint64_t vsyncTick);
    static GSDisplayRegs captureDisplayRegs(const struct GSRegisters &regs);
    // Called before every entry point that changes GS state, outside the GS lock (the chain worker installs a fence
    // here: a call from another thread waits until the worker has finished what it was given).
    void setForeignCallGate(void (*gate)(void *), void *context)
    {
        m_gateContext = context;
        m_gate.store(gate, std::memory_order_release);
    }
    // The main thread: the newest presentation, converted to 640-wide rows packed to outWidth, when one newer than
    // lastSeq was published (true; outWidth / outHeight 0 = no picture: the display is not set up).
    bool takePresentedFrame(uint64_t &lastSeq,
                            std::vector<uint8_t> &outPixels,
                            uint32_t &outWidth,
                            uint32_t &outHeight,
                            uint32_t *outDisplayFbp = nullptr,
                            uint32_t *outSourceFbp = nullptr,
                            bool *outUsedPreferred = nullptr);
    // The flip throttle (the game thread, holding no lock): wait while the renderer is more than maxAhead flips behind.
    bool waitFlipBacklog(uint32_t maxAhead, const std::atomic<bool> &stop);
    void wakeRasterWaiters();  // requestStop
    void shutdownRasterThread(); // after the game thread has ended
    void shutdownGpu();          // after the GS thread (the graphics card renderer's thread owns its Vulkan objects)
    GSRendererMode rendererMode() const { return m_rendererMode; }
    const GSGpuRenderer *gpuRenderer() const { return m_gpu.get(); }
    // The main thread: whether the window shows the graphics card's picture, and the newest one (null: none newer).
    bool showsGpuPicture() const { return m_showGpu; }
    // VENDETTA_GS=both: which renderer's picture the window shows (the debug panel switches it live)
    void setShowGpuPicture(bool show) { m_showGpu = show && m_gpu != nullptr; }
    GSGpuRenderer *gpuRendererForSettings() { return m_gpu.get(); }
    // The main thread (window, debug panel): how the picture is scaled to the window.
    GSPresentFilter presentFilter() const { return m_presentFilter; }
    void setPresentFilter(GSPresentFilter filter)
    {
        m_presentFilter = filter;
        if (m_gpu)
            m_gpu->setFsr(filter == GSPresentFilter::Fsr);
    }
    const GSGpuFrame *takeGpuFrame(uint64_t &lastSeq) { return m_gpu ? m_gpu->takeFrame(lastSeq) : nullptr; }
    // Present now and convert it (tests and tools; the game uses the two above).
    void latchHostPresentationFrame();
    bool copyLatchedHostPresentationFrame(std::vector<uint8_t> &outPixels,
                                          uint32_t &outWidth,
                                          uint32_t &outHeight,
                                          uint32_t *outDisplayFbp = nullptr,
                                          uint32_t *outSourceFbp = nullptr,
                                          bool *outUsedPreferred = nullptr) const;
    bool clearFramebufferContext(uint32_t contextIndex, uint32_t rgba);
    bool clearActiveFramebuffer(uint32_t rgba);
    uint64_t nativeImageUploadCount() const { return m_nativeImageUploadCount; }
    uint64_t nativePackedGIFPacketCount() const { return m_nativePackedGIFPacketCount; }

    uint32_t consumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes);

    void refreshDisplaySnapshot();

    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);
    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;

private:
    void snapshotVRAM();
    void writeRegisterUnlocked(uint8_t regAddr, uint64_t value);
    void writeRegisterPacked(uint8_t regDesc, uint64_t lo, uint64_t hi);
    void uploadImageNativeUnlocked(uint64_t bitbltbuf,
                                   uint64_t trxpos,
                                   uint64_t trxreg,
                                   uint64_t trxdir,
                                   const uint8_t *data,
                                   uint32_t sizeBytes);
    void vertexKick(bool drawing);

    void recordDebugEventUnlocked(GSDebugHistoryEntry entry);
    GSDebugHistoryEntry makeDebugEventUnlocked(GSDebugEventKind kind) const;
    void recordGifTagDebugEventUnlocked(uint32_t sizeBytes, uint32_t nloop, uint8_t flg, uint32_t nreg);
    void recordRegisterDebugEventUnlocked(uint8_t regAddr, uint64_t value);
    void recordDrawDebugEventUnlocked(int vertexCount);
    void recordTransferDebugEventUnlocked();
    void recordPresentDebugEventUnlocked(uint32_t displayFbp, uint32_t sourceFbp, uint32_t width, uint32_t height, bool usedPreferred);

    void processImageData(const uint8_t *data, uint32_t sizeBytes);
    bool tryProcessNativeImageUploadPacket(const uint8_t *data, uint32_t sizeBytes);
    GSPrimitiveBatch buildDrawBatch(int vertexCount) const;
    void updatePreferredDisplaySourceForDraw(const GSPrimitiveBatch &batch);
    GSPresentationRequest completePresentationRequestUnlocked(const GSDisplayRegs &regs, uint64_t vsyncTick) const;
    void presentUnlocked(const GSDisplayRegs &regs, uint64_t vsyncTick, bool force);
    void gateForeignCall() const
    {
        if (void (*gate)(void *) = m_gate.load(std::memory_order_acquire))
            gate(m_gateContext);
    }
    std::atomic<void (*)(void *)> m_gate{nullptr};
    void *m_gateContext = nullptr;
    bool takePresentation(uint64_t &lastSeq, PresentationFrame &frame);

    // At the end of every entry point that hands the renderer work (declared after the lock, so it runs holding it):
    // the renderer may start on what it was given.
    struct KickGuard
    {
        GS &gs;
        explicit KickGuard(GS &owner) : gs(owner) {}
        ~KickGuard();
        KickGuard(const KickGuard &) = delete;
        KickGuard &operator=(const KickGuard &) = delete;
    };


    GSContext &activeContext();

    uint8_t *m_localMemoryStorage = nullptr;
    uint32_t m_localMemorySize = 0u;
    struct GSRegisters *m_privRegs = nullptr;
    mutable std::recursive_mutex m_stateMutex;
    mutable std::mutex m_backendLifetimeMutex;
    mutable std::mutex m_presentationMutex;

    GSContext m_ctx[2];
    GSPrimReg m_prim{};
    GSPrimReg m_primRegister{};
    GSPrimReg m_prmodeRegister{};

    uint8_t m_curR = 0x80, m_curG = 0x80, m_curB = 0x80, m_curA = 0x80;
    float m_curQ = 1.0f;
    // A GIF tag whose data did not fit in the packet that carried it: on hardware the GIF keeps the tag and goes on
    // with the next data on the path (games send an IMAGE tag and its pixels as separate DMA transfers).
    struct GifContinuation
    {
        bool active = false;
        uint8_t flg = 0;
        uint32_t loopsLeft = 0;  // PACKED / REGLIST: whole loops still to come (after the current one)
        uint32_t nreg = 0;
        uint8_t regs[16] = {};
        uint32_t regIndex = 0;   // PACKED / REGLIST: next register within the current loop
        uint32_t imageBytesLeft = 0;
        uint32_t regListWords = 0; // REGLIST: 64-bit words consumed, for the odd-count padding
    } m_gifContinuation[4];     // per GIF path (index 1..3)
    uint32_t continueGifTag(GifContinuation &tag, const uint8_t *data, uint32_t sizeBytes);
    float m_curS = 0.0f, m_curT = 0.0f;
    uint16_t m_curU = 0, m_curV = 0;
    uint8_t m_curFog = 0;
    uint8_t m_fogR = 0, m_fogG = 0, m_fogB = 0;

    bool m_prmodecont = true;
    bool m_pabe = false;
    uint64_t m_scanmsk = 0;
    uint64_t m_dimx = 0;
    uint64_t m_dthe = 0;
    uint64_t m_colclamp = 0;
    GSTexaReg m_texa{0u, false, 0u};
    GSTexClutReg m_texclut{0u, 0u, 0u};

    GSBitBltBuf m_bitbltbuf{};
    GSTrxPos m_trxpos{};
    GSTrxReg m_trxreg{};
    uint32_t m_trxdir = 3;


    static constexpr int kMaxVerts = 6;
    GSVertex m_vtxQueue[kMaxVerts];

    // Widescreen overlay (setWidescreenOverlayRule): where each queued vertex's XYZ sits (packet serial, byte offset;
    // serial 0 = not in a packet that can be held), the packets held for the graphics card, the 2-D layer's primitives.
    struct VertexSource
    {
        uint64_t packet = 0;
        uint32_t offset = 0;
    };
    struct HeldGpuData
    {
        uint64_t packet = 0; // 0 = a register write
        uint8_t path = 0;
        uint8_t regAddr = 0;
        uint64_t regValue = 0;
        std::vector<uint8_t> bytes;
    };
    struct OverlayPrimitive
    {
        float minX = 0.0f, minY = 0.0f, maxX = 0.0f, maxY = 0.0f; // screen pixels (after XYOFFSET)
        float ofx = 0.0f;                                          // XYOFFSET's X, pixels
        float width = 512.0f;                                      // the frame buffer's width
        uint8_t count = 0;
        VertexSource vertices[3];
    };
    OverlayRule m_overlayRule = nullptr;
    VertexSource m_vtxSource[kMaxVerts];
    const uint8_t *m_packetData = nullptr; // the packet being read when it may be held, else null
    const uint8_t *m_regSource = nullptr;  // the register write being read (inside m_packetData)
    uint64_t m_packetSerial = 0;
    bool m_holding = false;
    size_t m_heldBytes = 0;
    std::vector<HeldGpuData> m_held;
    std::vector<OverlayPrimitive> m_overlay;
    bool beginOverlayPacket(const uint8_t *data);
    void forwardGpuGif(uint8_t path, const uint8_t *data, uint32_t sizeBytes);
    void forwardGpuReg(uint8_t regAddr, uint64_t value);
    void recordOverlayPrimitive(const struct GSPrimitiveBatch &batch, int vertexCount);
    void flushOverlayUnlocked();
    VertexSource currentVertexSource() const
    {
        if (!m_packetData || !m_regSource)
            return {};
        return {m_packetSerial, static_cast<uint32_t>(m_regSource - m_packetData)};
    }
    int m_vtxCount = 0;
    int m_vtxIndex = 0;

    std::vector<uint8_t> m_displaySnapshot;
    std::mutex m_snapshotMutex;
    uint32_t m_lastDisplayBaseBytes = 0;
    GSFrameReg m_preferredDisplaySourceFrame{};
    uint32_t m_preferredDisplayDestFbp = 0;
    bool m_hasPreferredDisplaySource = false;
    std::vector<uint8_t> m_hostPresentationFrame;
    uint32_t m_hostPresentationWidth = 0;
    uint32_t m_hostPresentationHeight = 0;
    uint32_t m_hostPresentationDisplayFbp = 0;
    uint32_t m_hostPresentationSourceFbp = 0;
    bool m_hostPresentationUsedPreferred = false;
    bool m_hasHostPresentationFrame = false;
    uint64_t m_nativeImageUploadCount = 0;
    uint64_t m_nativePackedGIFPacketCount = 0;

    static constexpr size_t kDebugHistoryCapacity = 512;
    std::array<GSDebugHistoryEntry, kDebugHistoryCapacity> m_debugHistory{};
    size_t m_debugHistoryWrite = 0;
    size_t m_debugHistoryCount = 0;
    uint64_t m_debugNextSeq = 1;
    uint32_t m_debugFrameIndex = 0;
    uint64_t m_debugLastVsyncTick = UINT64_MAX;
    bool m_debugHistoryPaused = true;

    // presentation state (the producer: whoever holds m_stateMutex)
    bool m_vramWrittenSincePresent = true;
    bool m_hasPresented = false;
    GSPresentationRequest m_lastPresentRequest{};
    GSDisplayRegs m_lastPresentedRegs{};
    uint64_t m_presentSeq = 0;
    uint64_t m_latchedSeq = ~0ull;
    std::thread::id m_lastProducer{};

    // The graphics card renderer (VENDETTA_GS=gpu / both), or null: the CPU renderer alone.
    GSRendererMode m_rendererMode = GSRendererMode::Cpu;
    bool m_showGpu = false;
    GSPresentFilter m_presentFilter = GSPresentFilter::Bilinear;
    bool m_warnedGpuUnmirrored = false;
    uint32_t m_gpuVerifyEvery = 0;   // VENDETTA_GPU_VERIFY=1[,N] (needs VENDETTA_GS=both): compare GS memory every N
    uint64_t m_gpuVerifyPresents = 0;
    void verifyGpuMemoryUnlocked(uint64_t vsyncTick);
    std::unique_ptr<GSGpuRenderer> m_gpu;
    void warnGpuUnmirrored(const char *what);
    // Declared before m_backend: a renderer thread may write it until m_backend is destroyed.
    GSPresentMailbox m_presentation;
    std::unique_ptr<GSRasterBackend> m_backend;
    // The same renderer, for the calls made without m_stateMutex (the flip throttle, waking, stopping).
    std::atomic<GSRasterBackend *> m_backendRaw{nullptr};
};

#endif
