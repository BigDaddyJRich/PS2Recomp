#include <set>
#include "runtime/ps2_host_profile.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_threaded_backend.h"
#include "runtime/gs/gs_gpu_backend.h"
#include "ps2_log.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_widescreen.h"
#include <atomic>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>

namespace
{
    static constexpr uint32_t kHostFrameWidth = 640u;

    GSPrimReg decodePrimRegister(uint64_t value)
    {
        GSPrimReg prim{};
        prim.type = static_cast<GSPrimType>(value & 0x7u);
        prim.iip = ((value >> 3) & 1u) != 0u;
        prim.tme = ((value >> 4) & 1u) != 0u;
        prim.fge = ((value >> 5) & 1u) != 0u;
        prim.abe = ((value >> 6) & 1u) != 0u;
        prim.aa1 = ((value >> 7) & 1u) != 0u;
        prim.fst = ((value >> 8) & 1u) != 0u;
        prim.ctxt = ((value >> 9) & 1u) != 0u;
        prim.fix = ((value >> 10) & 1u) != 0u;
        return prim;
    }

    static inline uint64_t loadLE64(const uint8_t *p)
    {
        uint64_t v;
        std::memcpy(&v, p, 8);
        return v;
    }

    struct PackedGifPacketTag
    {
        uint64_t lo = 0u;
        uint64_t hi = 0u;
        uint32_t payloadOffset = 0u;
        uint32_t nloop = 0u;
        uint32_t nreg = 0u;
        uint8_t regs[16]{};
    };

    template <typename Visitor>
    bool visitPackedGifPacket(const uint8_t *data, uint32_t sizeBytes, Visitor &&visitor)
    {
        uint32_t offset = 0u;
        while (offset + 16u <= sizeBytes)
        {
            PackedGifPacketTag tag{};
            tag.lo = loadLE64(data + offset);
            tag.hi = loadLE64(data + offset + 8u);

            const uint8_t flg = static_cast<uint8_t>((tag.lo >> 58u) & 0x3u);
            if (flg != GIF_FMT_PACKED)
                return false;

            tag.nloop = static_cast<uint32_t>(tag.lo & 0x7FFFu);
            tag.nreg = static_cast<uint32_t>((tag.lo >> 60u) & 0xFu);
            if (tag.nreg == 0u)
                tag.nreg = 16u;

            const uint64_t payloadBytes64 =
                static_cast<uint64_t>(tag.nloop) * static_cast<uint64_t>(tag.nreg) * 16ull;
            if (payloadBytes64 > 0xFFFFFFFFull)
                return false;

            offset += 16u;
            const uint32_t payloadBytes = static_cast<uint32_t>(payloadBytes64);
            if (payloadBytes > sizeBytes - offset)
                return false;

            tag.payloadOffset = offset;
            for (uint32_t i = 0u; i < tag.nreg; ++i)
                tag.regs[i] = static_cast<uint8_t>((tag.hi >> (i * 4u)) & 0xFu);

            if (!visitor(tag))
                return false;

            offset += payloadBytes;
        }

        return offset == sizeBytes;
    }

    bool validatePackedGifPacket(const uint8_t *data, uint32_t sizeBytes)
    {
        return visitPackedGifPacket(data, sizeBytes, [](const PackedGifPacketTag &)
                                    { return true; });
    }

    std::atomic<uint32_t> s_debugGifPacketCount{0};
    std::atomic<uint32_t> s_debugGsRegisterCount{0};
    std::atomic<uint32_t> s_debugGsPackedVertexCount{0};
    std::atomic<uint32_t> s_debugGsVertexKickCount{0};
    std::atomic<uint32_t> s_debugCopyRegCount{0};
    std::atomic<uint32_t> s_debugTexaWriteCount{0};
    std::atomic<uint32_t> s_debugCvFontUploadCount{0};
    std::atomic<uint32_t> s_debugLocalCopyCount{0};
}


GS::GS()
    : m_rendererMode(gsRendererModeFromEnvironment())
{
    m_presentFilter = gsPresentFilterFromEnvironment();
    if (m_rendererMode != GSRendererMode::Cpu)
    {
        std::string whyNot;
        m_gpu = makeGpuRenderer(GSGpuOptions::fromEnvironment(), whyNot);
        if (!m_gpu)
        {
            std::cerr << "[gpu-gs] unavailable: " << whyNot << "; drawing on the CPU" << std::endl;
            m_rendererMode = GSRendererMode::Cpu;
        }
        else
        {
            const char *show = std::getenv("VENDETTA_GS_SHOW");
            m_showGpu = !(m_rendererMode == GSRendererMode::Both && show && std::string(show) == "cpu");
            if (const char *verify = std::getenv("VENDETTA_GPU_VERIFY"); verify && *verify && *verify != '0')
            {
                const char *comma = std::strchr(verify, ',');
                m_gpuVerifyEvery = std::max<uint32_t>(1u, comma ? static_cast<uint32_t>(std::strtoul(comma + 1, nullptr, 10)) : 60u);
                if (m_rendererMode != GSRendererMode::Both)
                {
                    std::cerr << "[gpu-verify] needs VENDETTA_GS=both (the CPU renderer is the reference)" << std::endl;
                    m_gpuVerifyEvery = 0u;
                }
            }
        }
    }
    if (m_rendererMode == GSRendererMode::Gpu)
    {
        // nothing drawn on the CPU: the CPU renderer's switches have nothing to act on
        for (const char *name : {"VENDETTA_GS_VERIFY", "VENDETTA_GS_ASYNC", "VENDETTA_SPRITELOG", "VENDETTA_GSDUMP"})
            if (const char *value = std::getenv(name); value && *value)
                std::cerr << "[gpu-gs] " << name << " needs the CPU renderer: VENDETTA_GS=cpu or both" << std::endl;
        m_backend = std::make_unique<GSGpuBackend>(*m_gpu);
    }
    else
        m_backend = makeRasterBackendFromEnvironment();
    m_backendRaw.store(m_backend.get(), std::memory_order_release);
    if (!m_gpu && m_presentFilter == GSPresentFilter::Fsr)
        m_presentFilter = GSPresentFilter::Bilinear; // FSR runs on the graphics card
    reset();
}

GS::KickGuard::~KickGuard()
{
    // A different thread feeding the renderer (expected once: the main thread sets up, the game thread runs).
    const std::thread::id self = std::this_thread::get_id();
    if (gs.m_lastProducer != self)
    {
        if (gs.m_lastProducer != std::thread::id{})
            ps2_host_profile::count(ps2_host_profile::GsProducerChanges);
        gs.m_lastProducer = self;
    }
    if (gs.m_backend)
        gs.m_backend->Kick();
    if (gs.m_gpu)
        gs.m_gpu->kick();
}

// VENDETTA_GPU_VERIFY: GS memory from both renderers at the same point in the order, compared by 8 KiB page. The
// graphics card's GS is not bit-exact with the CPU renderer (interpolation, depth precision), so this counts identical
// pages, pages within +-2 per byte and pages that differ more, and names the worst ones. Super-sampling keeps the
// single-sample memory, but compare with VENDETTA_SSAA=1 for the cleanest numbers.
void GS::verifyGpuMemoryUnlocked(uint64_t vsyncTick)
{
    const GSPresentationRequest &request = m_lastPresentRequest; // the presentation just made
    static uint32_t dumpsWritten = 0u;
    std::vector<uint8_t> cpu;
    std::vector<uint8_t> gpu;
    m_backend->Sync(GSSyncReason::DebugReadback);
    m_backend->SnapshotVram(cpu);
    if (m_holding)
        flushOverlayUnlocked();
    m_gpu->snapshotVram(gpu);
    const size_t bytes = std::min(cpu.size(), gpu.size());
    if (bytes == 0u)
        return;
    constexpr size_t kPage = 8192u;
    struct Worst
    {
        uint32_t page;
        uint32_t differing;
        uint32_t maxDelta;
    };
    std::vector<Worst> worst;
    uint32_t same = 0u, near = 0u, differ = 0u;
    uint64_t totalDiffering = 0u;
    uint64_t byLane[4] = {}; // bytes off by >2, by position in a 32-bit word (0-2 = R G B of a 32-bit pixel, 3 = A)
    for (size_t page = 0; page * kPage < bytes; ++page)
    {
        const uint8_t *a = cpu.data() + page * kPage;
        const uint8_t *b = gpu.data() + page * kPage;
        if (std::memcmp(a, b, kPage) == 0)
        {
            ++same;
            continue;
        }
        uint32_t differing = 0u, maxDelta = 0u;
        for (size_t i = 0; i < kPage; ++i)
        {
            const uint32_t delta = static_cast<uint32_t>(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]);
            if (delta > 2u)
            {
                ++differing;
                ++byLane[i & 3u];
            }
            maxDelta = std::max(maxDelta, delta);
        }
        if (differing == 0u)
            ++near;
        else
        {
            ++differ;
            totalDiffering += differing;
            worst.push_back({static_cast<uint32_t>(page), differing, maxDelta});
        }
    }
    std::sort(worst.begin(), worst.end(), [](const Worst &x, const Worst &y)
              { return x.differing > y.differing; });
    std::cerr << "[gpu-verify] vsync=" << vsyncTick << " pages: same=" << same << " near=" << near << " differ=" << differ
              << " (" << totalDiffering << " bytes off by >2)";
    for (size_t i = 0; i < std::min<size_t>(worst.size(), 5u); ++i)
        std::cerr << " page " << worst[i].page << " (block 0x" << std::hex << worst[i].page * 32u << std::dec << ": "
                  << worst[i].differing << " bytes, max " << worst[i].maxDelta << ")";
    std::cerr << " by byte: " << byLane[0] << "/" << byLane[1] << "/" << byLane[2] << "/" << byLane[3];

    // The picture each renderer's memory makes (read the CPU renderer's way from both): what differs on screen.
    const PresentationFrame a = GSCpuBackend::ConvertSnapshot(cpu.data(), static_cast<uint32_t>(cpu.size()), request);
    const PresentationFrame b = GSCpuBackend::ConvertSnapshot(gpu.data(), static_cast<uint32_t>(gpu.size()), request);
    if (a.width == b.width && a.height == b.height && !a.pixels.empty() && a.pixels.size() == b.pixels.size())
    {
        const size_t pixels = a.pixels.size() / 4u;
        uint64_t sum = 0u, off8 = 0u;
        for (size_t i = 0; i < pixels; ++i)
        {
            uint32_t worstChannel = 0u;
            for (size_t ch = 0; ch < 3u; ++ch)
            {
                const int d = static_cast<int>(a.pixels[i * 4u + ch]) - static_cast<int>(b.pixels[i * 4u + ch]);
                const uint32_t m = static_cast<uint32_t>(d < 0 ? -d : d);
                sum += m;
                worstChannel = std::max(worstChannel, m);
            }
            if (worstChannel > 8u)
                ++off8;
        }
        std::cerr << " picture " << a.width << "x" << a.height << ": mean |d| " << (static_cast<double>(sum) / (pixels * 3.0))
                  << ", " << (100.0 * static_cast<double>(off8) / static_cast<double>(pixels)) << "% of pixels off by >8";
        // VENDETTA_GPU_VERIFY_DUMP=<dir>: the first 3 picture pairs as PPM
        static const char *dumpDir = std::getenv("VENDETTA_GPU_VERIFY_DUMP");
        if (dumpDir && *dumpDir && dumpsWritten < 3u && off8 != 0u)
        {
            for (int which = 0; which < 2; ++which)
            {
                const PresentationFrame &f = which == 0 ? a : b;
                const std::string path = std::string(dumpDir) + "/" + (which == 0 ? "cpu-" : "gpu-") +
                                         std::to_string(vsyncTick) + ".ppm";
                if (FILE *file = std::fopen(path.c_str(), "wb"))
                {
                    std::fprintf(file, "P6%c%u %u%c255%c", 10, f.width, f.height, 10, 10);
                    for (size_t i = 0; i < static_cast<size_t>(f.width) * f.height; ++i)
                        std::fwrite(&f.pixels[i * 4u], 1, 3, file);
                    std::fclose(file);
                }
            }
            ++dumpsWritten;
        }
    }
    std::cerr << std::endl;
}

// Input the graphics card renderer does not get yet (none of it reaches the GS in Def Jam Vendetta).
void GS::warnGpuUnmirrored(const char *what)
{
    if (m_warnedGpuUnmirrored)
        return;
    m_warnedGpuUnmirrored = true;
    std::cerr << "[gpu-gs] " << what << " is not passed to the graphics card; its picture may differ" << std::endl;
}

void GS::init(uint8_t *vram, uint32_t vramSize, GSRegisters *privRegs)
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    m_localMemoryStorage = vram;
    m_localMemorySize = vramSize;
    m_privRegs = privRegs;
    if (!m_backend)
    {
        m_backend = std::make_unique<GSCpuBackend>();
        m_backendRaw.store(m_backend.get(), std::memory_order_release);
    }
    m_backend->Initialize(vram, vramSize);
    m_vramWrittenSincePresent = true;
    reset();
}

void GS::reset()
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    m_held.clear();
    m_overlay.clear();
    m_holding = false;
    m_heldBytes = 0;
    if (m_gpu) // none with the CPU renderer (VENDETTA_GS=cpu)
        m_gpu->reset();
    std::memset(m_ctx, 0, sizeof(m_ctx));
    m_prim = {};
    m_primRegister = {};
    m_prmodeRegister = {};
    m_curR = 0x80;
    m_curG = 0x80;
    m_curB = 0x80;
    m_curA = 0x80;
    m_curQ = 1.0f;
    m_curS = 0.0f;
    m_curT = 0.0f;
    m_curU = 0;
    m_curV = 0;
    m_curFog = 0;
    m_fogR = 0;
    m_fogG = 0;
    m_fogB = 0;
    m_prmodecont = true;
    m_pabe = false;
    m_scanmsk = 0u;
    m_dimx = 0u;
    m_dthe = 0u;
    m_colclamp = 0u;
    m_texa = {0u, false, 0u};
    m_texclut = {0u, 0u, 0u};
    m_bitbltbuf = {};
    m_trxpos = {};
    m_trxreg = {};
    m_trxdir = 3;
    m_vtxCount = 0;
    m_vtxIndex = 0;
    m_preferredDisplaySourceFrame = {};
    m_preferredDisplayDestFbp = 0;
    m_hasPreferredDisplaySource = false;
    if (m_backend)
    {
        m_backend->Flush();
        m_backend->Sync(GSSyncReason::Reset);
        m_backend->Reset();
        m_backend->ResetPresentation(m_presentation, ++m_presentSeq);
    }
    m_vramWrittenSincePresent = true;
    m_hasPresented = false;
    m_lastPresentRequest = {};
    m_lastPresentedRegs = {};
    {
        std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
        m_hostPresentationFrame.clear();
        m_hostPresentationWidth = 0u;
        m_hostPresentationHeight = 0u;
        m_hostPresentationDisplayFbp = 0u;
        m_hostPresentationSourceFbp = 0u;
        m_hostPresentationUsedPreferred = false;
        m_hasHostPresentationFrame = false;
    }

    m_debugHistoryWrite = 0;
    m_debugHistoryCount = 0;
    m_debugNextSeq = 1;
    m_debugFrameIndex = 0;
    m_debugLastVsyncTick = UINT64_MAX;

    for (int i = 0; i < 2; ++i)
    {
        m_ctx[i].frame.fbw = 10;
        m_ctx[i].scissor = {0, 639, 0, 447};
        m_ctx[i].xyoffset = {0, 0};
    }
}

GSContext &GS::activeContext()
{
    return m_ctx[m_prim.ctxt ? 1 : 0];
}

void GS::snapshotVRAM()
{
    // Presentation/debug snapshots run outside m_stateMutex so the EE can keep
    // feeding the GS while a backend performs host-side conversion. Keep the
    // selected backend alive and unswappable for the duration of the call.
    gateForeignCall();
    std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
    if (!m_backend)
        return;
    std::vector<uint8_t> snapshot;
    ps2_host_profile::count(ps2_host_profile::GsReplyCalls);
    m_backend->Sync(GSSyncReason::DebugReadback);
    m_backend->SnapshotVram(snapshot);
    std::lock_guard<std::mutex> lock(m_snapshotMutex);
    m_displaySnapshot.swap(snapshot);
}

const uint8_t *GS::lockDisplaySnapshot(uint32_t &outSize)
{
    m_snapshotMutex.lock();
    if (m_displaySnapshot.empty())
    {
        outSize = 0;
        return nullptr;
    }

    outSize = static_cast<uint32_t>(m_displaySnapshot.size());
    return m_displaySnapshot.data();
}

GSDebugSnapshot GS::getDebugSnapshot() const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);

    GSDebugSnapshot snapshot{};
    snapshot.ctx[0] = m_ctx[0];
    snapshot.ctx[1] = m_ctx[1];
    snapshot.prim = m_prim;
    snapshot.texa = m_texa;
    snapshot.texclut = m_texclut;
    snapshot.scanmsk = m_scanmsk;
    snapshot.dimx = m_dimx;
    snapshot.dthe = m_dthe;
    snapshot.colclamp = m_colclamp;
    snapshot.bitbltbuf = m_bitbltbuf;
    snapshot.trxpos = m_trxpos;
    snapshot.trxreg = m_trxreg;
    const GSTransferSnapshot transfer = m_backend ? m_backend->GetTransferSnapshot() : GSTransferSnapshot{};
    snapshot.trxdir = transfer.direction;
    snapshot.transferX = transfer.x;
    snapshot.transferY = transfer.y;
    snapshot.transferTotalPixels = transfer.totalPixels;
    snapshot.transferCopiedPixels = transfer.copiedPixels;
    snapshot.lastDisplayBaseBytes = m_lastDisplayBaseBytes;
    snapshot.preferredDisplaySourceFrame = m_preferredDisplaySourceFrame;
    snapshot.preferredDisplayDestFbp = m_preferredDisplayDestFbp;
    snapshot.hasPreferredDisplaySource = m_hasPreferredDisplaySource;
    {
        std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
        snapshot.hostPresentationWidth = m_hostPresentationWidth;
        snapshot.hostPresentationHeight = m_hostPresentationHeight;
        snapshot.hostPresentationDisplayFbp = m_hostPresentationDisplayFbp;
        snapshot.hostPresentationSourceFbp = m_hostPresentationSourceFbp;
        snapshot.hostPresentationUsedPreferred = m_hostPresentationUsedPreferred;
        snapshot.hasHostPresentationFrame = m_hasHostPresentationFrame;
    }
    snapshot.localToHostPendingBytes = transfer.localToHostPendingBytes;
    return snapshot;
}

std::vector<GSDebugHistoryEntry> GS::getDebugHistory() const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);

    std::vector<GSDebugHistoryEntry> out;
    out.reserve(m_debugHistoryCount);
    const size_t first = (m_debugHistoryWrite + kDebugHistoryCapacity - m_debugHistoryCount) % kDebugHistoryCapacity;
    for (size_t i = 0; i < m_debugHistoryCount; ++i)
    {
        out.push_back(m_debugHistory[(first + i) % kDebugHistoryCapacity]);
    }
    return out;
}

void GS::clearDebugHistory()
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    m_debugHistoryWrite = 0;
    m_debugHistoryCount = 0;
    m_debugNextSeq = 1;
    m_debugFrameIndex = 0;
    m_debugLastVsyncTick = UINT64_MAX;
}

bool GS::isDebugHistoryPaused() const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_debugHistoryPaused;
}

void GS::setDebugHistoryPaused(bool paused)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    m_debugHistoryPaused = paused;
}

GSDebugHistoryEntry GS::makeDebugEventUnlocked(GSDebugEventKind kind) const
{
    GSDebugHistoryEntry entry{};
    entry.kind = kind;
    entry.prim = m_prim;
    const uint32_t ci = m_prim.ctxt ? 1u : 0u;
    entry.frame = m_ctx[ci].frame;
    entry.zbuf = m_ctx[ci].zbuf;
    entry.tex0 = m_ctx[ci].tex0;
    entry.scissor = m_ctx[ci].scissor;
    entry.test = m_ctx[ci].test;
    entry.alpha = m_ctx[ci].alpha;
    entry.bitbltbuf = m_bitbltbuf;
    entry.trxpos = m_trxpos;
    entry.trxreg = m_trxreg;
    const GSTransferSnapshot transfer = m_backend ? m_backend->GetTransferSnapshot() : GSTransferSnapshot{};
    entry.trxdir = transfer.direction;
    entry.transferPixels = transfer.totalPixels;
    return entry;
}

void GS::recordDebugEventUnlocked(GSDebugHistoryEntry entry)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    const uint64_t tick = m_privRegs ? m_privRegs->vsyncTick.load(std::memory_order_acquire) : 0u;
    if (m_debugLastVsyncTick == UINT64_MAX)
    {
        m_debugLastVsyncTick = tick;
    }
    else if (tick != m_debugLastVsyncTick)
    {
        ++m_debugFrameIndex;
        m_debugLastVsyncTick = tick;
    }

    entry.seq = m_debugNextSeq++;
    entry.vsyncTick = tick;
    entry.frameIndex = m_debugFrameIndex;

    m_debugHistory[m_debugHistoryWrite] = entry;
    m_debugHistoryWrite = (m_debugHistoryWrite + 1u) % kDebugHistoryCapacity;
    if (m_debugHistoryCount < kDebugHistoryCapacity)
    {
        ++m_debugHistoryCount;
    }
}

void GS::recordGifTagDebugEventUnlocked(uint32_t sizeBytes, uint32_t nloop, uint8_t flg, uint32_t nreg)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::GifTag);
    entry.gifSizeBytes = sizeBytes;
    entry.gifNloop = nloop;
    entry.gifFlg = flg;
    entry.gifNreg = static_cast<uint8_t>(std::min<uint32_t>(nreg, 16u));
    recordDebugEventUnlocked(entry);
}

void GS::recordRegisterDebugEventUnlocked(uint8_t regAddr, uint64_t value)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    switch (regAddr)
    {
    case GS_REG_PRIM:
    case GS_REG_TEX0_1:
    case GS_REG_TEX0_2:
    case GS_REG_TEX2_1:
    case GS_REG_TEX2_2:
    case GS_REG_TEXA:
    case GS_REG_TEXCLUT:
    case GS_REG_FRAME_1:
    case GS_REG_FRAME_2:
    case GS_REG_ZBUF_1:
    case GS_REG_ZBUF_2:
    case GS_REG_ALPHA_1:
    case GS_REG_ALPHA_2:
    case GS_REG_TEST_1:
    case GS_REG_TEST_2:
    case GS_REG_SCISSOR_1:
    case GS_REG_SCISSOR_2:
    case GS_REG_XYOFFSET_1:
    case GS_REG_XYOFFSET_2:
    case GS_REG_BITBLTBUF:
    case GS_REG_TRXPOS:
    case GS_REG_TRXREG:
    case GS_REG_TRXDIR:
        break;
    default:
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Register);
    entry.reg = regAddr;
    entry.regValue = value;
    recordDebugEventUnlocked(entry);
}

void GS::recordDrawDebugEventUnlocked(int vertexCount)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    if (vertexCount <= 0)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Draw);
    entry.vertexCount = static_cast<uint32_t>(vertexCount);

    const int count = std::min(vertexCount, kMaxVerts);
    entry.xMin = entry.xMax = m_vtxQueue[0].x;
    entry.yMin = entry.yMax = m_vtxQueue[0].y;
    entry.zMin = entry.zMax = m_vtxQueue[0].z;
    entry.aMin = entry.aMax = m_vtxQueue[0].a;

    for (int i = 1; i < count; ++i)
    {
        const GSVertex &v = m_vtxQueue[i];
        entry.xMin = std::min(entry.xMin, v.x);
        entry.xMax = std::max(entry.xMax, v.x);
        entry.yMin = std::min(entry.yMin, v.y);
        entry.yMax = std::max(entry.yMax, v.y);
        entry.zMin = std::min(entry.zMin, v.z);
        entry.zMax = std::max(entry.zMax, v.z);
        entry.aMin = std::min(entry.aMin, v.a);
        entry.aMax = std::max(entry.aMax, v.a);
    }

    recordDebugEventUnlocked(entry);
}

void GS::recordTransferDebugEventUnlocked()
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Transfer);
    entry.transferPixels = m_backend ? m_backend->GetTransferSnapshot().totalPixels : 0u;
    recordDebugEventUnlocked(entry);
}

void GS::recordPresentDebugEventUnlocked(uint32_t displayFbp, uint32_t sourceFbp, uint32_t width, uint32_t height, bool usedPreferred)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Present);
    entry.displayFbp = displayFbp;
    entry.sourceFbp = sourceFbp;
    entry.width = width;
    entry.height = height;
    entry.usedPreferred = usedPreferred;
    recordDebugEventUnlocked(entry);
}

bool GS::getPreferredDisplaySource(GSFrameReg &outSource, uint32_t &outDestFbp) const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!m_hasPreferredDisplaySource)
    {
        outSource = {};
        outDestFbp = 0u;
        return false;
    }

    outSource = m_preferredDisplaySourceFrame;
    outDestFbp = m_preferredDisplayDestFbp;
    return true;
}

void GS::unlockDisplaySnapshot()
{
    m_snapshotMutex.unlock();
}

uint32_t GS::getLastDisplayBaseBytes() const
{
    return m_lastDisplayBaseBytes;
}

void GS::refreshDisplaySnapshot()
{
    snapshotVRAM();
}

GSDisplayRegs GS::captureDisplayRegs(const GSRegisters &regs)
{
    GSDisplayRegs display{};
    display.pmode = regs.pmode;
    display.smode1 = regs.smode1;
    display.smode2 = regs.smode2;
    display.srfsh = regs.srfsh;
    display.synch1 = regs.synch1;
    display.synch2 = regs.synch2;
    display.syncv = regs.syncv;
    display.dispfb1 = regs.dispfb1;
    display.display1 = regs.display1;
    display.dispfb2 = regs.dispfb2;
    display.display2 = regs.display2;
    display.extbuf = regs.extbuf;
    display.extdata = regs.extdata;
    display.extwrite = regs.extwrite;
    display.bgcolor = regs.bgcolor;
    return display;
}

namespace
{
    bool sameFrame(const GSFrameReg &a, const GSFrameReg &b)
    {
        return a.fbp == b.fbp && a.fbw == b.fbw && a.psm == b.psm && a.fbmsk == b.fbmsk;
    }

    // Everything the picture is built from except the vblank number (the display registers are compared apart).
    bool samePresentation(const GSPresentationRequest &a, const GSPresentationRequest &b)
    {
        return sameFrame(a.contextFrames[0], b.contextFrames[0]) && sameFrame(a.contextFrames[1], b.contextFrames[1]) &&
               sameFrame(a.preferredSource, b.preferredSource) && a.preferredDestFbp == b.preferredDestFbp &&
               a.hasPreferredSource == b.hasPreferredSource;
    }
}

GSPresentationRequest GS::completePresentationRequestUnlocked(const GSDisplayRegs &regs, uint64_t vsyncTick) const
{
    GSPresentationRequest request{};
    request.pmode = regs.pmode;
    request.smode1 = regs.smode1;
    request.smode2 = regs.smode2;
    request.dispfb1 = regs.dispfb1;
    request.display1 = regs.display1;
    request.dispfb2 = regs.dispfb2;
    request.display2 = regs.display2;
    request.bgcolor = regs.bgcolor;
    request.vsyncTick = vsyncTick;
    request.contextFrames[0] = m_ctx[0].frame;
    request.contextFrames[1] = m_ctx[1].frame;
    request.preferredSource = m_preferredDisplaySourceFrame;
    request.preferredDestFbp = m_preferredDisplayDestFbp;
    request.hasPreferredSource = m_hasPreferredDisplaySource;
    return request;
}

namespace
{
    // VENDETTA_HUDLOG=<seconds>[,<frames>][,all]: from the first presentation after that many seconds, every primitive
    // of the next <frames> pictures (default 1) drawn with the depth test off (TEST.ZTE = 0) or as a sprite - the 2-D
    // layers such as a HUD - or every primitive with "all": type, screen box (after XYOFFSET), frame buffer, texture,
    // TEST. Kept in memory and printed when the frames are done (printing each line slowed the game below a frame).
    struct HudLog
    {
        enum Phase
        {
            Off,
            Waiting,
            Capturing,
            Done,
        };
        Phase phase = Off;
        double from = 0.0;
        int frames = 1;
        bool all = false;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        std::string text;
    };
    HudLog s_hudLog = []
    {
        HudLog log;
        if (const char *value = std::getenv("VENDETTA_HUDLOG"))
        {
            log.phase = HudLog::Waiting;
            std::sscanf(value, "%lf,%d", &log.from, &log.frames);
            log.frames = std::max(1, log.frames);
            log.all = std::strstr(value, "all") != nullptr;
        }
        return log;
    }();
    bool s_hudLogArmed = false; // the time has come: capture from the next presentation
    uint32_t s_hudPacket = 0, s_hudTag = 0, s_hudPath = 0; // which packet / tag / path a logged primitive came in

    void hudLogPrimitive(const GSPrimitiveBatch &batch)
    {
        if (s_hudLog.phase == HudLog::Waiting)
        {
            s_hudLogArmed = std::chrono::duration<double>(std::chrono::steady_clock::now() - s_hudLog.start).count() >= s_hudLog.from;
            return;
        }
        if (s_hudLog.phase != HudLog::Capturing)
            return;
        const GSContext &ctx = batch.state.context;
        if (!s_hudLog.all && (ctx.test & 0x10000u) != 0u && batch.state.prim.type != GS_PRIM_SPRITE)
            return;
        const float ox = static_cast<float>(ctx.xyoffset.ofx) / 16.0f, oy = static_cast<float>(ctx.xyoffset.ofy) / 16.0f;
        float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
        for (uint32_t i = 0; i < batch.vertexCount; ++i)
        {
            minX = std::min(minX, batch.vertices[i].x - ox);
            maxX = std::max(maxX, batch.vertices[i].x - ox);
            minY = std::min(minY, batch.vertices[i].y - oy);
            maxY = std::max(maxY, batch.vertices[i].y - oy);
        }
        char line[256];
        std::snprintf(line, sizeof(line), "[hud] pk=%u tag=%u path=%u ctxt=%u type=%u box=(%.1f,%.1f)-(%.1f,%.1f) fbp=0x%x tme=%u tex=0x%x/psm0x%x abe=%u test=0x%llx rgba=%u,%u,%u,%u\n",
                      s_hudPacket, s_hudTag, s_hudPath, static_cast<unsigned>(batch.state.prim.ctxt), static_cast<unsigned>(batch.state.prim.type), minX, minY, maxX, maxY,
                      ctx.frame.fbp, batch.state.prim.tme, ctx.tex0.tbp0, ctx.tex0.psm, batch.state.prim.abe,
                      static_cast<unsigned long long>(ctx.test), batch.vertices[0].r, batch.vertices[0].g, batch.vertices[0].b,
                      batch.vertices[0].a);
        s_hudLog.text += line;
    }

    void hudLogPresentation()
    {
        if (s_hudLog.phase == HudLog::Waiting && s_hudLogArmed)
        {
            s_hudLog.phase = HudLog::Capturing;
            s_hudLog.text += "[hud] P (capture starts)\n";
        }
        else if (s_hudLog.phase == HudLog::Capturing)
        {
            s_hudLog.text += "[hud] P\n";
            if (--s_hudLog.frames == 0)
            {
                s_hudLog.phase = HudLog::Done;
                std::fputs(s_hudLog.text.c_str(), stderr);
                std::fflush(stderr);
                s_hudLog.text.clear();
            }
        }
    }
}

void GS::presentUnlocked(const GSDisplayRegs &regs, uint64_t vsyncTick, bool force)
{
    if (m_holding)
        flushOverlayUnlocked(); // the frame's 2-D layer, placed for widescreen, before the picture
    // HCTP_VRAM_DUMP_SEC=N: the 4 MB GS memory as the renderer sees it, once, N s after start (vram_dump.bin in the
    // working directory) - to diff the native renderers' memory against paraLLEl-GS
    {
        static const char *s_dumpEnv = std::getenv("HCTP_VRAM_DUMP_SEC");
        static const double s_dumpSec = s_dumpEnv ? std::atof(s_dumpEnv) : -1.0;
        static const auto s_t0 = std::chrono::steady_clock::now();
        static bool s_dumped = false;
        // HCTP_VRAM_DUMP_GO=1: also dump whenever vram_dump.go appears in the working directory (deleted after; to catch
        // a screen that is hard to time, e.g. an entrance nameplate)
        static const bool s_goEnv = std::getenv("HCTP_VRAM_DUMP_GO") != nullptr;
        static uint32_t s_goCheck = 0;
        bool goNow = false;
        if (s_goEnv && (m_backend || m_gpu) && (++s_goCheck % 30u) == 0u)
            if (FILE *g = std::fopen("vram_dump.go", "rb"))
            {
                std::fclose(g);
                std::remove("vram_dump.go");
                goNow = true;
            }
        if (goNow || (s_dumpSec >= 0.0 && !s_dumped && (m_backend || m_gpu) &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - s_t0).count() >= s_dumpSec))
        {
            s_dumped = true;
            std::vector<uint8_t> vram;
            if (m_gpu)
                m_gpu->snapshotVram(vram);
            else
                m_backend->SnapshotVram(vram);
            if (FILE *f = std::fopen("vram_dump.bin", "wb"))
            {
                std::fwrite(vram.data(), 1, vram.size(), f);
                std::fclose(f);
            }
            std::fprintf(stderr, "[vramdump] %zu bytes%c", vram.size(), 10);
        }
    }
    const GSPresentationRequest request = completePresentationRequestUnlocked(regs, vsyncTick);
    const bool flip = !m_hasPresented || regs != m_lastPresentedRegs;
    // The same display registers, the same request and no write to GS memory since: the same picture.
    if (!force && !flip && !m_vramWrittenSincePresent && samePresentation(request, m_lastPresentRequest))
    {
        ps2_host_profile::count(ps2_host_profile::GsPresentsSkipped);
        return;
    }
    if (s_hudLog.phase != HudLog::Off)
        hudLogPresentation();
    m_vramWrittenSincePresent = false;
    m_hasPresented = true;
    m_lastPresentRequest = request;
    m_lastPresentedRegs = regs;
    m_backend->PresentToMailbox(request, GSPresentTicket{++m_presentSeq, flip}, m_presentation);
    if (m_gpu)
    {
        const uint64_t privRegs[GSGpuRenderer::kPrivRegCount] = {
            regs.pmode, regs.smode1, regs.smode2, regs.srfsh, regs.synch1, regs.synch2, regs.syncv, regs.dispfb1,
            regs.display1, regs.dispfb2, regs.display2, regs.extbuf, regs.extdata, regs.extwrite, regs.bgcolor};
        m_gpu->present(privRegs, vsyncTick, flip);
        if (m_gpuVerifyEvery != 0u && ++m_gpuVerifyPresents % m_gpuVerifyEvery == 0u)
            verifyGpuMemoryUnlocked(vsyncTick);
    }
    ps2_host_profile::count(ps2_host_profile::GsPresents);
    if (flip)
        ps2_host_profile::count(ps2_host_profile::GsFlips);
    const uint32_t displayFbp = static_cast<uint32_t>(((regs.pmode & 1u) != 0u ? regs.dispfb1 : regs.dispfb2) & 0x1FFu);
    recordPresentDebugEventUnlocked(displayFbp, request.hasPreferredSource ? request.preferredSource.fbp : displayFbp, 0u, 0u,
                                    request.hasPreferredSource);
}

void GS::presentAtSettlePoint(uint64_t vsyncTick)
{
    if (!m_privRegs)
        return;
    presentCaptured(captureDisplayRegs(*m_privRegs), vsyncTick);
}

void GS::presentCaptured(const GSDisplayRegs &regs, uint64_t vsyncTick)
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    if (!m_backend || !m_privRegs)
        return;
    presentUnlocked(regs, vsyncTick, false);
}

bool GS::takePresentation(uint64_t &lastSeq, PresentationFrame &frame)
{
    if (m_presentation.sequence() == lastSeq)
        return false;
    const GSPresentSlot *slot = m_presentation.acquire();
    lastSeq = slot->seq;
    frame = slot->valid ? GSCpuBackend::ConvertSnapshot(slot->vram.data(), static_cast<uint32_t>(slot->vram.size()), slot->request)
                        : PresentationFrame{};
    std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
    m_hostPresentationWidth = frame.width;
    m_hostPresentationHeight = frame.height;
    m_hostPresentationDisplayFbp = frame.displayFbp;
    m_hostPresentationSourceFbp = frame.sourceFbp;
    m_hostPresentationUsedPreferred = frame.usedPreferred;
    m_hasHostPresentationFrame = static_cast<bool>(frame);
    return true;
}

bool GS::takePresentedFrame(uint64_t &lastSeq,
                            std::vector<uint8_t> &outPixels,
                            uint32_t &outWidth,
                            uint32_t &outHeight,
                            uint32_t *outDisplayFbp,
                            uint32_t *outSourceFbp,
                            bool *outUsedPreferred)
{
    PresentationFrame frame{};
    if (!takePresentation(lastSeq, frame))
        return false;
    outPixels.clear();
    outWidth = 0u;
    outHeight = 0u;
    if (outDisplayFbp)
        *outDisplayFbp = frame.displayFbp;
    if (outSourceFbp)
        *outSourceFbp = frame.sourceFbp;
    if (outUsedPreferred)
        *outUsedPreferred = frame.usedPreferred;
    if (!frame)
        return true;
    // 640-wide rows packed to the picture's width
    const size_t packedRowBytes = static_cast<size_t>(frame.width) * 4u;
    const size_t sourceRowBytes = static_cast<size_t>(kHostFrameWidth) * 4u;
    if (static_cast<size_t>(frame.height - 1u) * sourceRowBytes + packedRowBytes > frame.pixels.size())
        return true;
    outPixels.resize(packedRowBytes * frame.height);
    for (uint32_t y = 0; y < frame.height; ++y)
        std::memcpy(outPixels.data() + y * packedRowBytes, frame.pixels.data() + y * sourceRowBytes, packedRowBytes);
    outWidth = frame.width;
    outHeight = frame.height;
    return true;
}

bool GS::waitFlipBacklog(uint32_t maxAhead, const std::atomic<bool> &stop)
{
    if (m_gpu && !m_gpu->waitPresentBacklog(maxAhead, stop))
        return false;
    GSRasterBackend *backend = m_backendRaw.load(std::memory_order_acquire);
    return !backend || backend->WaitFlipBacklog(maxAhead, stop);
}

void GS::wakeRasterWaiters()
{
    if (m_gpu)
        m_gpu->wake();
    if (GSRasterBackend *backend = m_backendRaw.load(std::memory_order_acquire))
        backend->WakeWaiters();
}

void GS::shutdownRasterThread()
{
    if (GSRasterBackend *backend = m_backendRaw.load(std::memory_order_acquire))
        backend->Shutdown();
}

void GS::shutdownGpu()
{
    if (m_gpu)
        m_gpu->shutdown();
}

void GS::latchHostPresentationFrame()
{
    gateForeignCall();
    {
        std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
        KickGuard kick(*this);
        if (!m_backend || !m_privRegs)
        {
            std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
            m_hostPresentationFrame.clear();
            m_hasHostPresentationFrame = false;
            m_hostPresentationWidth = m_hostPresentationHeight = 0u;
            return;
        }
        presentUnlocked(captureDisplayRegs(*m_privRegs), m_privRegs->vsyncTick.load(std::memory_order_acquire), true);
        m_backend->Sync(GSSyncReason::DebugReadback); // a renderer thread has published it once this returns
    }
    PresentationFrame frame{};
    takePresentation(m_latchedSeq, frame);
    std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
    m_hostPresentationFrame = std::move(frame.pixels);
}

bool GS::copyLatchedHostPresentationFrame(std::vector<uint8_t> &outPixels,
                                          uint32_t &outWidth,
                                          uint32_t &outHeight,
                                          uint32_t *outDisplayFbp,
                                          uint32_t *outSourceFbp,
                                          bool *outUsedPreferred) const
{
    std::lock_guard<std::mutex> lock(m_presentationMutex);
    if (!m_hasHostPresentationFrame || m_hostPresentationFrame.empty())
    {
        outPixels.clear();
        outWidth = 0u;
        outHeight = 0u;
        if (outDisplayFbp)
            *outDisplayFbp = 0u;
        if (outSourceFbp)
            *outSourceFbp = 0u;
        if (outUsedPreferred)
            *outUsedPreferred = false;
        return false;
    }

    outWidth = m_hostPresentationWidth;
    outHeight = m_hostPresentationHeight;
    if (outDisplayFbp)
        *outDisplayFbp = m_hostPresentationDisplayFbp;
    if (outSourceFbp)
        *outSourceFbp = m_hostPresentationSourceFbp;
    if (outUsedPreferred)
        *outUsedPreferred = m_hostPresentationUsedPreferred;

    const size_t packedRowBytes = static_cast<size_t>(outWidth) * 4u;
    outPixels.resize(packedRowBytes * static_cast<size_t>(outHeight));
    if (outWidth != 0u && outHeight != 0u)
    {
        const size_t sourceRowBytes = static_cast<size_t>(kHostFrameWidth) * 4u;
        for (uint32_t y = 0; y < outHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * sourceRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * packedRowBytes;
            if (srcOffset + packedRowBytes > m_hostPresentationFrame.size() ||
                dstOffset + packedRowBytes > outPixels.size())
            {
                outPixels.clear();
                outWidth = 0u;
                outHeight = 0u;
                if (outDisplayFbp)
                    *outDisplayFbp = 0u;
                if (outSourceFbp)
                    *outSourceFbp = 0u;
                if (outUsedPreferred)
                    *outUsedPreferred = false;
                return false;
            }

            std::memcpy(outPixels.data() + dstOffset,
                        m_hostPresentationFrame.data() + srcOffset,
                        packedRowBytes);
        }
    }
    return true;
}

// HCTP native-VU1 verify: while set (per thread), PATH1 packets are also appended here (hctp_native_vu1.cpp)
thread_local std::vector<uint8_t> *g_hctpPath1Capture = nullptr;

void GS::processGIFPacket(const uint8_t *data, uint32_t sizeBytes, uint8_t path)
{
    if (path == 1u && g_hctpPath1Capture && data && sizeBytes)
        g_hctpPath1Capture->insert(g_hctpPath1Capture->end(), data, data + sizeBytes);
    gateForeignCall();
    ps2_host_profile::Scope profile(ps2_host_profile::GsLock);
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    ps2_host_profile::setPhase(ps2_host_profile::Gs);
    if (!data || sizeBytes < 16 || !m_backend)
        return;
    // the raw bytes: the graphics card's GS keeps its own tag state per path. In a widescreen frame with an overlay
    // rule they go after this front end has read them (a held 2-D layer is patched first).
    const bool late = m_gpu && beginOverlayPacket(data);
    if (m_gpu && !late)
        m_gpu->gif(path, data, sizeBytes);
    struct LateForward
    {
        GS &gs;
        bool on;
        uint8_t path;
        const uint8_t *data;
        uint32_t size;
        ~LateForward()
        {
            if (on)
                gs.forwardGpuGif(path, data, size);
            gs.m_packetData = nullptr;
        }
    } lateForward{*this, late, path, data, sizeBytes};
    ++s_hudPacket;
    s_hudPath = path;

    GifContinuation &continuation = m_gifContinuation[path & 3u];
    uint32_t offset = 0;
    if (continuation.active)
    {
        offset = continueGifTag(continuation, data, sizeBytes);
        if (offset >= sizeBytes)
            return;
    }
    else if (tryProcessNativeImageUploadPacket(data, sizeBytes))
        return;

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t packetIndex = s_debugGifPacketCount.fetch_add(1, std::memory_order_relaxed);
        if (packetIndex < 48u)
        {
            const uint64_t tagLo = loadLE64(data);
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            RUNTIME_LOG("[gs:gif] idx=" << packetIndex
                                        << " size=" << sizeBytes
                                        << " nloop=" << nloop
                                        << " flg=" << static_cast<uint32_t>(flg)
                                        << " nreg=" << nreg
                                        << " ctx0fbp=" << m_ctx[0].frame.fbp
                                        << " ctx1fbp=" << m_ctx[1].frame.fbp
                                        << std::endl);
        }
    });

    while (offset + 16 <= sizeBytes)
    {
        uint64_t tagLo = loadLE64(data + offset);
        uint64_t tagHi = loadLE64(data + offset + 8);
        offset += 16;

        m_curQ = 1.0f;

        uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFF);
        uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3);
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xF);
        if (nreg == 0)
            nreg = 16;

        recordGifTagDebugEventUnlocked(sizeBytes, nloop, flg, nreg);
        ++s_hudTag;
        if (s_hudLog.phase == HudLog::Capturing)
        {
            char line[200];
            std::snprintf(line, sizeof(line), "[hud] TAG path=%u pk=%u off=%u size=%u nloop=%u eop=%u pre=%u prim=0x%x flg=%u nreg=%u regs=%016llx\n",
                          static_cast<unsigned>(path), s_hudPacket, offset - 16u, sizeBytes, nloop,
                          static_cast<unsigned>((tagLo >> 15) & 1u), static_cast<unsigned>((tagLo >> 46) & 1u),
                          static_cast<unsigned>((tagLo >> 47) & 0x7FFu), flg, nreg, static_cast<unsigned long long>(tagHi));
            s_hudLog.text += line;
        }

        bool pre = ((tagLo >> 46) & 1) != 0;
        if (pre)
        {
            writeRegisterUnlocked(GS_REG_PRIM, (tagLo >> 47) & 0x7FF);
        }

        uint8_t regs[16];
        for (uint32_t i = 0; i < nreg; ++i)
            regs[i] = static_cast<uint8_t>((tagHi >> (i * 4)) & 0xF);

        // Hand the tag's data to continueGifTag: it consumes what this packet holds and keeps the rest pending.
        if (nloop == 0)
            continue;
        continuation.active = true;
        continuation.flg = flg;
        continuation.nreg = nreg;
        std::memcpy(continuation.regs, regs, sizeof(regs));
        continuation.regIndex = 0;
        continuation.regListWords = 0;
        if (flg == GIF_FMT_IMAGE || flg == 3u) // 3 = disabled, behaves as IMAGE
        {
            continuation.flg = GIF_FMT_IMAGE;
            continuation.loopsLeft = 0;
            continuation.imageBytesLeft = nloop * 16u;
        }
        else
        {
            continuation.loopsLeft = nloop;
            continuation.imageBytesLeft = 0;
        }
        offset += continueGifTag(continuation, data + offset, sizeBytes - offset);
    }
    static const bool contLog = std::getenv("VENDETTA_GSLOG") != nullptr;
    static uint32_t contLogs = 0u;
    if (contLog && continuation.active && path == 2u && contLogs < 40u && (continuation.imageBytesLeft > 4096u || continuation.loopsLeft > 64u))
    {
        ++contLogs;
        std::fprintf(stderr, "[gslog] path2 tag left open: flg=%u loopsLeft=%u nreg=%u regIndex=%u imageBytesLeft=%u packet=%u%c",
                     continuation.flg, continuation.loopsLeft, continuation.nreg, continuation.regIndex, continuation.imageBytesLeft, sizeBytes, 10);
        for (uint32_t q = 0; q + 16u <= sizeBytes; q += 16u)
            std::fprintf(stderr, "[gslog]   +%03x: %016llx %016llx%c", q, static_cast<unsigned long long>(loadLE64(data + q)),
                         static_cast<unsigned long long>(loadLE64(data + q + 8u)), 10);
    }
}

uint32_t GS::continueGifTag(GifContinuation &tag, const uint8_t *data, uint32_t sizeBytes)
{
    uint32_t offset = 0;
    if (tag.flg == GIF_FMT_IMAGE)
    {
        const uint32_t bytes = std::min(tag.imageBytesLeft, sizeBytes);
        if (bytes != 0u)
            processImageData(data, bytes);
        tag.imageBytesLeft -= bytes;
        offset = bytes;
        tag.active = tag.imageBytesLeft != 0u;
        return offset;
    }

    while (tag.loopsLeft != 0u)
    {
        if (tag.flg == GIF_FMT_PACKED)
        {
            if (offset + 16u > sizeBytes)
                return offset;
            const uint64_t lo = loadLE64(data + offset);
            const uint64_t hi = loadLE64(data + offset + 8);
            m_regSource = data + offset;
            offset += 16u;
            writeRegisterPacked(tag.regs[tag.regIndex], lo, hi);
            m_regSource = nullptr;
        }
        else // REGLIST
        {
            if (offset + 8u > sizeBytes)
                return offset;
            m_regSource = data + offset;
            writeRegisterUnlocked(tag.regs[tag.regIndex], loadLE64(data + offset));
            m_regSource = nullptr;
            offset += 8u;
            ++tag.regListWords;
        }
        if (++tag.regIndex == tag.nreg)
        {
            tag.regIndex = 0;
            --tag.loopsLeft;
        }
    }
    if (tag.flg == GIF_FMT_REGLIST && (tag.regListWords & 1u))
    {
        if (offset + 8u > sizeBytes)
        {
            // the padding word is in the next packet: keep the tag open (loopsLeft 0, regListWords odd) so the
            // next call skips exactly that 8-byte word instead of parsing the next packet 8 bytes off
            tag.active = true;
            return offset;
        }
        offset += 8u;
    }
    tag.active = false;
    return offset;
}

bool GS::processNativePackedGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    if (!data || sizeBytes < 16u || !m_backend)
        return false;

    if (!validatePackedGifPacket(data, sizeBytes))
        return false;
    // a whole packet, on a path the GIF arbiter never uses (after it is read in a widescreen frame, as above)
    const bool late = m_gpu && beginOverlayPacket(data);
    if (m_gpu && !late)
        m_gpu->gif(0u, data, sizeBytes);
    struct LateForward
    {
        GS &gs;
        bool on;
        const uint8_t *data;
        uint32_t size;
        ~LateForward()
        {
            if (on)
                gs.forwardGpuGif(0u, data, size);
            gs.m_packetData = nullptr;
        }
    } lateForward{*this, late, data, sizeBytes};
    ++s_hudPacket;
    s_hudPath = 9u; // the native packed path

    const bool processed = visitPackedGifPacket(data, sizeBytes, [&](const PackedGifPacketTag &tag)
                                                {
        m_curQ = 1.0f;

        recordGifTagDebugEventUnlocked(sizeBytes, tag.nloop, GIF_FMT_PACKED, tag.nreg);
        ++s_hudTag;

        const bool pre = ((tag.lo >> 46u) & 1u) != 0u;
        if (pre)
            writeRegisterUnlocked(GS_REG_PRIM, (tag.lo >> 47u) & 0x7FFu);

        uint32_t offset = tag.payloadOffset;
        for (uint32_t loop = 0u; loop < tag.nloop; ++loop)
        {
            for (uint32_t r = 0u; r < tag.nreg; ++r)
            {
                const uint64_t lo = loadLE64(data + offset);
                const uint64_t hi = loadLE64(data + offset + 8u);
                m_regSource = data + offset;
                offset += 16u;
                writeRegisterPacked(tag.regs[r], lo, hi);
                m_regSource = nullptr;
            }
        }

        return true; });

    if (!processed)
        return false;

    ++m_nativePackedGIFPacketCount;
    return true;
}

void GS::uploadImageNative(uint64_t bitbltbuf,
                           uint64_t trxpos,
                           uint64_t trxreg,
                           uint64_t trxdir,
                           const uint8_t *data,
                           uint32_t sizeBytes)
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    uploadImageNativeUnlocked(bitbltbuf, trxpos, trxreg, trxdir, data, sizeBytes);
}

void GS::uploadImageNativeUnlocked(uint64_t bitbltbuf,
                                   uint64_t trxpos,
                                   uint64_t trxreg,
                                   uint64_t trxdir,
                                   const uint8_t *data,
                                   uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0 || !m_backend)
        return;
    if (m_gpu)
    {
        // the same upload as a GIF packet: A+D writes of the four transfer registers, then IMAGE data
        const uint32_t qwords = sizeBytes / 16u;
        std::vector<uint8_t> packet((6u + qwords) * 16u, 0u);
        auto put = [&](uint32_t qword, uint64_t lo, uint64_t hi)
        {
            std::memcpy(packet.data() + qword * 16u, &lo, 8u);
            std::memcpy(packet.data() + qword * 16u + 8u, &hi, 8u);
        };
        put(0u, 4u | (1ull << 15) | (1ull << 60), 0xEu); // NLOOP 4, EOP, PACKED, NREG 1: A+D
        put(1u, bitbltbuf, GS_REG_BITBLTBUF);
        put(2u, trxpos, GS_REG_TRXPOS);
        put(3u, trxreg, GS_REG_TRXREG);
        put(4u, trxdir, GS_REG_TRXDIR);
        put(5u, qwords | (1ull << 15) | (2ull << 58), 0u); // NLOOP = the data, EOP, IMAGE
        std::memcpy(packet.data() + 6u * 16u, data, qwords * 16u);
        forwardGpuGif(0u, packet.data(), static_cast<uint32_t>(packet.size()));
    }

    writeRegisterUnlocked(GS_REG_BITBLTBUF, bitbltbuf);
    writeRegisterUnlocked(GS_REG_TRXPOS, trxpos);
    writeRegisterUnlocked(GS_REG_TRXREG, trxreg);
    writeRegisterUnlocked(GS_REG_TRXDIR, trxdir);
    processImageData(data, sizeBytes);
    ++m_nativeImageUploadCount;
}

bool GS::tryProcessNativeImageUploadPacket(const uint8_t *data, uint32_t sizeBytes)
{
    constexpr uint32_t kSetupRegisters = 4u;
    constexpr uint32_t kPackedAdPayloadBytes = kSetupRegisters * 16u;
    constexpr uint64_t kPackedAdDescriptor = 0x0Eull;

    if (!data || sizeBytes < 16u + kPackedAdPayloadBytes + 16u)
        return false;

    const uint64_t setupTagLo = loadLE64(data);
    const uint64_t setupTagHi = loadLE64(data + 8u);
    const uint32_t setupNloop = static_cast<uint32_t>(setupTagLo & 0x7FFFu);
    const uint8_t setupFlg = static_cast<uint8_t>((setupTagLo >> 58u) & 0x3u);
    uint32_t setupNreg = static_cast<uint32_t>((setupTagLo >> 60u) & 0xFu);
    if (setupNreg == 0u)
        setupNreg = 16u;

    if (setupNloop != kSetupRegisters ||
        setupFlg != GIF_FMT_PACKED ||
        setupNreg != 1u ||
        (setupTagHi & 0xFull) != kPackedAdDescriptor)
    {
        return false;
    }

    uint64_t regs[kSetupRegisters] = {};
    uint32_t offset = 16u;
    constexpr uint8_t expectedRegs[kSetupRegisters] = {
        GS_REG_BITBLTBUF,
        GS_REG_TRXPOS,
        GS_REG_TRXREG,
        GS_REG_TRXDIR,
    };

    for (uint32_t i = 0; i < kSetupRegisters; ++i)
    {
        regs[i] = loadLE64(data + offset);
        const uint64_t reg = loadLE64(data + offset + 8u);
        if ((reg & 0xFFu) != expectedRegs[i])
            return false;
        offset += 16u;
    }

    const uint32_t trxdirMode = static_cast<uint32_t>(regs[3] & 0x3ull);
    const uint32_t rrw = static_cast<uint32_t>(regs[2] & 0xFFFull);
    const uint32_t rrh = static_cast<uint32_t>((regs[2] >> 32u) & 0xFFFull);
    if (trxdirMode != 0u || rrw == 0u || rrh == 0u)
        return false;

    if (offset + 16u > sizeBytes)
        return false;

    const uint64_t imageTagLo = loadLE64(data + offset);
    const uint8_t imageFlg = static_cast<uint8_t>((imageTagLo >> 58u) & 0x3u);
    const uint32_t imageNloop = static_cast<uint32_t>(imageTagLo & 0x7FFFu);
    if (imageFlg != GIF_FMT_IMAGE || imageNloop == 0u)
        return false;

    offset += 16u;
    const uint64_t imageBytes64 = static_cast<uint64_t>(imageNloop) * 16ull;
    if (imageBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t imageBytes = static_cast<uint32_t>(imageBytes64);
    if (offset + imageBytes != sizeBytes)
        return false;

    uploadImageNativeUnlocked(regs[0], regs[1], regs[2], regs[3], data + offset, imageBytes);
    return true;
}

void GS::writeRegisterPacked(uint8_t regDesc, uint64_t lo, uint64_t hi)
{
    switch (regDesc)
    {
    case 0x00:
        writeRegisterUnlocked(GS_REG_PRIM, lo & 0x7FF);
        break;
    case 0x01:
        m_curR = static_cast<uint8_t>(lo & 0xFF);
        m_curG = static_cast<uint8_t>((lo >> 32) & 0xFF);
        m_curB = static_cast<uint8_t>(hi & 0xFF);
        m_curA = static_cast<uint8_t>((hi >> 32) & 0xFF);
        break;
    case 0x02:
    {
        uint32_t sBits = static_cast<uint32_t>(lo & 0xFFFFFFFF);
        uint32_t tBits = static_cast<uint32_t>((lo >> 32) & 0xFFFFFFFF);
        uint32_t qBits = static_cast<uint32_t>(hi & 0xFFFFFFFF);
        std::memcpy(&m_curS, &sBits, 4);
        std::memcpy(&m_curT, &tBits, 4);
        std::memcpy(&m_curQ, &qBits, 4);
        if (m_curQ == 0.0f)
            m_curQ = 1.0f;
        break;
    }
    case 0x03:
        m_curU = static_cast<uint16_t>(lo & 0x3FFFu);
        m_curV = static_cast<uint16_t>((lo >> 32) & 0x3FFFu);
        break;
    case 0x04:
    {
        uint16_t x = static_cast<uint16_t>(lo & 0xFFFF);
        uint16_t y = static_cast<uint16_t>((lo >> 32) & 0xFFFF);
        uint32_t z = static_cast<uint32_t>((hi >> 4) & 0xFFFFFF);
        uint8_t f = static_cast<uint8_t>((hi >> 36) & 0xFF);
        bool adk = ((hi >> 47) & 1) != 0;
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyzf] idx=" << debugIndex
                                                    << " x=" << x
                                                    << " y=" << y
                                                    << " z=0x" << std::hex << z
                                                    << std::dec
                                                    << " fog=" << static_cast<uint32_t>(f)
                                                    << " kick=" << static_cast<uint32_t>(!adk ? 1u : 0u)
                                                    << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                    << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        m_vtxSource[m_vtxCount % kMaxVerts] = currentVertexSource();
        vtx.x = static_cast<float>(x) / 16.0f;
        vtx.y = static_cast<float>(y) / 16.0f;
        vtx.z = static_cast<float>(z);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = f;
        vertexKick(!adk);
        break;
    }
    case 0x05:
    {
        uint16_t x = static_cast<uint16_t>(lo & 0xFFFF);
        uint16_t y = static_cast<uint16_t>((lo >> 32) & 0xFFFF);
        uint32_t z = static_cast<uint32_t>(hi & 0xFFFFFFFF);
        bool adk = ((hi >> 47) & 1) != 0;
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyz] idx=" << debugIndex
                                                   << " x=" << x
                                                   << " y=" << y
                                                   << " z=0x" << std::hex << z
                                                   << std::dec
                                                   << " kick=" << static_cast<uint32_t>(!adk ? 1u : 0u)
                                                   << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                   << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        m_vtxSource[m_vtxCount % kMaxVerts] = currentVertexSource();
        vtx.x = static_cast<float>(x) / 16.0f;
        vtx.y = static_cast<float>(y) / 16.0f;
        vtx.z = static_cast<float>(z);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        vertexKick(!adk);
        break;
    }
    case 0x0A:
        m_curFog = static_cast<uint8_t>((hi >> 36) & 0xFF);
        break;
    case 0x0C:
    {
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyzf3] idx=" << debugIndex
                                                     << " x=" << static_cast<uint32_t>(lo & 0xFFFFu)
                                                     << " y=" << static_cast<uint32_t>((lo >> 32) & 0xFFFFu)
                                                     << " kick=0"
                                                     << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                     << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        m_vtxSource[m_vtxCount % kMaxVerts] = currentVertexSource();
        vtx.x = static_cast<float>(lo & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((lo >> 32) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>((hi >> 4) & 0xFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = static_cast<uint8_t>((hi >> 36) & 0xFF);
        vertexKick(false);
        break;
    }
    case 0x0D:
    {
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyz3] idx=" << debugIndex
                                                    << " x=" << static_cast<uint32_t>(lo & 0xFFFFu)
                                                    << " y=" << static_cast<uint32_t>((lo >> 32) & 0xFFFFu)
                                                    << " kick=0"
                                                    << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                    << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        m_vtxSource[m_vtxCount % kMaxVerts] = currentVertexSource();
        vtx.x = static_cast<float>(lo & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((lo >> 32) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>(hi & 0xFFFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        vertexKick(false);
        break;
    }
    case 0x0E:
    {
        uint8_t addr = static_cast<uint8_t>(hi & 0xFF);
        writeRegisterUnlocked(addr, lo);
        break;
    }
    case 0x0F:
        break;
    default:
        writeRegisterUnlocked(regDesc, lo);
        break;
    }
}

void GS::writeRegister(uint8_t regAddr, uint64_t value)
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    // 0x59-0x5F are this front end's own aliases of display registers, which reach the graphics card at presentation
    if (m_gpu && (regAddr < 0x59u || regAddr > 0x5Fu))
        forwardGpuReg(regAddr, value);
    writeRegisterUnlocked(regAddr, value);
}

void GS::writeRegisterUnlocked(uint8_t regAddr, uint64_t value)
{
    const bool interestingReg =
        regAddr == GS_REG_PRIM ||
        regAddr == GS_REG_RGBAQ ||
        regAddr == GS_REG_ST ||
        regAddr == GS_REG_UV ||
        regAddr == GS_REG_XYZ2 ||
        regAddr == GS_REG_XYZ3 ||
        regAddr == GS_REG_XYZF2 ||
        regAddr == GS_REG_XYZF3 ||
        regAddr == GS_REG_TEX0_1 ||
        regAddr == GS_REG_TEX0_2 ||
        regAddr == GS_REG_TEX2_1 ||
        regAddr == GS_REG_TEX2_2 ||
        regAddr == GS_REG_TEXCLUT ||
        regAddr == GS_REG_TEXA ||
        regAddr == GS_REG_XYOFFSET_1 ||
        regAddr == GS_REG_XYOFFSET_2 ||
        regAddr == GS_REG_SCISSOR_1 ||
        regAddr == GS_REG_SCISSOR_2 ||
        regAddr == GS_REG_FRAME_1 ||
        regAddr == GS_REG_FRAME_2 ||
        regAddr == GS_REG_ALPHA_1 ||
        regAddr == GS_REG_ALPHA_2 ||
        regAddr == GS_REG_TEST_1 ||
        regAddr == GS_REG_TEST_2 ||
        regAddr == GS_REG_BITBLTBUF ||
        regAddr == GS_REG_TRXPOS ||
        regAddr == GS_REG_TRXREG ||
        regAddr == GS_REG_TRXDIR;

    PS2_IF_AGRESSIVE_LOGS({
        if (interestingReg)
        {
            const uint32_t debugIndex = s_debugGsRegisterCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 128u)
            {
                RUNTIME_LOG("[gs:reg] idx=" << debugIndex
                                            << " reg=0x" << std::hex << static_cast<uint32_t>(regAddr)
                                            << " value=0x" << value
                                            << std::dec
                                            << std::endl);
            }
        }
    });

    const bool isCopyRelevantReg =
        regAddr == GS_REG_PRIM ||
        regAddr == GS_REG_TEX0_2 ||
        regAddr == GS_REG_TEX1_2 ||
        regAddr == GS_REG_ALPHA_2 ||
        regAddr == GS_REG_TEST_2 ||
        regAddr == GS_REG_PABE ||
        regAddr == GS_REG_FRAME_2 ||
        regAddr == GS_REG_XYOFFSET_2 ||
        regAddr == GS_REG_SCISSOR_2;
    PS2_IF_AGRESSIVE_LOGS({
        if (isCopyRelevantReg &&
            s_debugCopyRegCount.fetch_add(1u, std::memory_order_relaxed) < 64u)
        {
            RUNTIME_LOG("[gs:copy-reg] reg=0x"
                        << std::hex << static_cast<uint32_t>(regAddr)
                        << " value=0x" << value
                        << std::dec
                        << " primCtxt=" << static_cast<uint32_t>(m_prim.ctxt)
                        << " ctx0fbp=" << m_ctx[0].frame.fbp
                        << " ctx1fbp=" << m_ctx[1].frame.fbp
                        << std::endl);
        }
    });

    switch (regAddr)
    {
    case GS_REG_PRIM:
    {
        m_primRegister = decodePrimRegister(value);
        if (m_prmodecont)
        {
            m_prim = m_primRegister;
        }
        else
        {
            // PRIM always selects the primitive topology. With AC=0, all
            // rendering attributes remain sourced from PRMODE.
            m_prim.type = m_primRegister.type;
        }
        m_vtxCount = 0;
        m_vtxIndex = 0;
        break;
    }
    case GS_REG_RGBAQ:
    {
        m_curR = static_cast<uint8_t>(value & 0xFF);
        m_curG = static_cast<uint8_t>((value >> 8) & 0xFF);
        m_curB = static_cast<uint8_t>((value >> 16) & 0xFF);
        m_curA = static_cast<uint8_t>((value >> 24) & 0xFF);
        uint32_t qBits = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        std::memcpy(&m_curQ, &qBits, 4);
        if (m_curQ == 0.0f)
            m_curQ = 1.0f;
        break;
    }
    case GS_REG_ST:
    {
        uint32_t sBits = static_cast<uint32_t>(value & 0xFFFFFFFF);
        uint32_t tBits = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        std::memcpy(&m_curS, &sBits, 4);
        std::memcpy(&m_curT, &tBits, 4);
        break;
    }
    case GS_REG_UV:
    {
        m_curU = static_cast<uint16_t>(value & 0x3FFFu);
        m_curV = static_cast<uint16_t>((value >> 16) & 0x3FFFu);
        break;
    }
    case GS_REG_XYZF2:
    case GS_REG_XYZF3:
    {
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        m_vtxSource[m_vtxCount % kMaxVerts] = currentVertexSource();
        vtx.x = static_cast<float>(value & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((value >> 16) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<double>((value >> 32) & 0xFFFFFF);
        vtx.fog = static_cast<uint8_t>((value >> 56) & 0xFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vertexKick(regAddr == GS_REG_XYZF2);
        break;
    }
    case GS_REG_XYZ2:
    case GS_REG_XYZ3:
    {
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        m_vtxSource[m_vtxCount % kMaxVerts] = currentVertexSource();
        vtx.x = static_cast<float>(value & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((value >> 16) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<double>((value >> 32) & 0xFFFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        vertexKick(regAddr == GS_REG_XYZ2);
        break;
    }
    case GS_REG_TEX0_1:
    case GS_REG_TEX0_2:
    {
        int ci = (regAddr == GS_REG_TEX0_2) ? 1 : 0;
        auto &t = m_ctx[ci].tex0;
        t.tbp0 = static_cast<uint32_t>(value & 0x3FFF);
        t.tbw = static_cast<uint8_t>((value >> 14) & 0x3F);
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
        t.tw = static_cast<uint8_t>((value >> 26) & 0xF);
        t.th = static_cast<uint8_t>((value >> 30) & 0xF);
        t.tcc = static_cast<uint8_t>((value >> 34) & 0x1);
        t.tfx = static_cast<uint8_t>((value >> 35) & 0x3);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
        m_backend->LoadClut(t, m_texclut);
        if (s_hudLog.phase == HudLog::Capturing)
        {
            char line[160];
            std::snprintf(line, sizeof(line), "[hud] TEX0_%d tbp=%u tbw=%u psm=0x%x tw=%u th=%u cbp=%u cld=%u path=%u pk=%u\n", ci + 1,
                          t.tbp0, t.tbw, t.psm, t.tw, t.th, t.cbp, t.cld, s_hudPath, s_hudPacket);
            s_hudLog.text += line;
        }
        static const bool texLog = std::getenv("VENDETTA_GSLOG") != nullptr;
        static uint32_t texLogs = 0u;
        if (texLog && texLogs < 3000u && (t.psm == 0x14u || t.psm == 0x13u))
        {
            ++texLogs;
            std::fprintf(stderr, "[gslog] tex0 ctx=%d tbp=0x%x tbw=%u psm=0x%x tw=%u th=%u tcc=%u tfx=%u cbp=0x%x cpsm=0x%x csm=%u csa=%u cld=%u%c",
                         ci, t.tbp0, t.tbw, t.psm, t.tw, t.th, t.tcc, t.tfx, t.cbp, t.cpsm, t.csm, t.csa, t.cld, 10);
        }
        break;
    }
    case GS_REG_CLAMP_1:
    case GS_REG_CLAMP_2:
    {
        int ci = (regAddr == GS_REG_CLAMP_2) ? 1 : 0;
        m_ctx[ci].clamp = value;
        break;
    }
    case GS_REG_FOG:
        m_curFog = static_cast<uint8_t>((value >> 56) & 0xFF);
        break;
    case GS_REG_TEX1_1:
    case GS_REG_TEX1_2:
    {
        int ci = (regAddr == GS_REG_TEX1_2) ? 1 : 0;
        m_ctx[ci].tex1 = value;
        break;
    }
    case GS_REG_TEX2_1:
    case GS_REG_TEX2_2:
    {
        int ci = (regAddr == GS_REG_TEX2_2) ? 1 : 0;
        auto &t = m_ctx[ci].tex0;
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
        m_backend->LoadClut(t, m_texclut);
        break;
    }
    case GS_REG_XYOFFSET_1:
    case GS_REG_XYOFFSET_2:
    {
        int ci = (regAddr == GS_REG_XYOFFSET_2) ? 1 : 0;
        m_ctx[ci].xyoffset.ofx = static_cast<uint16_t>(value & 0xFFFF);
        m_ctx[ci].xyoffset.ofy = static_cast<uint16_t>((value >> 32) & 0xFFFF);
        break;
    }
    case GS_REG_PRMODECONT:
    {
        m_prmodecont = (value & 1) != 0;
        const GSPrimType type = m_primRegister.type;
        m_prim = m_prmodecont ? m_primRegister : m_prmodeRegister;
        m_prim.type = type;
        break;
    }
    case GS_REG_PRMODE:
    {
        m_prmodeRegister = decodePrimRegister(value);
        if (!m_prmodecont)
        {
            const GSPrimType type = m_primRegister.type;
            m_prim = m_prmodeRegister;
            m_prim.type = type;
        }
        break;
    }
    case GS_REG_TEXCLUT:
        m_texclut.cbw = static_cast<uint8_t>(value & 0x3Fu);
        m_texclut.cou = static_cast<uint8_t>((value >> 6) & 0x3Fu);
        m_texclut.cov = static_cast<uint16_t>((value >> 12) & 0x3FFu);
        break;
    case GS_REG_SCISSOR_1:
    case GS_REG_SCISSOR_2:
    {
        int ci = (regAddr == GS_REG_SCISSOR_2) ? 1 : 0;
        m_ctx[ci].scissor.x0 = static_cast<uint16_t>(value & 0x7FF);
        m_ctx[ci].scissor.x1 = static_cast<uint16_t>((value >> 16) & 0x7FF);
        m_ctx[ci].scissor.y0 = static_cast<uint16_t>((value >> 32) & 0x7FF);
        m_ctx[ci].scissor.y1 = static_cast<uint16_t>((value >> 48) & 0x7FF);
        break;
    }
    case GS_REG_ALPHA_1:
    case GS_REG_ALPHA_2:
    {
        int ci = (regAddr == GS_REG_ALPHA_2) ? 1 : 0;
        m_ctx[ci].alpha = value;
        break;
    }
    case GS_REG_TEST_1:
    case GS_REG_TEST_2:
    {
        int ci = (regAddr == GS_REG_TEST_2) ? 1 : 0;
        m_ctx[ci].test = value;
        break;
    }
    case GS_REG_FRAME_1:
    case GS_REG_FRAME_2:
    {
        int ci = (regAddr == GS_REG_FRAME_2) ? 1 : 0;
        m_ctx[ci].frame.fbp = static_cast<uint32_t>(value & 0x1FF);
        m_ctx[ci].frame.fbw = static_cast<uint32_t>((value >> 16) & 0x3F);
        m_ctx[ci].frame.psm = static_cast<uint8_t>((value >> 24) & 0x3F);
        m_ctx[ci].frame.fbmsk = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        break;
    }
    case GS_REG_ZBUF_1:
    case GS_REG_ZBUF_2:
    {
        int ci = (regAddr == GS_REG_ZBUF_2) ? 1 : 0;
        m_ctx[ci].zbuf.zbp = value & 0x1FF;
        m_ctx[ci].zbuf.psm = ((value >> 24) & 0xF) | 0x30;
        m_ctx[ci].zbuf.zmask = (value >> 32) & 1;
        break;
    }
    case GS_REG_FBA_1:
    case GS_REG_FBA_2:
    {
        int ci = (regAddr == GS_REG_FBA_2) ? 1 : 0;
        m_ctx[ci].fba = value;
        break;
    }
    case GS_REG_BITBLTBUF:
    {
        m_bitbltbuf.sbp = static_cast<uint32_t>(value & 0x3FFF);
        m_bitbltbuf.sbw = static_cast<uint8_t>((value >> 16) & 0x3F);
        m_bitbltbuf.spsm = static_cast<uint8_t>((value >> 24) & 0x3F);
        m_bitbltbuf.dbp = static_cast<uint32_t>((value >> 32) & 0x3FFF);
        m_bitbltbuf.dbw = static_cast<uint8_t>((value >> 48) & 0x3F);
        m_bitbltbuf.dpsm = static_cast<uint8_t>((value >> 56) & 0x3F);
        // HCTP diag (HCTP_FONTLOG=1): image transfers into the font sheets' area (dbp 0x2200-0x23ff)
        {
            static const bool fontLog = std::getenv("HCTP_FONTLOG") != nullptr;
            static int logged = 0;
            static const auto t0 = std::chrono::steady_clock::now();
            if (fontLog && m_bitbltbuf.dpsm == 0x14u && logged < 400)
            {
                ++logged;
                std::fprintf(stderr, "[fontxfer] t=%.1f dbp=0x%x dbw=%u dpsm=0x%x\n",
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                             m_bitbltbuf.dbp, m_bitbltbuf.dbw, m_bitbltbuf.dpsm);
            }
        }
        break;
    }
    case GS_REG_TRXPOS:
    {
        m_trxpos.ssax = static_cast<uint16_t>(value & 0x7FF);
        m_trxpos.ssay = static_cast<uint16_t>((value >> 16) & 0x7FF);
        m_trxpos.dsax = static_cast<uint16_t>((value >> 32) & 0x7FF);
        m_trxpos.dsay = static_cast<uint16_t>((value >> 48) & 0x7FF);
        m_trxpos.dir = static_cast<uint8_t>((value >> 59) & 0x3);
        break;
    }
    case GS_REG_TRXREG:
    {
        m_trxreg.rrw = static_cast<uint16_t>(value & 0xFFF);
        m_trxreg.rrh = static_cast<uint16_t>((value >> 32) & 0xFFF);
        break;
    }
    case GS_REG_TRXDIR:
    {
        m_trxdir = static_cast<uint32_t>(value & 0x3);

        if (m_backend)
        {
            GSTransferCommand command{};
            command.bitbltbuf = m_bitbltbuf;
            command.trxpos = m_trxpos;
            command.trxreg = m_trxreg;
            command.direction = m_trxdir;
            m_backend->BeginTransfer(command);
            m_vramWrittenSincePresent = true;
        }
        recordTransferDebugEventUnlocked();
        break;
    }
    case GS_REG_HWREG:
    {
        uint8_t buf[8];
        std::memcpy(buf, &value, 8);
        processImageData(buf, 8);
        break;
    }
    case GS_REG_PABE:
        m_pabe = (value & 1u) != 0u;
        break;
    case GS_REG_FOGCOL:
        m_fogR = static_cast<uint8_t>(value & 0xFFu);
        m_fogG = static_cast<uint8_t>((value >> 8) & 0xFFu);
        m_fogB = static_cast<uint8_t>((value >> 16) & 0xFFu);
        break;
    case GS_REG_TEXFLUSH:
        if (m_backend)
            m_backend->TextureFlush();
        break;
    case GS_REG_SCANMSK:
        m_scanmsk = value;
        break;
    case GS_REG_DIMX:
        m_dimx = value;
        break;
    case GS_REG_DTHE:
        m_dthe = value;
        break;
    case GS_REG_COLCLAMP:
        m_colclamp = value;
        break;
    case GS_REG_MIPTBP1_1:
    case GS_REG_MIPTBP1_2:
    {
        const int ci = (regAddr == GS_REG_MIPTBP1_2) ? 1 : 0;
        m_ctx[ci].miptbp1 = value;
        break;
    }
    case GS_REG_MIPTBP2_1:
    case GS_REG_MIPTBP2_2:
    {
        const int ci = (regAddr == GS_REG_MIPTBP2_2) ? 1 : 0;
        m_ctx[ci].miptbp2 = value;
        break;
    }
    case GS_REG_TEXA:
    {
        m_texa.ta0 = static_cast<uint8_t>(value & 0xFFu);
        m_texa.aem = ((value >> 15) & 0x1u) != 0u;
        m_texa.ta1 = static_cast<uint8_t>((value >> 32) & 0xFFu);
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t texaIndex = s_debugTexaWriteCount.fetch_add(1u, std::memory_order_relaxed);
            if (texaIndex < 24u)
            {
                RUNTIME_LOG("[gs:texa] idx=" << texaIndex
                                             << " value=0x" << std::hex << value
                                             << " ta0=0x" << ((value >> 0) & 0xFFu)
                                             << " aem=" << ((value >> 15) & 0x1u)
                                             << " ta1=0x" << ((value >> 32) & 0xFFu)
                                             << std::dec
                                             << std::endl);
            }
        });
        break;
    }
    case GS_REG_SIGNAL:
    {
        if (m_privRegs)
        {
            uint32_t id = static_cast<uint32_t>(value & 0xFFFFFFFF);
            uint32_t mask = static_cast<uint32_t>(value >> 32);
            uint32_t lo = static_cast<uint32_t>(m_privRegs->siglblid & 0xFFFFFFFF);
            lo = (lo & ~mask) | (id & mask);
            m_privRegs->siglblid = (m_privRegs->siglblid & 0xFFFFFFFF00000000ULL) | lo;
            m_privRegs->csr.fetch_or(0x1);
        }
        break;
    }
    case GS_REG_FINISH:
    {
        if (m_backend)
        {
            m_backend->Flush();
            m_backend->Sync(GSSyncReason::Finish);
        }
        if (m_privRegs)
            m_privRegs->csr.fetch_or(0x2);
        break;
    }
    case GS_REG_LABEL:
    {
        if (m_privRegs)
        {
            uint32_t id = static_cast<uint32_t>(value & 0xFFFFFFFF);
            uint32_t mask = static_cast<uint32_t>(value >> 32);
            uint32_t hi = static_cast<uint32_t>(m_privRegs->siglblid >> 32);
            hi = (hi & ~mask) | (id & mask);
            m_privRegs->siglblid = (static_cast<uint64_t>(hi) << 32) | (m_privRegs->siglblid & 0xFFFFFFFF);
        }
        break;
    }
    case 0x59:
        if (m_privRegs)
            m_privRegs->dispfb1 = value;
        break;
    case 0x5a:
        if (m_privRegs)
            m_privRegs->display1 = value;
        break;
    case 0x5b:
        if (m_privRegs)
            m_privRegs->dispfb2 = value;
        break;
    case 0x5c:
        if (m_privRegs)
            m_privRegs->display2 = value;
        break;
    case 0x5f:
        if (m_privRegs)
            m_privRegs->bgcolor = value;
        break;
    default:
        break;
    }

    recordRegisterDebugEventUnlocked(regAddr, value);
}

static const auto s_fontLogT0 = std::chrono::steady_clock::now();

void GS::vertexKick(bool drawing)
{
    ++m_vtxCount;
    ++m_vtxIndex;

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t debugIndex = s_debugGsVertexKickCount.fetch_add(1, std::memory_order_relaxed);
        if (debugIndex < 96u)
        {
            RUNTIME_LOG("[gs:kick] idx=" << debugIndex
                                         << " drawing=" << static_cast<uint32_t>(drawing ? 1u : 0u)
                                         << " prim=" << static_cast<uint32_t>(m_prim.type)
                                         << " vtxCount=" << m_vtxCount
                                         << std::endl);
        }
    });

    int needed = 0;
    switch (m_prim.type)
    {
    case GS_PRIM_POINT:
        needed = 1;
        break;
    case GS_PRIM_LINE:
        needed = 2;
        break;
    case GS_PRIM_LINESTRIP:
        needed = 2;
        break;
    case GS_PRIM_TRIANGLE:
        needed = 3;
        break;
    case GS_PRIM_TRISTRIP:
        needed = 3;
        break;
    case GS_PRIM_TRIFAN:
        needed = 3;
        break;
    case GS_PRIM_SPRITE:
        needed = 2;
        break;
    default:
        return;
    }

    if (m_vtxCount < needed)
        return;

    if (drawing && m_backend)
    {
        GSPrimitiveBatch batch = buildDrawBatch(needed);
        // HCTP diag: HCTP_CAPTURE_TBP=<tbp> (+ HCTP_CAPTURE_SEC=<s>): at the first draw sampling that texture after
        // the time, write tex_capture.txt (working dir): draw registers, the texture's indices (de-swizzled through
        // the CPU renderer's addressing on a snapshot of this renderer's GS memory) and the CLUT entries in CSM order.
        {
            static const char *s_capEnv = std::getenv("HCTP_CAPTURE_TBP");
            static const long s_capTbp = s_capEnv ? std::atol(s_capEnv) : -1;
            static const char *s_capSecEnv = std::getenv("HCTP_CAPTURE_SEC");
            static const double s_capSec = s_capSecEnv ? std::atof(s_capSecEnv) : 0.0;
            static const auto s_capT0 = std::chrono::steady_clock::now();
            static bool s_captured = false;
            const GSContext &cc = batch.state.context;
            if (s_capTbp >= 0 && !s_captured && batch.state.prim.tme && static_cast<long>(cc.tex0.tbp0) == s_capTbp &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - s_capT0).count() >= s_capSec)
            {
                s_captured = true;
                std::vector<uint8_t> vram;
                if (m_gpu)
                    m_gpu->snapshotVram(vram);
                else
                    m_backend->SnapshotVram(vram);
                if (FILE *f = std::fopen("tex_capture.txt", "w"))
                {
                    const GSTex0Reg &t = cc.tex0;
                    std::fprintf(f, "renderer=%s\nPRIM type=%u iip=%u tme=%u fge=%u abe=%u aa1=%u fst=%u ctxt=%u fix=%u\n",
                                 m_gpu ? "paraLLEl" : "native", static_cast<unsigned>(batch.state.prim.type),
                                 batch.state.prim.iip ? 1u : 0u, batch.state.prim.tme ? 1u : 0u, batch.state.prim.fge ? 1u : 0u,
                                 batch.state.prim.abe ? 1u : 0u, batch.state.prim.aa1 ? 1u : 0u, batch.state.prim.fst ? 1u : 0u,
                                 batch.state.prim.ctxt ? 1u : 0u, batch.state.prim.fix ? 1u : 0u);
                    std::fprintf(f, "TEX0 tbp=%u tbw=%u psm=0x%x tw=%u th=%u tcc=%u tfx=%u cbp=%u cpsm=0x%x csm=%u csa=%u cld=%u\n",
                                 t.tbp0, t.tbw, t.psm, t.tw, t.th, t.tcc, t.tfx, t.cbp, t.cpsm, t.csm, t.csa, t.cld);
                    std::fprintf(f, "TEX1=%016llx CLAMP=%016llx ALPHA=%016llx TEST=%016llx FBA=%u\n",
                                 static_cast<unsigned long long>(cc.tex1), static_cast<unsigned long long>(cc.clamp),
                                 static_cast<unsigned long long>(cc.alpha), static_cast<unsigned long long>(cc.test),
                                 static_cast<unsigned>(cc.fba));
                    std::fprintf(f, "TEXA ta0=%u ta1=%u aem=%u TEXCLUT cbw=%u cou=%u cov=%u PABE=%u FRAME fbp=%u psm=0x%x fbmsk=%08x ZBUF zbp=%u zpsm=0x%x zmsk=%u\n",
                                 batch.state.texa.ta0, batch.state.texa.ta1, batch.state.texa.aem ? 1u : 0u,
                                 batch.state.texclut.cbw, batch.state.texclut.cou, batch.state.texclut.cov, m_pabe ? 1u : 0u,
                                 cc.frame.fbp, cc.frame.psm, cc.frame.fbmsk, static_cast<unsigned>(cc.zbuf.zbp),
                                 static_cast<unsigned>(cc.zbuf.psm), cc.zbuf.zmask ? 1u : 0u);
                    for (uint32_t i = 0; i < batch.vertexCount; ++i)
                    {
                        const GSVertex &v = batch.vertices[i];
                        std::fprintf(f, "V%u xy=%.2f,%.2f z=%.0f rgba=%u,%u,%u,%u stq=%g,%g,%g uv=%u,%u fog=%u\n", i, v.x, v.y, v.z,
                                     v.r, v.g, v.b, v.a, v.s, v.t, v.q, v.u, v.v, v.fog);
                    }
                    if (!vram.empty())
                    {
                        GSCpuBackend reader(1u);
                        reader.Initialize(vram.data(), static_cast<uint32_t>(vram.size()));
                        const uint32_t tw = 1u << std::min<uint32_t>(t.tw, 10u), th = 1u << std::min<uint32_t>(t.th, 10u);
                        std::fprintf(f, "INDICES %ux%u\n", tw, th);
                        for (uint32_t y = 0; y < th && y < 64u; ++y)
                        {
                            for (uint32_t x = 0; x < tw && x < 64u; ++x)
                                std::fprintf(f, "%02x", reader.ReadVram(t.psm, t.tbp0, std::max<uint32_t>(t.tbw, 1u), x, y) & 0xFFu);
                            std::fprintf(f, "\n");
                        }
                        std::fprintf(f, "CLUT (entry: rgba as stored, CSM1 order)\n");
                        const uint32_t entries = (t.psm == 0x14u || t.psm == 0x24u || t.psm == 0x2Cu) ? 16u : 256u;
                        for (uint32_t e = 0; e < entries; ++e)
                        {
                            uint32_t sx, sy;
                            if (t.csm == 0u)
                            {
                                const uint32_t si = entries == 16u ? e : ((e & ~0x18u) | ((e & 0x08u) << 1u) | ((e & 0x10u) >> 1u));
                                sx = entries == 16u ? (e & 7u) : (si & 0x0Fu);
                                sy = entries == 16u ? (e >> 3u) : (si >> 4u);
                            }
                            else
                            {
                                sx = (static_cast<uint32_t>(batch.state.texclut.cou) << 4u) + e;
                                sy = batch.state.texclut.cov;
                            }
                            std::fprintf(f, "%3u:%08x%s", e, reader.ReadVram(t.cpsm, t.cbp, 1u, sx, sy), (e % 8u) == 7u ? "\n" : " ");
                        }
                        reader.Initialize(nullptr, 0u);
                    }
                    std::fclose(f);
                    std::fprintf(stderr, "[texcap] tbp=%ld captured%c", s_capTbp, 10);
                }
            }
        }
        // HCTP diag (HCTP_FONTLOG=1): draws whose frame / Z buffer reaches the font sheets (blocks 0x2240-0x233f)
        {
            static const bool fontLog = std::getenv("HCTP_FONTLOG") != nullptr;
            if (fontLog)
            {
                const GSContext &c = batch.state.context;
                const uint32_t lines = static_cast<uint32_t>(c.scissor.y1) + 1u;
                auto pagesFor = [&](uint8_t psm, uint32_t fbw) {
                    const bool half = (psm & 0x0Fu) == 0x02u || (psm & 0x0Fu) == 0x0Au; // 16-bit: 64-line pages
                    return std::max<uint32_t>(fbw, 1u) * ((lines + (half ? 63u : 31u)) / (half ? 64u : 32u));
                };
                const uint32_t fb0 = c.frame.fbp * 32u, fb1 = fb0 + pagesFor(c.frame.psm, c.frame.fbw) * 32u;
                const bool zWrites = (c.test & (1ull << 16)) != 0u && !c.zbuf.zmask;
                const uint32_t z0 = c.zbuf.zbp * 32u, z1 = z0 + pagesFor(c.zbuf.psm, c.frame.fbw) * 32u;
                // textured 4-bit sprites (text): which texture / CLUT they sample, per distinct TEX0
                if (m_prim.tme && (m_prim.type == GS_PRIM_SPRITE || m_prim.type == GS_PRIM_TRISTRIP) && std::chrono::duration<double>(std::chrono::steady_clock::now() - s_fontLogT0).count() > 400.0)
                {
                    static std::set<uint64_t> texSeen;
                    const uint64_t tk = (static_cast<uint64_t>(c.tex0.tbp0) << 32) | (c.tex0.cbp << 8) | (c.tex0.tw << 4) | c.tex0.th;
                    static const auto tt0 = std::chrono::steady_clock::now();
                    if (texSeen.size() < 400 && texSeen.insert(tk ^ (static_cast<uint64_t>(std::chrono::duration<double>(std::chrono::steady_clock::now() - tt0).count() / 60.0) << 56)).second)
                        std::fprintf(stderr, "[fonttex] t=%.1f prim=%u psm=0x%x tbp=0x%x tbw=%u tw=%u th=%u cbp=0x%x cpsm=0x%x csa=%u tcc=%u tfx=%u fst=%d | frame fbp=%u\n",
                                     std::chrono::duration<double>(std::chrono::steady_clock::now() - tt0).count(),
                                     static_cast<unsigned>(m_prim.type), c.tex0.psm, c.tex0.tbp0, c.tex0.tbw, c.tex0.tw, c.tex0.th, c.tex0.cbp, c.tex0.cpsm, c.tex0.csa,
                                     c.tex0.tcc, c.tex0.tfx, m_prim.fst ? 1 : 0, c.frame.fbp);
                }
                const bool hitF = c.frame.fbmsk != 0xFFFFFFFFu && fb0 < 0x2340u && fb1 > 0x2240u;
                const bool hitZ = zWrites && z0 < 0x2340u && z1 > 0x2240u;
                if (hitF || hitZ)
                {
                    static std::set<uint64_t> seen;
                    const uint64_t key = (static_cast<uint64_t>(c.frame.fbp) << 40) ^ (static_cast<uint64_t>(c.zbuf.zbp) << 20) ^
                                         (static_cast<uint64_t>(lines) << 4) ^ (hitF ? 1u : 0u) ^ (hitZ ? 2u : 0u) ^
                                         (static_cast<uint64_t>(c.frame.fbw) << 60);
                    if (seen.size() < 40 && seen.insert(key).second)
                        std::fprintf(stderr, "[fontclob] %s%s frame fbp=%u fbw=%u psm=0x%x msk=%08x | zbuf zbp=%u psm=0x%x zte=%d zmsk=%d | scissor %u-%u x %u-%u prim=%u\n",
                                     hitF ? "FRAME" : "", hitZ ? "ZBUF" : "", c.frame.fbp, c.frame.fbw, c.frame.psm,
                                     c.frame.fbmsk, c.zbuf.zbp, c.zbuf.psm, zWrites ? 1 : 0, c.zbuf.zmask ? 1 : 0,
                                     c.scissor.x0, c.scissor.x1, c.scissor.y0, c.scissor.y1, static_cast<unsigned>(m_prim.type));
                }
            }
        }
        if (s_hudLog.phase != HudLog::Off)
            hudLogPrimitive(batch);
        if (m_packetData && m_overlayRule && m_overlayRule(batch.state))
            recordOverlayPrimitive(batch, needed);
        updatePreferredDisplaySourceForDraw(batch);
        {
            ps2_host_profile::Scope drawProfile(ps2_host_profile::GsSubmit);
            m_backend->Submit(batch);
        }
        m_vramWrittenSincePresent = true;
        recordDrawDebugEventUnlocked(needed);
    }

    switch (m_prim.type)
    {
    case GS_PRIM_LINE:
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_SPRITE:
    case GS_PRIM_POINT:
        m_vtxCount = 0;
        break;
    case GS_PRIM_LINESTRIP:
        m_vtxQueue[0] = m_vtxQueue[1];
        m_vtxSource[0] = m_vtxSource[1];
        m_vtxCount = 1;
        break;
    case GS_PRIM_TRISTRIP:
        m_vtxQueue[0] = m_vtxQueue[1];
        m_vtxQueue[1] = m_vtxQueue[2];
        m_vtxSource[0] = m_vtxSource[1];
        m_vtxSource[1] = m_vtxSource[2];
        m_vtxCount = 2;
        break;
    case GS_PRIM_TRIFAN:
        m_vtxQueue[1] = m_vtxQueue[2];
        m_vtxSource[1] = m_vtxSource[2];
        m_vtxCount = 2;
        break;
    default:
        m_vtxCount = 0;
        break;
    }
}

// --- widescreen overlay (setWidescreenOverlayRule) ---------------------------------------------------------------

bool GS::beginOverlayPacket(const uint8_t *data)
{
    const bool on = m_overlayRule != nullptr && ps2_widescreen::wideNow();
    if (!on)
    {
        if (m_holding)
            flushOverlayUnlocked();
        m_packetData = nullptr;
        return false;
    }
    ++m_packetSerial;
    m_packetData = data;
    return true;
}

void GS::forwardGpuGif(uint8_t path, const uint8_t *data, uint32_t sizeBytes)
{
    if (!m_gpu)
        return;
    if (!m_holding)
    {
        m_gpu->gif(path, data, sizeBytes);
        return;
    }
    HeldGpuData held;
    held.packet = data == m_packetData ? m_packetSerial : ~0ull; // ~0: a packet of our own (never patched)
    held.path = path;
    held.bytes.assign(data, data + sizeBytes);
    m_heldBytes += sizeBytes;
    m_held.push_back(std::move(held));
    if (m_heldBytes > (16u << 20)) // a 2-D layer is small: something kept drawing without a picture
        flushOverlayUnlocked();
}

void GS::forwardGpuReg(uint8_t regAddr, uint64_t value)
{
    if (!m_holding)
    {
        m_gpu->reg(regAddr, value);
        return;
    }
    HeldGpuData held;
    held.regAddr = regAddr;
    held.regValue = value;
    m_held.push_back(std::move(held));
}

void GS::recordOverlayPrimitive(const GSPrimitiveBatch &batch, int vertexCount)
{
    const GSContext &ctx = batch.state.context;
    OverlayPrimitive prim;
    prim.ofx = static_cast<float>(ctx.xyoffset.ofx) / 16.0f;
    const float ofy = static_cast<float>(ctx.xyoffset.ofy) / 16.0f;
    prim.width = static_cast<float>(std::max<uint32_t>(1u, ctx.frame.fbw) * 64u);
    prim.minX = prim.minY = 1e9f;
    prim.maxX = prim.maxY = -1e9f;
    prim.count = static_cast<uint8_t>(std::min(vertexCount, 3));
    for (uint8_t i = 0; i < prim.count; ++i)
    {
        prim.minX = std::min(prim.minX, batch.vertices[i].x - prim.ofx);
        prim.maxX = std::max(prim.maxX, batch.vertices[i].x - prim.ofx);
        prim.minY = std::min(prim.minY, batch.vertices[i].y - ofy);
        prim.maxY = std::max(prim.maxY, batch.vertices[i].y - ofy);
        prim.vertices[i] = m_vtxSource[i];
    }
    m_overlay.push_back(prim);
    m_holding = true;
}

void GS::flushOverlayUnlocked()
{
    // 1. elements, in drawing order (a later layer sits on top of an earlier one):
    //  - a primitive joins the element drawn just before it if it touches it and isn't much wider (the letters of a
    //    name, a bar's parts - but a menu panel drawn over the bars starts an element of its own);
    //  - then an element drawn inside an earlier, larger one joins it (a menu's text and sliders join its panel), so a
    //    panel and what is on it move together, while the HUD under the panel keeps its own corner.
    struct Element
    {
        float minX, minY, maxX, maxY, width;
        std::vector<size_t> prims;
    };
    std::vector<Element> elements;
    constexpr float kGapX = 8.0f, kGapY = 4.0f, kInside = 2.0f;
    // which third of the screen a box's middle is in (its anchor below)
    const auto third = [](float minX, float maxX, float width)
    {
        const float centre = (minX + maxX) * 0.5f;
        return centre < width / 3.0f ? 0 : centre > width * 2.0f / 3.0f ? 2 : 1;
    };
    // a box at the screen's left or right edge (the bars and portraits): it keeps its corner - it never joins a group
    // of another third, even one drawn over it (a pop-up growing from the middle across the bars, like ROPE BREAK or
    // the match's opening graphic); -1 = not at an edge
    const auto edgeSide = [&](float minX, float maxX, float width)
    {
        const int t = third(minX, maxX, width);
        if (t == 0 && minX <= width * 0.055f)
            return 0;
        if (t == 2 && maxX >= width * 0.945f)
            return 2;
        return -1;
    };
    for (size_t i = 0; i < m_overlay.size(); ++i)
    {
        const OverlayPrimitive &p = m_overlay[i];
        if (p.maxX - p.minX >= 0.97f * p.width)
            continue; // a full-screen layer (a fade, a dimmed background): as it is, and no element grows around it
        if (!elements.empty())
        {
            Element &e = elements.back();
            const bool touches =
                p.minX <= e.maxX + kGapX && p.maxX >= e.minX - kGapX && p.minY <= e.maxY + kGapY && p.maxY >= e.minY - kGapY;
            const bool notMuchWider = (p.maxX - p.minX) <= 2.0f * (e.maxX - e.minX) + 16.0f;
            const int edge = edgeSide(p.minX, p.maxX, p.width);
            const bool keepsCorner = edge >= 0 && third(e.minX, e.maxX, e.width) != edge;
            if (touches && notMuchWider && !keepsCorner)
            {
                e.minX = std::min(e.minX, p.minX);
                e.maxX = std::max(e.maxX, p.maxX);
                e.minY = std::min(e.minY, p.minY);
                e.maxY = std::max(e.maxY, p.maxY);
                e.prims.push_back(i);
                continue;
            }
        }
        elements.push_back({p.minX, p.minY, p.maxX, p.maxY, p.width, {i}});
    }
    for (size_t b = 1; b < elements.size();)
    {
        const Element &y = elements[b];
        const auto inside = [&](const Element &x)
        { return y.minX >= x.minX - kInside && y.maxX <= x.maxX + kInside && y.minY >= x.minY - kInside && y.maxY <= x.maxY + kInside; };
        const auto touches = [&](const Element &x)
        { return y.minX <= x.maxX + kGapX && y.maxX >= x.minX - kGapX && y.minY <= x.maxY + kGapY && y.maxY >= x.minY - kGapY; };
        const auto thirdOf = [&](const Element &x) { return third(x.minX, x.maxX, x.width); };
        const int edge = edgeSide(y.minX, y.maxX, y.width);
        size_t into = b;
        // first a neighbour it touches without lying inside it, in the same third (a score next to its player's bar,
        // under a menu panel - but not the corner bars with a pop-up that grows from the middle up to them, like
        // ROPE BREAK), else the nearest earlier element it lies inside (a menu's text on its panel)
        for (size_t a = b; a-- > 0;)
            if (touches(elements[a]) && !inside(elements[a]) && thirdOf(elements[a]) == thirdOf(y) &&
                y.maxX - y.minX <= 2.0f * (elements[a].maxX - elements[a].minX) + 16.0f)
            {
                into = a;
                break;
            }
        if (into == b)
            for (size_t a = b; a-- > 0;)
                if (inside(elements[a]) && (edge < 0 || thirdOf(elements[a]) == edge))
                {
                    into = a;
                    break;
                }
        if (into == b)
        {
            ++b;
            continue;
        }
        Element &x = elements[into];
        x.minX = std::min(x.minX, y.minX);
        x.maxX = std::max(x.maxX, y.maxX);
        x.minY = std::min(x.minY, y.minY);
        x.maxY = std::max(x.maxY, y.maxY);
        x.prims.insert(x.prims.end(), y.prims.begin(), y.prims.end());
        elements.erase(elements.begin() + static_cast<std::ptrdiff_t>(b));
    }

    // 2. each element squeezed back to its shape around its anchor: x' = A + k (x - A)
    const float target = ps2_widescreen::targetAspect();
    const float k = target > 0.0f ? (4.0f / 3.0f) / target : 1.0f;
    std::unordered_map<uint64_t, size_t> heldIndex;
    for (size_t i = 0; i < m_held.size(); ++i)
        if (m_held[i].packet != 0u && m_held[i].packet != ~0ull)
            heldIndex[m_held[i].packet] = i;
    std::unordered_set<uint64_t> patched;
    uint32_t anchors[3] = {};
    uint32_t unplaced = 0;
    // VENDETTA_WIDEHUDLOG=1: once a second; =2: every frame whose elements differ from the frame before
    static const char *logSwitch = std::getenv("VENDETTA_WIDEHUDLOG");
    static const bool log = logSwitch != nullptr;
    static const bool logChanges = logSwitch && std::string(logSwitch) == "2";
    std::string logText;
    for (const Element &e : elements)
    {
        const float w = e.width;
        const float centre = (e.minX + e.maxX) * 0.5f;

        float anchor = w * 0.5f;
        int kind = 1;
        if (centre < w / 3.0f)
            anchor = 0.0f, kind = 0;
        else if (centre > w * 2.0f / 3.0f)
            anchor = w, kind = 2;
        ++anchors[kind];
        if (log)
        {
            char line[160];
            std::snprintf(line, sizeof(line), " %c(%.0f-%.0f,%.0f-%.0f)x%zu", "LCR"[kind], e.minX, e.maxX, e.minY, e.maxY,
                          e.prims.size());
            logText += line;
        }
        for (size_t pi : e.prims)
        {
            const OverlayPrimitive &p = m_overlay[pi];
            for (uint8_t v = 0; v < p.count; ++v)
            {
                const VertexSource &src = p.vertices[v];
                const auto held = heldIndex.find(src.packet);
                if (src.packet == 0u || held == heldIndex.end() || src.offset + 2u > m_held[held->second].bytes.size())
                {
                    ++unplaced;
                    continue;
                }
                const uint64_t key = (src.packet << 24) ^ src.offset;
                if (!patched.insert(key).second)
                    continue; // a vertex shared by two primitives (a strip): once
                uint8_t *bytes = m_held[held->second].bytes.data() + src.offset;
                const uint16_t x = static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
                const float screenX = static_cast<float>(x) / 16.0f - p.ofx;
                const float placed = anchor + k * (screenX - anchor);
                const long fixed = std::lround((placed + p.ofx) * 16.0f);
                const uint16_t nx = static_cast<uint16_t>(std::clamp<long>(fixed, 0, 0xFFFF));
                bytes[0] = static_cast<uint8_t>(nx & 0xFFu);
                bytes[1] = static_cast<uint8_t>(nx >> 8);
            }
        }
    }
    if (log)
    {
        static auto lastLog = std::chrono::steady_clock::time_point{};
        static std::string lastText;
        const auto now = std::chrono::steady_clock::now();
        const bool changed = logChanges && logText != lastText;
        lastText = logText;
        if (changed || now - lastLog > std::chrono::seconds(1))
        {
            lastLog = now;
            std::fprintf(stderr,
                         "[widescreen] HUD: %zu primitives, %zu elements (left %u, centre %u, right %u), %zu vertices placed, "
                         "%u not in a held packet;%s\n",
                         m_overlay.size(), elements.size(), anchors[0], anchors[1], anchors[2], patched.size(), unplaced,
                         logText.c_str());
        }
    }

    // 3. everything held goes to the graphics card, in order
    std::vector<HeldGpuData> held;
    held.swap(m_held);
    m_holding = false;
    m_heldBytes = 0;
    m_overlay.clear();
    for (const HeldGpuData &h : held)
    {
        if (h.packet == 0u)
            m_gpu->reg(h.regAddr, h.regValue);
        else
            m_gpu->gif(h.path, h.bytes.data(), static_cast<uint32_t>(h.bytes.size()));
    }
}

void GS::processImageData(const uint8_t *data, uint32_t sizeBytes)
{
    ps2_host_profile::Scope profile(ps2_host_profile::GsSubmit);
    if (!m_backend)
        return;
    ps2_host_profile::count(ps2_host_profile::GsUploadBytes, sizeBytes);
    m_backend->UploadImage(data, sizeBytes);
    m_vramWrittenSincePresent = true;
}


bool GS::clearFramebufferContext(uint32_t contextIndex, uint32_t rgba)
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    if (m_gpu)
        warnGpuUnmirrored("a direct frame buffer clear");
    m_vramWrittenSincePresent = true;
    return m_backend && m_backend->ClearFramebuffer(m_ctx[(contextIndex != 0u) ? 1 : 0], rgba);
}

bool GS::clearActiveFramebuffer(uint32_t rgba)
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    if (m_gpu)
        warnGpuUnmirrored("a direct frame buffer clear");
    m_vramWrittenSincePresent = true;
    return m_backend && m_backend->ClearFramebuffer(activeContext(), rgba);
}

uint32_t GS::consumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    ps2_host_profile::count(ps2_host_profile::GsReplyCalls);
    return m_backend ? m_backend->ConsumeLocalToHostBytes(dst, maxBytes) : 0u;
}

void GS::setRasterBackend(std::unique_ptr<GSRasterBackend> backend)
{
    if (!backend)
        backend = std::make_unique<GSCpuBackend>();

    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
    if (m_backend)
    {
        m_backend->Flush();
        m_backend->Sync(GSSyncReason::Reset);

        // The external 4 MiB GS allocation is the backend hand-off format.
        // This keeps hot backend replacement deterministic even when a future
        // GPU backend keeps a private/mirrored local-memory representation.
        if (m_localMemoryStorage && m_localMemorySize != 0u)
        {
            std::vector<uint8_t> localMemory;
            m_backend->SnapshotVram(localMemory);
            const size_t bytes = std::min<size_t>(localMemory.size(), m_localMemorySize);
            if (bytes != 0u)
                std::memcpy(m_localMemoryStorage, localMemory.data(), bytes);
        }
    }

    m_backendRaw.store(backend.get(), std::memory_order_release);
    m_backend = std::move(backend);
    m_backend->Initialize(m_localMemoryStorage, m_localMemorySize);
    m_vramWrittenSincePresent = true;
}

uint32_t GS::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    ps2_host_profile::count(ps2_host_profile::GsReplyCalls);
    return m_backend ? m_backend->ReadVram(psm, base, bw, x, y) : 0u;
}

void GS::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    gateForeignCall();
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    KickGuard kick(*this);
    if (m_gpu)
        warnGpuUnmirrored("a direct GS memory write");
    if (m_backend)
        m_backend->WriteVram(psm, base, bw, x, y, value);
    m_vramWrittenSincePresent = true;
}

GSPrimitiveBatch GS::buildDrawBatch(int vertexCount) const
{
    GSPrimitiveBatch batch{};
    batch.vertexCount = static_cast<uint8_t>(std::min(vertexCount, 3));
    for (int i = 0; i < batch.vertexCount; ++i)
        batch.vertices[static_cast<size_t>(i)] = m_vtxQueue[i];
    batch.state.context = m_ctx[m_prim.ctxt ? 1 : 0];
    batch.state.prim = m_prim;
    batch.state.texa = m_texa;
    batch.state.texclut = m_texclut;
    batch.state.pabe = m_pabe;
    batch.state.scanmsk = m_scanmsk;
    batch.state.dimx = m_dimx;
    batch.state.dthe = m_dthe;
    batch.state.colclamp = m_colclamp;
    batch.state.fogR = m_fogR;
    batch.state.fogG = m_fogG;
    batch.state.fogB = m_fogB;
    batch.state.textureWidth = static_cast<uint16_t>(1u << std::min<uint32_t>(batch.state.context.tex0.tw, 10u));
    batch.state.textureHeight = static_cast<uint16_t>(1u << std::min<uint32_t>(batch.state.context.tex0.th, 10u));
    const uint64_t tex1 = batch.state.context.tex1;
    const uint8_t mmag = static_cast<uint8_t>((tex1 >> 5u) & 0x1u);
    const uint8_t mmin = static_cast<uint8_t>((tex1 >> 6u) & 0x7u);
    batch.state.linearFilter = mmag != 0u || mmin == 1u || (mmin & 0x4u) != 0u;
    return batch;
}

void GS::updatePreferredDisplaySourceForDraw(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSContext &ctx = state.context;
    if (m_hasPreferredDisplaySource && ctx.frame.fbp == m_preferredDisplayDestFbp)
        m_hasPreferredDisplaySource = false;
    if (state.prim.type != GS_PRIM_SPRITE || batch.vertexCount < 2u)
        return;

    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    int x0 = static_cast<int>(v0.x) - (ctx.xyoffset.ofx >> 4);
    int y0 = static_cast<int>(v0.y) - (ctx.xyoffset.ofy >> 4);
    int x1 = static_cast<int>(v1.x) - (ctx.xyoffset.ofx >> 4);
    int y1 = static_cast<int>(v1.y) - (ctx.xyoffset.ofy >> 4);
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
    const int xEnd = x0 + std::max(1, x1 - x0) - 1;
    const int yEnd = y0 + std::max(1, y1 - y0) - 1;
    const uint8_t alphaMode = static_cast<uint8_t>(ctx.alpha & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((ctx.alpha >> 32u) & 0xFFu);
    const bool displayCopy = state.prim.tme && state.prim.abe && state.prim.fst && state.prim.ctxt &&
                             ctx.frame.fbp != ctx.tex0.tbp0 && alphaMode == 0x64u &&
                             (alphaFix == 0x60u || alphaFix == 0x80u) &&
                             x0 <= 0 && y0 <= 0 && xEnd >= 639 && yEnd >= 447;
    if (displayCopy)
    {
        m_preferredDisplaySourceFrame = {ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0u};
        m_preferredDisplayDestFbp = ctx.frame.fbp;
        m_hasPreferredDisplaySource = true;
    }
}
