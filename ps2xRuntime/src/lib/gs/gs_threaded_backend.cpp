#include "runtime/gs/gs_threaded_backend.h"

#include "runtime/ps2_host_profile.h"
#include "ThreadNaming.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <xmmintrin.h>
#define GS_HAS_MXCSR 1
#else
#include <cfenv>
#define GS_HAS_MXCSR 0
#endif

#if defined(_MSC_VER)
#define GS_NOINLINE __declspec(noinline)
#else
#define GS_NOINLINE __attribute__((noinline))
#endif

void ps2HostPreferPerformanceCores(); // Kernel/HostPower.cpp

// ---- The drawing rounding mode -----------------------------------------------------------------------------------
// MXCSR 0x1F80: round to nearest, all exceptions masked, no flush-to-zero / denormals-are-zero (the compiler's default
// and every new thread's). Only the control bits are compared: the low six bits are sticky exception flags.
namespace
{
#if GS_HAS_MXCSR
    constexpr unsigned kDrawMxcsr = 0x1F80u;
    constexpr unsigned kMxcsrFlags = 0x3Fu;
#endif

    bool producerRoundingIsDefault() noexcept
    {
#if GS_HAS_MXCSR
        return (_mm_getcsr() & ~kMxcsrFlags) == kDrawMxcsr;
#else
        return std::fegetround() == FE_TONEAREST;
#endif
    }
}

GS_NOINLINE unsigned gsEnterDrawFpEnv() noexcept
{
#if GS_HAS_MXCSR
    const unsigned saved = _mm_getcsr();
    if ((saved & ~kMxcsrFlags) != kDrawMxcsr)
    {
        _mm_setcsr(kDrawMxcsr | (saved & kMxcsrFlags));
        ps2_host_profile::count(ps2_host_profile::GsFpEnvFixups);
    }
    return saved;
#else
    const int saved = std::fegetround();
    if (saved != FE_TONEAREST)
    {
        std::fesetround(FE_TONEAREST);
        ps2_host_profile::count(ps2_host_profile::GsFpEnvFixups);
    }
    return static_cast<unsigned>(saved);
#endif
}

GS_NOINLINE void gsLeaveDrawFpEnv(unsigned saved) noexcept
{
#if GS_HAS_MXCSR
    if ((saved & ~kMxcsrFlags) != kDrawMxcsr)
        _mm_setcsr(saved);
#else
    if (static_cast<int>(saved) != FE_TONEAREST)
        std::fesetround(static_cast<int>(saved));
#endif
}

// ---- Records -------------------------------------------------------------------------------------------------------
namespace
{
    constexpr uint64_t kPublishBytes = 16u * 1024u; // hand the GS thread new records at least this often
    constexpr uint32_t kPublishProgressCommands = 64u;
    constexpr auto kSpin = std::chrono::microseconds(20);

    constexpr uint64_t align16(uint64_t bytes) { return (bytes + 15u) & ~uint64_t{15u}; }

    struct ClutPayload
    {
        GSTex0Reg tex0;
        GSTexClutReg texclut;
    };
    struct HeapUpload
    {
        uint8_t *data;
        uint32_t size;
    };
    struct WriteVramPayload
    {
        uint32_t psm, base, bw, x, y, value;
    };
    struct ClearPayload
    {
        GSContext context;
        uint32_t rgba;
    };
    struct PresentPayload
    {
        GSPresentationRequest request;
        GSPresentTicket ticket;
        GSPresentMailbox *box;
    };
    struct ResetPresentationPayload
    {
        GSPresentMailbox *box;
        uint64_t seq;
    };
    struct VerifyPayload
    {
        std::array<uint64_t, 512> pages;
        uint64_t check;
    };
    static_assert(std::is_trivially_copyable_v<GSPrimitiveBatch>);
    static_assert(std::is_trivially_copyable_v<ClutPayload>);
    static_assert(std::is_trivially_copyable_v<GSTransferCommand>);
    static_assert(std::is_trivially_copyable_v<HeapUpload>);
    static_assert(std::is_trivially_copyable_v<WriteVramPayload>);
    static_assert(std::is_trivially_copyable_v<ClearPayload>);
    static_assert(std::is_trivially_copyable_v<PresentPayload>);
    static_assert(std::is_trivially_copyable_v<ResetPresentationPayload>);
    static_assert(std::is_trivially_copyable_v<VerifyPayload>);

    template <typename T>
    T load(const uint8_t *payload)
    {
        T value;
        std::memcpy(&value, payload, sizeof(T));
        return value;
    }

    // VENDETTA_GS_VERIFY: a running hash of the command stream (command + payload bytes)
    uint64_t mixStream(uint64_t hash, uint16_t cmd, const uint8_t *data, size_t size)
    {
        hash = (hash ^ cmd) * 0x9E3779B97F4A7C15ull;
        size_t offset = 0;
        for (; offset + 8u <= size; offset += 8u)
        {
            uint64_t word;
            std::memcpy(&word, data + offset, sizeof(word));
            hash = (hash ^ word) * 0x9E3779B97F4A7C15ull;
            hash ^= hash >> 29u;
        }
        for (; offset < size; ++offset)
            hash = (hash ^ data[offset]) * 0x100000001B3ull;
        return hash;
    }

    void cpuRelax()
    {
#if GS_HAS_MXCSR
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }

    uint32_t envUnsigned(const char *name, uint32_t fallback)
    {
        const char *value = std::getenv(name);
        return value && *value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : fallback;
    }
}

// ---- GSThreadedBackend: producer side ------------------------------------------------------------------------------

GSThreadedBackend::GSThreadedBackend(std::unique_ptr<GSCpuBackend> inner, const Options &options)
    : m_inner(std::move(inner)), m_options(options)
{
    uint64_t size = 1u << 20u;
    while (size < options.ringBytes && size < (uint64_t{1} << 30u))
        size <<= 1u;
    m_size = size;
    m_mask = size - 1u;
    m_ringStorage = std::make_unique<uint8_t[]>(static_cast<size_t>(size) + 64u);
    m_ring = m_ringStorage.get();
    m_ring += (64u - (reinterpret_cast<uintptr_t>(m_ring) & 63u)) & 63u;
    // Started here: the front end's constructor already resets the renderer, which waits for this thread.
    m_thread = std::thread([this]() { threadMain(); });
}

GSThreadedBackend::~GSThreadedBackend()
{
    Shutdown();
}

uint8_t *GSThreadedBackend::reserve(Cmd cmd, uint32_t payloadBytes)
{
    const uint64_t recordBytes = sizeof(Header) + align16(payloadBytes);
    const uint64_t offset = m_localWrite & m_mask;
    const uint64_t tail = m_size - offset;
    const bool wrap = recordBytes > tail;
    const uint64_t needed = recordBytes + (wrap ? tail : 0u);
    if (m_size - (m_localWrite - m_cachedRead) < needed)
    {
        m_cachedRead = m_readPos.load(std::memory_order_acquire);
        if (m_size - (m_localWrite - m_cachedRead) < needed)
        {
            publish();
            ps2_host_profile::count(ps2_host_profile::GsRingFullWaits);
            waitProgress("room in the ring", [&]()
                         {
                             m_cachedRead = m_readPos.load(std::memory_order_acquire);
                             return m_size - (m_localWrite - m_cachedRead) >= needed; });
        }
    }
    if (wrap)
    {
        Header *filler = reinterpret_cast<Header *>(m_ring + offset);
        *filler = Header{};
        filler->cmd = static_cast<uint16_t>(Cmd::Wrap);
        filler->bytes = static_cast<uint32_t>(tail - sizeof(Header));
        m_localWrite += tail;
    }
    Header *header = reinterpret_cast<Header *>(m_ring + (m_localWrite & m_mask));
    *header = Header{};
    header->cmd = static_cast<uint16_t>(cmd);
    header->bytes = payloadBytes;
    m_pendingRecordBytes = recordBytes;
    if (m_options.verifyStream && !producerRoundingIsDefault())
        ps2_host_profile::count(ps2_host_profile::GsProducerMxcsr);
    return reinterpret_cast<uint8_t *>(header + 1);
}

void GSThreadedBackend::commit()
{
    if (m_options.verifyStream)
    {
        const Header *header = reinterpret_cast<const Header *>(m_ring + (m_localWrite & m_mask));
        if (header->cmd != static_cast<uint16_t>(Cmd::Verify))
            m_producerStream = mixStream(m_producerStream, header->cmd, reinterpret_cast<const uint8_t *>(header + 1), header->bytes);
    }
    m_localWrite += m_pendingRecordBytes;
    ++m_pendingCmds;
    m_pendingBytes += m_pendingRecordBytes;
    if (m_options.sync)
        drain();
    else if (m_localWrite - m_published >= kPublishBytes)
        publish();
}

template <typename T>
void GSThreadedBackend::push(Cmd cmd, const T &payload)
{
    static_assert(std::is_trivially_copyable_v<T>);
    uint8_t *dst = reserve(cmd, sizeof(T));
    std::memcpy(dst, &payload, sizeof(T));
    commit();
}

void GSThreadedBackend::pushEmpty(Cmd cmd)
{
    reserve(cmd, 0u);
    commit();
}

void GSThreadedBackend::publish()
{
    if (m_pendingCmds != 0u)
    {
        ps2_host_profile::count(ps2_host_profile::GsCmds, m_pendingCmds);
        ps2_host_profile::count(ps2_host_profile::GsRingBytes, m_pendingBytes);
        m_pendingCmds = 0u;
        m_pendingBytes = 0u;
    }
    if (m_published == m_localWrite)
        return;
    m_published = m_localWrite;
    m_writePos.store(m_localWrite, std::memory_order_seq_cst);
    if (m_consumerSleeping.load(std::memory_order_seq_cst) != 0u)
    {
        m_wakeEpoch.fetch_add(1u, std::memory_order_seq_cst);
        m_wakeEpoch.notify_one();
    }
}

void GSThreadedBackend::drain()
{
    publish();
    ps2_host_profile::count(ps2_host_profile::GsDrains);
    if (m_readPos.load(std::memory_order_acquire) >= m_localWrite)
        return;
    waitProgress("the GS thread to finish its commands", [&]()
                 { return m_readPos.load(std::memory_order_acquire) >= m_localWrite; });
}

template <typename Pred>
void GSThreadedBackend::waitProgress(const char *what, Pred pred)
{
    ps2_host_profile::Scope profile(ps2_host_profile::GsWait);
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < kSpin)
    {
        if (pred())
            return;
        cpuRelax();
    }
    bool reported = false;
    std::unique_lock<std::mutex> lock(m_progressMutex);
    m_producerWaiting.fetch_add(1u, std::memory_order_seq_cst);
    while (!pred())
    {
        if (m_progressCv.wait_for(lock, std::chrono::milliseconds(100)) == std::cv_status::timeout && !reported &&
            std::chrono::steady_clock::now() - start > std::chrono::seconds(2))
        {
            reported = true;
            std::fprintf(stderr, "[gs-thread] the game thread has waited over 2 s for %s: read=%llu write=%llu command=%u%c", what,
                         static_cast<unsigned long long>(m_readPos.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(m_localWrite), m_currentCmd.load(std::memory_order_relaxed), 10);
        }
    }
    m_producerWaiting.fetch_sub(1u, std::memory_order_seq_cst);
}

void GSThreadedBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    drain();
    m_inner->Initialize(vram, vramSize);
    m_hasVram = vram != nullptr;
}

void GSThreadedBackend::Reset() { pushEmpty(Cmd::Reset); }
void GSThreadedBackend::Submit(const GSPrimitiveBatch &batch) { push(Cmd::Submit, batch); }
void GSThreadedBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) { push(Cmd::LoadClut, ClutPayload{tex0, texclut}); }
void GSThreadedBackend::BeginTransfer(const GSTransferCommand &command) { push(Cmd::BeginTransfer, command); }

void GSThreadedBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    // One record per call: GSCpuBackend::UploadImage drops a pixel split across two calls, so the split must stay
    // where the caller made it.
    if (!data)
        sizeBytes = 0u;
    const uint32_t inlineLimit = static_cast<uint32_t>(std::min<uint64_t>(1u << 20u, m_size / 4u));
    if (sizeBytes <= inlineLimit)
    {
        uint8_t *dst = reserve(Cmd::UploadInline, sizeBytes);
        if (sizeBytes != 0u)
            std::memcpy(dst, data, sizeBytes);
        commit();
        return;
    }
    HeapUpload upload{new uint8_t[sizeBytes], sizeBytes};
    std::memcpy(upload.data, data, sizeBytes);
    if (m_options.verifyStream)
        m_producerStream = mixStream(m_producerStream, 0xFFFFu, upload.data, sizeBytes);
    push(Cmd::UploadHeap, upload);
}

void GSThreadedBackend::Flush() { pushEmpty(Cmd::Flush); }
void GSThreadedBackend::TextureFlush() { pushEmpty(Cmd::TextureFlush); }

void GSThreadedBackend::Sync(GSSyncReason reason)
{
    switch (reason)
    {
    case GSSyncReason::Reset:
    case GSSyncReason::DebugReadback:
    case GSSyncReason::LocalToHost:
        drain();
        break;
    case GSSyncReason::Finish:       // FINISH: CSR is set by the front end; nothing the game reads waits for drawing
    case GSSyncReason::Presentation: // presentations are commands in the order
        break;
    }
}

PresentationFrame GSThreadedBackend::Present(const GSPresentationRequest &request)
{
    drain();
    return m_inner->Present(request);
}

bool GSThreadedBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    push(Cmd::Clear, ClearPayload{context, rgba});
    return GSCpuBackend::ClearFramebufferWritesPixels(context, m_hasVram);
}

uint32_t GSThreadedBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    drain();
    return m_inner->ConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GSThreadedBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    drain();
    return m_inner->ReadVram(psm, base, bw, x, y);
}

void GSThreadedBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    push(Cmd::WriteVram, WriteVramPayload{psm, base, bw, x, y, value});
}

void GSThreadedBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    drain();
    m_inner->SnapshotVram(out);
}

GSTransferSnapshot GSThreadedBackend::GetTransferSnapshot() const
{
    // debug only: the transfer state as far as the GS thread has got (GSCpuBackend locks its own state)
    return m_inner->GetTransferSnapshot();
}

void GSThreadedBackend::PresentToMailbox(const GSPresentationRequest &request, GSPresentTicket ticket, GSPresentMailbox &box)
{
    if (ticket.flip)
        m_flipsIssued.fetch_add(1u, std::memory_order_relaxed);
    push(Cmd::Present, PresentPayload{request, ticket, &box});
    publish();
}

void GSThreadedBackend::ResetPresentation(GSPresentMailbox &box, uint64_t seq)
{
    push(Cmd::ResetPresentation, ResetPresentationPayload{&box, seq});
    publish();
}

void GSThreadedBackend::VerifyPages(const std::array<uint64_t, 512> &expected, uint64_t check, uint64_t)
{
    VerifyPayload payload{expected, check};
    uint8_t *dst = reserve(Cmd::Verify, sizeof(payload));
    std::memcpy(dst, &payload, sizeof(payload));
    reinterpret_cast<Header *>(dst)[-1].aux = m_producerStream;
    commit();
    publish();
}

void GSThreadedBackend::Kick()
{
    if (m_options.sync)
        drain();
    else
        publish();
}

bool GSThreadedBackend::WaitFlipBacklog(uint32_t maxAhead, const std::atomic<bool> &stop)
{
    auto caughtUp = [&]()
    {
        return m_flipsIssued.load(std::memory_order_relaxed) - m_flipsDone.load(std::memory_order_acquire) <= maxAhead;
    };
    if (caughtUp())
        return true;
    ps2_host_profile::count(ps2_host_profile::GsThrottleWaits);
    waitProgress("the GS thread to draw the frames it was given", [&]()
                 { return stop.load(std::memory_order_acquire) || caughtUp(); });
    return !stop.load(std::memory_order_acquire);
}

void GSThreadedBackend::WakeWaiters()
{
    {
        std::lock_guard<std::mutex> lock(m_progressMutex);
    }
    m_progressCv.notify_all();
}

void GSThreadedBackend::Shutdown()
{
    if (m_shutDown)
        return;
    m_shutDown = true;
    m_stop.store(true, std::memory_order_seq_cst);
    m_wakeEpoch.fetch_add(1u, std::memory_order_seq_cst);
    m_wakeEpoch.notify_one();
    if (m_thread.joinable())
        m_thread.join();
    // commands the GS thread did not get to: free their heap uploads
    for (uint64_t pos = m_readPos.load(std::memory_order_acquire); pos < m_localWrite;)
    {
        const Header *header = reinterpret_cast<const Header *>(m_ring + (pos & m_mask));
        if (header->cmd == static_cast<uint16_t>(Cmd::UploadHeap))
            delete[] load<HeapUpload>(reinterpret_cast<const uint8_t *>(header + 1)).data;
        pos += sizeof(Header) + align16(header->bytes);
    }
    m_readPos.store(m_localWrite, std::memory_order_release);
    WakeWaiters();
}

// ---- GSThreadedBackend: the GS thread ------------------------------------------------------------------------------

void GSThreadedBackend::publishProgress(uint64_t readPos)
{
    m_readPos.store(readPos, std::memory_order_seq_cst);
    if (m_producerWaiting.load(std::memory_order_seq_cst) != 0u)
    {
        {
            std::lock_guard<std::mutex> lock(m_progressMutex);
        }
        m_progressCv.notify_all();
    }
}

void GSThreadedBackend::threadMain()
{
    ThreadNaming::SetCurrentThreadName("GSThread");
    ps2HostPreferPerformanceCores();
#if GS_HAS_MXCSR
    _mm_setcsr(kDrawMxcsr);
#else
    std::fesetround(FE_TONEAREST);
#endif
    ps2_host_profile::recordThisThread(ps2_host_profile::GsTrack);
    ps2_host_profile::setPhase(ps2_host_profile::Idle);

    uint64_t read = m_readPos.load(std::memory_order_relaxed);
    for (;;)
    {
        uint64_t write = m_writePos.load(std::memory_order_acquire);
        if (read == write)
        {
            ps2_host_profile::setPhase(ps2_host_profile::Idle);
            publishProgress(read);
            if (m_stop.load(std::memory_order_acquire))
                return;
            const auto spinStart = std::chrono::steady_clock::now();
            while (m_writePos.load(std::memory_order_acquire) == read && !m_stop.load(std::memory_order_relaxed) &&
                   std::chrono::steady_clock::now() - spinStart < kSpin)
                cpuRelax();
            if (m_writePos.load(std::memory_order_acquire) == read && !m_stop.load(std::memory_order_relaxed))
            {
                m_consumerSleeping.store(1u, std::memory_order_seq_cst);
                const uint32_t epoch = m_wakeEpoch.load(std::memory_order_seq_cst);
                if (m_writePos.load(std::memory_order_seq_cst) == read && !m_stop.load(std::memory_order_seq_cst))
                    m_wakeEpoch.wait(epoch, std::memory_order_seq_cst);
                m_consumerSleeping.store(0u, std::memory_order_relaxed);
            }
            continue;
        }

        ps2_host_profile::setPhase(ps2_host_profile::GsCmd);
        uint32_t sinceProgress = 0u;
        while (read != write)
        {
            if (m_stop.load(std::memory_order_relaxed))
            {
                publishProgress(read);
                return; // what is left is dropped (Shutdown frees its heap uploads)
            }
            const Header header = *reinterpret_cast<const Header *>(m_ring + (read & m_mask));
            const uint8_t *payload = m_ring + (read & m_mask) + sizeof(Header);
            const Cmd cmd = static_cast<Cmd>(header.cmd);
            if (cmd != Cmd::Wrap)
            {
                m_currentCmd.store(header.cmd, std::memory_order_relaxed);
                try
                {
                    execute(header, payload);
                }
                catch (const std::exception &e)
                {
                    ps2_host_profile::count(ps2_host_profile::GsThreadExceptions);
                    if (!m_reportedException)
                    {
                        m_reportedException = true;
                        std::fprintf(stderr, "[gs-thread] command %u failed: %s (counted in gs-thread-exceptions; reported once)%c", header.cmd, e.what(), 10);
                    }
                }
                catch (...)
                {
                    ps2_host_profile::count(ps2_host_profile::GsThreadExceptions);
                }
            }
            read += sizeof(Header) + align16(header.bytes);
            const bool reply = cmd == Cmd::Present || cmd == Cmd::Verify || cmd == Cmd::Wrap || cmd == Cmd::ResetPresentation;
            if (reply || ++sinceProgress >= kPublishProgressCommands)
            {
                publishProgress(read);
                sinceProgress = 0u;
            }
        }
    }
}

void GSThreadedBackend::execute(const Header &header, const uint8_t *payload)
{
    const Cmd cmd = static_cast<Cmd>(header.cmd);
    if (m_options.verifyStream && cmd != Cmd::Verify)
    {
        if (cmd == Cmd::UploadHeap)
        {
            const HeapUpload upload = load<HeapUpload>(payload);
            m_consumerStream = mixStream(m_consumerStream, 0xFFFFu, upload.data, upload.size);
        }
        m_consumerStream = mixStream(m_consumerStream, header.cmd, payload, header.bytes);
    }
    switch (cmd)
    {
    case Cmd::Wrap:
        break;
    case Cmd::Submit:
        m_inner->Submit(load<GSPrimitiveBatch>(payload));
        break;
    case Cmd::LoadClut:
    {
        const ClutPayload clut = load<ClutPayload>(payload);
        m_inner->LoadClut(clut.tex0, clut.texclut);
        break;
    }
    case Cmd::BeginTransfer:
        m_inner->BeginTransfer(load<GSTransferCommand>(payload));
        break;
    case Cmd::UploadInline:
        m_inner->UploadImage(header.bytes != 0u ? payload : nullptr, header.bytes);
        break;
    case Cmd::UploadHeap:
    {
        const HeapUpload upload = load<HeapUpload>(payload);
        std::unique_ptr<uint8_t[]> owned(upload.data);
        m_inner->UploadImage(upload.data, upload.size);
        break;
    }
    case Cmd::Flush:
        m_inner->Flush();
        break;
    case Cmd::TextureFlush:
        m_inner->TextureFlush();
        break;
    case Cmd::WriteVram:
    {
        const WriteVramPayload write = load<WriteVramPayload>(payload);
        m_inner->WriteVram(write.psm, write.base, write.bw, write.x, write.y, write.value);
        break;
    }
    case Cmd::Clear:
    {
        const ClearPayload clear = load<ClearPayload>(payload);
        (void)m_inner->ClearFramebuffer(clear.context, clear.rgba);
        break;
    }
    case Cmd::Reset:
        m_inner->Reset();
        break;
    case Cmd::Present:
    {
        const PresentPayload present = load<PresentPayload>(payload);
        struct FlipDone
        {
            std::atomic<uint64_t> &done;
            bool flip;
            ~FlipDone()
            {
                if (flip)
                    done.fetch_add(1u, std::memory_order_release);
            }
        } flipDone{m_flipsDone, present.ticket.flip};
        if (m_options.slowPresentUs != 0u)
            std::this_thread::sleep_for(std::chrono::microseconds(m_options.slowPresentUs));
        m_inner->PresentToMailbox(present.request, present.ticket, *present.box);
        break;
    }
    case Cmd::ResetPresentation:
    {
        const ResetPresentationPayload reset = load<ResetPresentationPayload>(payload);
        m_inner->ResetPresentation(*reset.box, reset.seq);
        break;
    }
    case Cmd::Verify:
    {
        if (header.aux != m_consumerStream)
        {
            ps2_host_profile::count(ps2_host_profile::GsStreamMismatch);
            static uint32_t reports = 0u;
            if (reports++ < 10u)
                std::fprintf(stderr, "[gs-verify] the GS thread's command stream differs from what the game thread sent (producer %016llx, GS thread %016llx)%c",
                             static_cast<unsigned long long>(header.aux), static_cast<unsigned long long>(m_consumerStream), 10);
            m_consumerStream = header.aux; // report each divergence once
        }
        const VerifyPayload verify = load<VerifyPayload>(payload);
        m_inner->VerifyPages(verify.pages, verify.check, header.aux);
        break;
    }
    }
}

// ---- GSTeeBackend (VENDETTA_GS_VERIFY) -----------------------------------------------------------------------------

GSTeeBackend::GSTeeBackend(std::unique_ptr<GSRasterBackend> primary, uint32_t checkEvery)
    : m_primary(std::move(primary)), m_shadow(std::make_unique<GSCpuBackend>(1u, true)), m_checkEvery(checkEvery)
{
}

void GSTeeBackend::mismatch(const char *what) const
{
    ps2_host_profile::count(ps2_host_profile::GsVerifyMismatch);
    static uint32_t reports = 0u;
    if (reports++ < 20u)
        std::fprintf(stderr, "[gs-verify] %s differs from the 1-band shadow renderer (after %llu checks)%c", what,
                     static_cast<unsigned long long>(m_checks), 10);
}

void GSTeeBackend::called()
{
    if (m_checkEvery != 0u && ++m_sinceCheck >= m_checkEvery)
        check();
}

void GSTeeBackend::check()
{
    m_sinceCheck = 0u;
    std::array<uint64_t, 512> pages{};
    m_shadow->HashVram(&pages);
    m_primary->VerifyPages(pages, ++m_checks, 0u);
}

void GSTeeBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    m_primary->Initialize(vram, vramSize);
    if (vram && vramSize != 0u)
    {
        m_shadowVram.assign(vram, vram + vramSize);
        m_shadow->Initialize(m_shadowVram.data(), vramSize);
    }
    else
        m_shadow->Initialize(nullptr, 0u);
}

void GSTeeBackend::Reset()
{
    m_primary->Reset();
    m_shadow->Reset();
    called();
}

void GSTeeBackend::Submit(const GSPrimitiveBatch &batch)
{
    m_primary->Submit(batch);
    m_shadow->Submit(batch);
    called();
}

void GSTeeBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    m_primary->LoadClut(tex0, texclut);
    m_shadow->LoadClut(tex0, texclut);
    called();
}

void GSTeeBackend::BeginTransfer(const GSTransferCommand &command)
{
    m_primary->BeginTransfer(command);
    m_shadow->BeginTransfer(command);
    called();
}

void GSTeeBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    m_primary->UploadImage(data, sizeBytes);
    m_shadow->UploadImage(data, sizeBytes);
    called();
}

void GSTeeBackend::Flush()
{
    m_primary->Flush();
    m_shadow->Flush();
    called();
}

void GSTeeBackend::TextureFlush()
{
    m_primary->TextureFlush();
    m_shadow->TextureFlush();
    called();
}

void GSTeeBackend::Sync(GSSyncReason reason)
{
    m_primary->Sync(reason);
    m_shadow->Sync(reason);
}

PresentationFrame GSTeeBackend::Present(const GSPresentationRequest &request)
{
    return m_primary->Present(request);
}

bool GSTeeBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    const bool primary = m_primary->ClearFramebuffer(context, rgba);
    if (m_shadow->ClearFramebuffer(context, rgba) != primary)
        mismatch("ClearFramebuffer's result");
    called();
    return primary;
}

uint32_t GSTeeBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    const uint32_t count = m_primary->ConsumeLocalToHostBytes(dst, maxBytes);
    std::vector<uint8_t> shadow(maxBytes);
    const uint32_t shadowCount = m_shadow->ConsumeLocalToHostBytes(shadow.empty() ? nullptr : shadow.data(), maxBytes);
    if (shadowCount != count || (dst && count != 0u && std::memcmp(dst, shadow.data(), count) != 0))
        mismatch("local -> host transfer data");
    return count;
}

uint32_t GSTeeBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    const uint32_t value = m_primary->ReadVram(psm, base, bw, x, y);
    if (m_shadow->ReadVram(psm, base, bw, x, y) != value)
        mismatch("ReadVram");
    return value;
}

void GSTeeBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    m_primary->WriteVram(psm, base, bw, x, y, value);
    m_shadow->WriteVram(psm, base, bw, x, y, value);
    called();
}

void GSTeeBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    m_primary->SnapshotVram(out);
    std::vector<uint8_t> shadow;
    m_shadow->SnapshotVram(shadow);
    if (shadow != out)
        mismatch("SnapshotVram");
}

GSTransferSnapshot GSTeeBackend::GetTransferSnapshot() const
{
    return m_primary->GetTransferSnapshot();
}

void GSTeeBackend::PresentToMailbox(const GSPresentationRequest &request, GSPresentTicket ticket, GSPresentMailbox &box)
{
    m_primary->PresentToMailbox(request, ticket, box);
    check();
}

void GSTeeBackend::ResetPresentation(GSPresentMailbox &box, uint64_t seq) { m_primary->ResetPresentation(box, seq); }
void GSTeeBackend::Kick() { m_primary->Kick(); }
bool GSTeeBackend::WaitFlipBacklog(uint32_t maxAhead, const std::atomic<bool> &stop) { return m_primary->WaitFlipBacklog(maxAhead, stop); }
void GSTeeBackend::WakeWaiters() { m_primary->WakeWaiters(); }
void GSTeeBackend::Shutdown() { m_primary->Shutdown(); }

// ---- The renderer the environment asks for ------------------------------------------------------------------------

std::unique_ptr<GSRasterBackend> makeRasterBackendFromEnvironment()
{
    const char *async = std::getenv("VENDETTA_GS_ASYNC");
    const char *verify = std::getenv("VENDETTA_GS_VERIFY");
    const bool verifying = verify && *verify && *verify != '0';
    std::unique_ptr<GSRasterBackend> backend;
    if (!async || *async != '0') // the default; VENDETTA_GS_ASYNC=0 draws on the game thread
    {
        GSThreadedBackend::Options options;
        options.ringBytes = size_t{envUnsigned("VENDETTA_GS_RING_MB", 8u)} << 20u;
        options.sync = envUnsigned("VENDETTA_GS_SYNC", 0u) != 0u;
        options.slowPresentUs = envUnsigned("VENDETTA_GS_SLOW", 0u);
        options.verifyStream = verifying;
        backend = std::make_unique<GSThreadedBackend>(std::make_unique<GSCpuBackend>(), options);
    }
    else
        backend = std::make_unique<GSCpuBackend>();
    if (verifying)
    {
        // VENDETTA_GS_VERIFY=1 checks at every presentation; =1,N also every N renderer calls
        const char *comma = std::strchr(verify, ',');
        const uint32_t every = comma ? static_cast<uint32_t>(std::strtoul(comma + 1, nullptr, 10)) : 0u;
        backend = std::make_unique<GSTeeBackend>(std::move(backend), every);
    }
    return backend;
}
