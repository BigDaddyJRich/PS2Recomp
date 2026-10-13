#include "runtime/ps2_chain.h"

#include "ps2_runtime.h"
#include "runtime/ps2_host_profile.h"
#include "ThreadNaming.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <xmmintrin.h>
#define CHAIN_HAS_MXCSR 1
#else
#define CHAIN_HAS_MXCSR 0
#endif

void ps2HostPreferPerformanceCores(); // Kernel/HostPower.cpp

namespace
{
    thread_local bool t_onChainWorker = false;
    thread_local bool t_chainProducer = false;

    constexpr uint64_t kPublishBytes = 16u * 1024u;
    constexpr uint32_t kPublishProgressRecords = 64u;
    constexpr auto kSpin = std::chrono::microseconds(20);

    constexpr uint64_t align16(uint64_t bytes) { return (bytes + 15u) & ~uint64_t{15u}; }

    struct HeapData
    {
        uint8_t *data;
        uint32_t size;
    };
    struct RegisterWrite
    {
        uint32_t address;
        uint32_t value;
    };
    struct PresentPayload
    {
        GSDisplayRegs regs;
        uint64_t vsyncTick;
    };

    template <typename T>
    T load(const uint8_t *payload)
    {
        T value;
        std::memcpy(&value, payload, sizeof(T));
        return value;
    }

    uint32_t envUnsigned(const char *name, uint32_t fallback)
    {
        const char *value = std::getenv(name);
        return value && *value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : fallback;
    }

    uint32_t readMxcsr() noexcept
    {
#if CHAIN_HAS_MXCSR
        return _mm_getcsr() & ~0x3Fu; // control bits only (the low six are sticky exception flags)
#else
        return 0u;
#endif
    }

    void cpuRelax()
    {
#if CHAIN_HAS_MXCSR
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
}

ChainWorker::ChainWorker(PS2Runtime &runtime) : m_runtime(runtime) {}

ChainWorker::~ChainWorker()
{
    shutdown();
}

bool ChainWorker::onWorker() noexcept { return t_onChainWorker; }

void ChainWorker::allowUnfencedCsrRead(uint32_t pc) noexcept
{
    if (m_exemptCsrCount < 8u)
        m_exemptCsrPcs[m_exemptCsrCount++] = pc;
}

bool ChainWorker::csrReadExempt(uint32_t pc) const noexcept
{
    for (uint32_t index = 0; index < m_exemptCsrCount; ++index)
        if (m_exemptCsrPcs[index] == pc)
            return true;
    return false;
}
void ChainWorker::markProducerThread() noexcept { t_chainProducer = true; }

void ChainWorker::start()
{
    const char *async = std::getenv("VENDETTA_CHAIN_ASYNC");
    m_async = !(async && *async == '0'); // the default; VENDETTA_CHAIN_ASYNC=0 = inline
    if (!m_async)
        return;
    if (const char *exempt = std::getenv("VENDETTA_CHAIN_CSR_EXEMPT"); exempt && *exempt == '0')
        m_exemptCsrCount = 0u;
    m_verifyExempt = envUnsigned("VENDETTA_CHAIN_VERIFY", 0u) != 0u;
    m_sync = envUnsigned("VENDETTA_CHAIN_SYNC", 0u) != 0u;
    m_slowPresentUs = envUnsigned("VENDETTA_CHAIN_SLOW", 0u);
    uint64_t size = 1u << 20u;
    const uint64_t wanted = uint64_t{envUnsigned("VENDETTA_CHAIN_RING_MB", 16u)} << 20u;
    while (size < wanted && size < (uint64_t{1} << 30u))
        size <<= 1u;
    m_size = size;
    m_mask = size - 1u;
    m_ringStorage = std::make_unique<uint8_t[]>(static_cast<size_t>(size) + 64u);
    m_ring = m_ringStorage.get();
    m_ring += (64u - (reinterpret_cast<uintptr_t>(m_ring) & 63u)) & 63u;
    // GS:: calls from any other thread wait for the worker first (it is the GS front end's user now)
    m_runtime.gs().setForeignCallGate([](void *self) { static_cast<ChainWorker *>(self)->fence(ChainFence::GsCall); }, this);
    m_thread = std::thread([this]() { threadMain(); });
}

// ---- What runs a record (on the worker, or at once on the game thread when inline) ------------------------------

void ChainWorker::runGif(const uint8_t *data, uint32_t sizeBytes)
{
    m_runtime.memory().submitGifPacket(GifPathId::Path3, data, sizeBytes, false);
}

void ChainWorker::runVif1(const uint8_t *data, uint32_t sizeBytes, uint32_t fbrst)
{
    m_kickFbrst = fbrst;
    m_runtime.memory().processVIF1Data(data, sizeBytes);
}

void ChainWorker::runPresent(const uint8_t *payload)
{
    const PresentPayload present = load<PresentPayload>(payload);
    m_runtime.gs().presentCaptured(present.regs, present.vsyncTick);
}

void ChainWorker::noteVu1Run(uint32_t vpuStatStopBits)
{
    if (!m_async)
    {
        // inline: VPU_STAT as the callbacks always wrote it (the game thread is inside the kick)
        R5900Context *ctx = m_runtime.currentEeContext();
        ctx->vu0_vpu_stat = (ctx->vu0_vpu_stat & ~0x0600u) | vpuStatStopBits;
        return;
    }
    m_runStopBits = vpuStatStopBits;
    m_runReported = true;
}

uint32_t ChainWorker::currentFbrst() const
{
    return m_runtime.currentEeContext()->vu0_fbrst & 0x0C00u;
}

bool ChainWorker::needsDtSync() const
{
    // D / T bits can stop VU1 and set VPU_STAT bits the game may read: such kicks run synchronously. Otherwise every
    // run leaves VPU_STAT bits 9 / 10 clear, as they already are.
    const R5900Context *ctx = m_runtime.currentEeContext();
    return (ctx->vu0_fbrst & 0x0C00u) != 0u || (ctx->vu0_vpu_stat & 0x0600u) != 0u;
}

// ---- The producer (the game thread) ------------------------------------------------------------------------------

void ChainWorker::submitGif(const uint8_t *data, uint32_t sizeBytes)
{
    if (!m_async)
    {
        runGif(data, sizeBytes);
        return;
    }
    pushData(Cmd::Gif, Cmd::GifHeap, data, sizeBytes, 0u);
}

void ChainWorker::submitVif1(const uint8_t *data, uint32_t sizeBytes)
{
    const uint32_t fbrst = currentFbrst();
    if (!m_async)
    {
        runVif1(data, sizeBytes, fbrst);
        return;
    }
    if (needsDtSync())
    {
        // exact VPU_STAT: let the worker finish, run this piece, wait, then apply what its runs left
        ps2_host_profile::count(ps2_host_profile::ChainDtKicks);
        fence(ChainFence::DtKick);
        m_runReported = false;
        pushData(Cmd::Vif1, Cmd::Vif1Heap, data, sizeBytes, fbrst);
        fence(ChainFence::DtKick);
        if (m_runReported)
        {
            R5900Context *ctx = m_runtime.currentEeContext();
            ctx->vu0_vpu_stat = (ctx->vu0_vpu_stat & ~0x0600u) | m_runStopBits;
        }
        return;
    }
    pushData(Cmd::Vif1, Cmd::Vif1Heap, data, sizeBytes, fbrst);
}

void ChainWorker::drainArbiter()
{
    if (!m_async)
    {
        m_runtime.gifArbiter().drain();
        return;
    }
    reserve(Cmd::DrainArbiter, 0u);
    commit();
    publish(); // the end of a kick: the worker may start
}

void ChainWorker::writeVif1Register(uint32_t address, uint32_t value)
{
    if (!m_async)
    {
        m_runtime.memory().applyVif1RegisterWrite(address, value);
        return;
    }
    const RegisterWrite write{address, value};
    std::memcpy(reserve(Cmd::Vif1Register, sizeof(write)), &write, sizeof(write));
    commit();
}

void ChainWorker::present(const GSDisplayRegs &regs, uint64_t vsyncTick)
{
    const PresentPayload payload{regs, vsyncTick};
    if (!m_async)
    {
        uint8_t bytes[sizeof(payload)];
        std::memcpy(bytes, &payload, sizeof(payload));
        runPresent(bytes);
        return;
    }
    ++m_presentsPushed;
    std::memcpy(reserve(Cmd::Present, sizeof(payload)), &payload, sizeof(payload));
    commit();
    publish();
}

void ChainWorker::pushData(Cmd inlineCmd, Cmd heapCmd, const uint8_t *data, uint32_t sizeBytes, uint32_t aux)
{
    ps2_host_profile::count(ps2_host_profile::ChainBytes, sizeBytes);
    if (sizeBytes <= m_size / 4u)
    {
        uint8_t *dst = reserve(inlineCmd, sizeBytes);
        if (sizeBytes != 0u)
            std::memcpy(dst, data, sizeBytes);
        reinterpret_cast<Header *>(dst)[-1].aux = aux;
        commit();
        return;
    }
    HeapData heap{new uint8_t[sizeBytes], sizeBytes};
    std::memcpy(heap.data, data, sizeBytes);
    uint8_t *dst = reserve(heapCmd, sizeof(heap));
    std::memcpy(dst, &heap, sizeof(heap));
    reinterpret_cast<Header *>(dst)[-1].aux = aux;
    commit();
}

uint8_t *ChainWorker::reserve(Cmd cmd, uint32_t payloadBytes)
{
    if (!t_chainProducer)
        ps2_host_profile::count(ps2_host_profile::ChainOwnerViolations);
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
            ps2_host_profile::count(ps2_host_profile::ChainRingFullWaits);
            // (the worker stops only after the game thread has ended, so this ends through its progress)
            waitProgress("room in the chain ring", [&]()
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
    header->mxcsr = readMxcsr();
    m_pendingRecordBytes = recordBytes;
    return reinterpret_cast<uint8_t *>(header + 1);
}

void ChainWorker::commit()
{
    m_localWrite += m_pendingRecordBytes;
    ps2_host_profile::count(ps2_host_profile::ChainRecords);
    if (m_sync)
        fence(ChainFence::Sync);
    else if (m_localWrite - m_published >= kPublishBytes)
        publish();
}

void ChainWorker::publish()
{
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

void ChainWorker::fence(ChainFence reason)
{
    if (!m_async || t_onChainWorker)
        return;
    if (!t_chainProducer)
    {
        // only the game thread may wait for the worker (another thread would race with it)
        ps2_host_profile::count(ps2_host_profile::ChainOwnerViolations);
        return;
    }
    publish();
    if (m_readPos.load(std::memory_order_acquire) >= m_localWrite)
        return;
    ps2_host_profile::count(ps2_host_profile::ChainFences);
    ps2_host_profile::count(reason == ChainFence::Csr ? ps2_host_profile::ChainFenceCsr : ps2_host_profile::ChainFenceOther);
    waitProgress("the chain worker to finish", [&]()
                 { return m_stop.load(std::memory_order_acquire) || m_readPos.load(std::memory_order_acquire) >= m_localWrite; });
}

bool ChainWorker::waitBacklog(uint32_t maxAhead, const std::atomic<bool> &stop)
{
    if (!m_async)
        return true;
    auto caughtUp = [&]()
    { return m_presentsPushed - m_presentsDone.load(std::memory_order_acquire) <= maxAhead; };
    if (caughtUp())
        return true;
    ps2_host_profile::count(ps2_host_profile::ChainThrottleWaits);
    waitProgress("the chain worker to catch up", [&]()
                 { return stop.load(std::memory_order_acquire) || m_stop.load(std::memory_order_acquire) || caughtUp(); });
    return !stop.load(std::memory_order_acquire);
}

template <typename Pred>
void ChainWorker::waitProgress(const char *what, Pred pred)
{
    ps2_host_profile::Scope profile(ps2_host_profile::ChainWait);
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
            std::fprintf(stderr, "[chain] the game thread has waited over 2 s for %s: read=%llu write=%llu record=%u%c", what,
                         static_cast<unsigned long long>(m_readPos.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(m_localWrite), m_currentCmd.load(std::memory_order_relaxed), 10);
        }
    }
    m_producerWaiting.fetch_sub(1u, std::memory_order_seq_cst);
}

void ChainWorker::wakeWaiters()
{
    {
        std::lock_guard<std::mutex> lock(m_progressMutex);
    }
    m_progressCv.notify_all();
}

void ChainWorker::shutdown()
{
    if (m_shutDown)
        return;
    m_shutDown = true;
    if (!m_thread.joinable())
        return;
    m_stop.store(true, std::memory_order_seq_cst);
    m_wakeEpoch.fetch_add(1u, std::memory_order_seq_cst);
    m_wakeEpoch.notify_one();
    m_thread.join();
    // records the worker did not get to: free their heap data
    for (uint64_t pos = m_readPos.load(std::memory_order_acquire); pos < m_localWrite;)
    {
        const Header *header = reinterpret_cast<const Header *>(m_ring + (pos & m_mask));
        const Cmd cmd = static_cast<Cmd>(header->cmd);
        if (cmd == Cmd::GifHeap || cmd == Cmd::Vif1Heap)
            delete[] load<HeapData>(reinterpret_cast<const uint8_t *>(header + 1)).data;
        pos += sizeof(Header) + align16(header->bytes);
    }
    m_readPos.store(m_localWrite, std::memory_order_release);
    m_runtime.gs().setForeignCallGate(nullptr, nullptr);
    wakeWaiters();
}

// ---- The worker ----------------------------------------------------------------------------------------------------

void ChainWorker::publishProgress(uint64_t readPos)
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

void ChainWorker::threadMain()
{
    ThreadNaming::SetCurrentThreadName("ChainThread");
    ps2HostPreferPerformanceCores();
#if CHAIN_HAS_MXCSR
    _mm_setcsr(0x1F80u);
#endif
    t_onChainWorker = true;
    ps2_host_profile::recordThisThread(ps2_host_profile::ChainTrack);
    ps2_host_profile::setPhase(ps2_host_profile::Idle);

    uint64_t read = m_readPos.load(std::memory_order_relaxed);
    for (;;)
    {
        const uint64_t write = m_writePos.load(std::memory_order_acquire);
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

        ps2_host_profile::setPhase(ps2_host_profile::ChainCmd);
        uint32_t sinceProgress = 0u;
        uint64_t localRead = read;
        while (localRead != write)
        {
            if (m_stop.load(std::memory_order_relaxed))
            {
                publishProgress(localRead);
                return; // what is left is dropped (shutdown frees its heap data)
            }
            const Header header = *reinterpret_cast<const Header *>(m_ring + (localRead & m_mask));
            const uint8_t *payload = m_ring + (localRead & m_mask) + sizeof(Header);
            const Cmd cmd = static_cast<Cmd>(header.cmd);
            if (cmd != Cmd::Wrap)
            {
                m_currentCmd.store(header.cmd, std::memory_order_relaxed);
#if CHAIN_HAS_MXCSR
                if (header.mxcsr != readMxcsr())
                {
                    ps2_host_profile::count(ps2_host_profile::ChainKickMxcsr);
                    _mm_setcsr(header.mxcsr);
                }
#endif
                try
                {
                    execute(header, payload);
                }
                catch (const std::exception &e)
                {
                    ps2_host_profile::count(ps2_host_profile::ChainExceptions);
                    if (!m_reportedException)
                    {
                        m_reportedException = true;
                        std::fprintf(stderr, "[chain] record %u failed: %s (counted in chain-exceptions; reported once)%c", header.cmd, e.what(), 10);
                    }
                }
                catch (...)
                {
                    ps2_host_profile::count(ps2_host_profile::ChainExceptions);
                }
            }
            localRead += sizeof(Header) + align16(header.bytes);
            const bool reply = cmd == Cmd::Present || cmd == Cmd::Wrap || cmd == Cmd::DrainArbiter;
            if (reply || ++sinceProgress >= kPublishProgressRecords)
            {
                publishProgress(localRead);
                sinceProgress = 0u;
            }
        }
        read = localRead;
    }
}

void ChainWorker::execute(const Header &header, const uint8_t *payload)
{
    switch (static_cast<Cmd>(header.cmd))
    {
    case Cmd::Wrap:
        break;
    case Cmd::Gif:
        runGif(payload, header.bytes);
        break;
    case Cmd::GifHeap:
    {
        const HeapData heap = load<HeapData>(payload);
        std::unique_ptr<uint8_t[]> owned(heap.data);
        runGif(heap.data, heap.size);
        break;
    }
    case Cmd::Vif1:
        runVif1(payload, header.bytes, header.aux);
        break;
    case Cmd::Vif1Heap:
    {
        const HeapData heap = load<HeapData>(payload);
        std::unique_ptr<uint8_t[]> owned(heap.data);
        runVif1(heap.data, heap.size, header.aux);
        break;
    }
    case Cmd::DrainArbiter:
        m_runtime.gifArbiter().drain();
        break;
    case Cmd::Vif1Register:
    {
        const RegisterWrite write = load<RegisterWrite>(payload);
        m_runtime.memory().applyVif1RegisterWrite(write.address, write.value);
        break;
    }
    case Cmd::Present:
    {
        struct Done
        {
            std::atomic<uint64_t> &done;
            ~Done() { done.fetch_add(1u, std::memory_order_release); }
        } done{m_presentsDone};
        if (m_slowPresentUs != 0u)
            std::this_thread::sleep_for(std::chrono::microseconds(m_slowPresentUs));
        runPresent(payload);
        break;
    }
    }
}
