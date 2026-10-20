#include <cstdio>
#include "iop_emulator.h"
#include "imports/iop_cdvd.h"
#include "core/iop_cpu.h"
#include "imports/iop_heaplib.h"
#include "imports/iop_imports.h"
#include "imports/iop_intrman.h"
#include "imports/iop_ioman.h"
#include "core/iop_kernel.h"
#include "imports/iop_loadcore.h"
#include "core/iop_memory.h"
#include "services/iop_module_loader.h"
#include "services/iop_rpc.h"
#include "imports/iop_stdio.h"
#include "imports/iop_sysclib.h"
#include "imports/iop_sysmem.h"
#include "imports/iop_timrman.h"
#include "imports/iop_vblank.h"
#include "iop_emulator_const.h"
#include "ps2x/iop/iop_native_hooks.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <unordered_map>
#include <optional>
#include <span>
#include <sstream>
#include <utility>
#include <vector>
#include <mutex>

namespace ps2x::iop
{
    namespace
    {
        // the host-side sound driver's SPU2 writes (spu2QueueWrite) and the read-back snapshot (spu2Snapshot)
        std::mutex g_spu2NativeLock;
        std::vector<std::pair<uint32_t, uint16_t>> g_spu2NativeQueue;
        Spu2Snapshot g_spu2Snapshot;
        uint64_t g_spu2SnapshotCycle = 0;
    }

    void spu2QueueWrite(uint32_t offset, uint16_t value)
    {
        std::lock_guard<std::mutex> lock(g_spu2NativeLock);
        g_spu2NativeQueue.emplace_back(offset, value);
    }

    Spu2Snapshot spu2Snapshot()
    {
        std::lock_guard<std::mutex> lock(g_spu2NativeLock);
        return g_spu2Snapshot;
    }
}

namespace ps2x::iop::detail
{
    // at the start of each IOP time slice: the queued writes, in order; the snapshot every ~1 ms (36,864 IOP cycles)
    static void spu2NativeService(Spu2 &spu2, uint64_t iopCycle)
    {
        std::lock_guard<std::mutex> lock(g_spu2NativeLock);
        for (const auto &w : g_spu2NativeQueue)
            spu2.write16(w.first, w.second);
        g_spu2NativeQueue.clear();
        if (iopCycle - g_spu2SnapshotCycle < 36864u && g_spu2Snapshot.serial != 0)
            return;
        g_spu2SnapshotCycle = iopCycle;
        for (int core = 0; core < 2; ++core)
        {
            const uint32_t base = static_cast<uint32_t>(core) * 0x400u;
            for (int v = 0; v < 24; ++v)
                g_spu2Snapshot.envx[core][v] = spu2.read16(base + static_cast<uint32_t>(v) * 0x10u + 0xAu);
            g_spu2Snapshot.endx[core] = spu2.read16(base + 0x340u) | (static_cast<uint32_t>(spu2.read16(base + 0x342u)) << 16u);
        }
        ++g_spu2Snapshot.serial;
    }
}

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kRamSize = IopMemory::RamSize;
        constexpr uint32_t kKernelHeapBase = IopMemory::HeapBase;
        constexpr uint32_t kKernelHeapLimit = IopMemory::HeapLimit;
        constexpr uint32_t kCallStackBase = kKernelHeapLimit;
        constexpr uint32_t kCallStackLimit = 0x001FFF00u;
        constexpr uint32_t kCallStackSize = 0x2000u;
        constexpr uint32_t kCallStackCapacity = (kCallStackLimit - kCallStackBase) / kCallStackSize;
        constexpr uint64_t kCdvdCompletionCycles = 128u;

        uint32_t physicalAddress(uint32_t address)
        {
            return IopMemory::physicalAddress(address);
        }

        // The native hooks (ps2x/iop/iop_native_hooks.h), and which 4 KB pages of IOP RAM hold one (tested per instruction)
        std::vector<ps2x::iop::NativeHook> &nativeHooks()
        {
            static std::vector<ps2x::iop::NativeHook> hooks;
            return hooks;
        }
        std::array<uint8_t, kRamSize / 4096u> g_nativeHookPages{};

        int32_t sign16(uint32_t value)
        {
            return static_cast<int16_t>(value & 0xFFFFu);
        }

        bool iequals(std::string_view lhs, std::string_view rhs)
        {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
                    std::tolower(static_cast<unsigned char>(rhs[i])))
                    return false;
            }
            return true;
        }

    }

    class IopEmulator::Impl final : public IopGuestExecutor
    {
    public:
        using CpuState = IopCpuState;

        struct Module
        {
            int id = 0;
            std::string path;
            std::string name;
            uint32_t base = 0;
            uint32_t size = 0;
            uint32_t entry = 0;
            uint32_t gp = 0;
            bool resident = false;
        };

        struct GuestCallback
        {
            uint32_t function = 0;
            uint32_t gp = 0;
        };

        struct ScheduledGuestCallback
        {
            uint32_t function = 0u;
            uint32_t gp = 0u;
            uint32_t argument = 0u;
        };

        explicit Impl(IopHost &hostRef)
            : host(hostRef),
              sysmem(host, memory),
              kernel(memory),
              cdvd(host, memory, kernel),
              vblank(kernel),
              rpc(host, memory, kernel),
              sysclib(memory),
              stdio(host, memory),
              heaplib(memory),
              intrman(memory),
              timrman(),
              ioman(memory),
              cpuCore(memory),
              imports(memory),
              loadcore(memory, imports)
        {
            reset();
        }

        void reset()
        {
            memory.reset();
            kernel.reset();
            modules.clear();
            imports.reset();
            rpc.reset();
            cdvd.reset();
            intrman.reset();
            timrman.reset();
            ioman.reset();
            pendingDmaInterrupts.clear();
            pendingGuestCallbacks.clear();
            dmaDueFloor = UINT64_MAX;
            idleKnown = false;
            idleUntil = 0u;
            nextModuleId = 1;
            moduleCursor = kModuleLoadBase;
            totalCycles = 0;
            grantedCycles = 0;
            cyclesAfterLastRun = 0;
            totalInstructions = 0;
            eeCycleCarry = 0;
            activeCpu = nullptr;
            lastError.clear();
            servicingDmaInterrupts = false;
            servicingGuestCallbacks = false;
            callDepth = 0u;
            secrMcCommandHandler = {};
            secrMcDevIdHandler = {};
            checkKelfPathCallback = {};
        }

        uint8_t read8(uint32_t address) const
        {
            return memory.read8(address);
        }

        uint16_t read16(uint32_t address) const
        {
            return memory.read16(address);
        }

        uint32_t read32(uint32_t address) const
        {
            return memory.read32(address);
        }

        void write8(uint32_t address, uint8_t value)
        {
            memory.write8(address, value);
            schedulePendingDma();
        }

        void write16(uint32_t address, uint16_t value)
        {
            memory.write16(address, value);
            schedulePendingDma();
        }

        void write32(uint32_t address, uint32_t value)
        {
            memory.write32(address, value);
            schedulePendingDma();
        }

        void schedulePendingDma()
        {
            if (!memory.hasDmaStart())
                return;
            if (const auto dma = memory.takeDmaStart())
            {
                // A new completion: the floor must stay at or below it, and the idle skip must look again.
                // (Replacing an irq's earlier entry with a later one leaves the floor low; that only costs one
                // walk, which then sets it exactly.)
                pendingDmaInterrupts[dma->irq] = totalCycles + dma->delayCycles;
                dmaDueFloor = std::min(dmaDueFloor, totalCycles + dma->delayCycles);
                idleKnown = false;
                if (!dma->pcm.empty())
                    memory.spu2().queueInput(dma->core, dma->pcm.data(), dma->pcm.size() / 2u);
            }
        }

        bool readRam(uint32_t address, void *destination, size_t size) const
        {
            return memory.readRam(address, destination, size);
        }

        bool writeRam(uint32_t address, const void *source, size_t size)
        {
            return memory.writeRam(address, source, size);
        }

        bool zeroRam(uint32_t address, size_t size)
        {
            return memory.zeroRam(address, size);
        }

        bool isHardwareAddress(uint32_t phys) const
        {
            return memory.isHardwareAddress(phys);
        }

        uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt)
        {
            return memory.allocate(size, alignment, fixed);
        }

        bool freeAllocation(uint32_t address)
        {
            return memory.freeAllocation(address);
        }

        void log(LogLevel level, std::string_view text)
        {
            host.log(level, text);
        }

        bool checkInterrupt(CpuState &cpu)
        {
            const uint32_t status = cpu.cop0[12];
            if ((status & 1u) == 0u)
                return false;
            if ((status & 0x2u) != 0u)
                return false;
            const bool pending = memory.interruptControl() != 0u && (memory.interruptStatus() & memory.interruptMask()) != 0u;
            if (!pending)
                return false;
            cpu.cop0[13] |= 0x400u;
            cpuCore.raiseException(cpu, 0u, cpu.pc, false);
            return true;
        }

        enum class ImportDisposition
        {
            Handled,
            JumpToGuest,
            Missing,
        };

        ImportDisposition dispatchImport(const IopImportCall &call, CpuState &cpu)
        {
            // HCTP: libsd's sceSdInit (libsd:4) clears the handler registrations (transfer-done / SPU2 interrupt
            // callbacks). HCTP's CSL sound server registers its transfer handler (libsd:26) as the EE starts it;
            // on the PS2 that lands after the CRI ADX driver's sceSdInit, here (RPCs handled at once) before it -
            // the registration was wiped, every sound-bank upload stopped after its first 32 KB chunk and in-match
            // hit sounds played from empty sound memory. Registrations made before a later sceSdInit are replayed
            // before the next transfer (libsd:17 / :18). HCTP_SD_NOREPLAY=1 = old behaviour.
            if (iequals(call.library, "libsd") && !sdReplaying)
            {
                if (call.ordinal == 4u || call.ordinal == 21u || call.ordinal == 22u || call.ordinal == 26u || call.ordinal == 27u)
                    std::fprintf(stderr, "[sd-reg] libsd:%u a0=0x%x a1=0x%x a2=0x%x ra=0x%x cyc=%llu%c", static_cast<unsigned>(call.ordinal),
                                 cpu.gpr[4], cpu.gpr[5], cpu.gpr[6], cpu.gpr[31], static_cast<unsigned long long>(totalCycles), 10);
                static const bool noReplay = std::getenv("HCTP_SD_NOREPLAY") != nullptr;
                if (!noReplay)
                {
                    if (call.ordinal == 4u)
                        sdReplayPending = !sdRegistrations.empty();
                    else if (sdReplayPending && (call.ordinal == 17u || call.ordinal == 18u)) // a transfer: sceSdInit (which waits midway) is long done
                    {
                        sdReplayPending = false;
                        sdReplaying = true;
                        for (const auto &kv : sdRegistrations)
                        {
                            const SdRegistration &r = kv.second;
                            (void)callFunction(r.guest, r.a0, r.a1, r.a2, 0u, r.gp);
                            std::fprintf(stderr, "[sd-replay] libsd:%u(0x%x, 0x%x, 0x%x) re-applied after sceSdInit%c",
                                         static_cast<unsigned>(kv.first >> 8), r.a0, r.a1, r.a2, 10);
                        }
                        sdReplaying = false;
                    }
                }
                if (!noReplay && (call.ordinal == 21u || call.ordinal == 22u || call.ordinal == 26u || call.ordinal == 27u))
                {
                    const uint32_t a0 = cpu.gpr[4], a1 = cpu.gpr[5], a2 = cpu.gpr[6], gp = cpu.gpr[28];
                    const uint32_t key = (call.ordinal << 8) | (call.ordinal == 22u || call.ordinal == 27u ? 0u : (a0 & 0xFFu));
                    const ImportDisposition d = dispatchImportTraced(call, cpu);
                    if (d == ImportDisposition::JumpToGuest)
                        sdRegistrations[key] = SdRegistration{cpu.pc, a0, a1, a2, gp};
                    return d;
                }
            }
            return dispatchImportTraced(call, cpu);
        }

        struct SdRegistration
        {
            uint32_t guest, a0, a1, a2, gp;
        };
        std::map<uint32_t, SdRegistration> sdRegistrations;
        bool sdReplayPending = false;
        bool sdReplaying = false;

        ImportDisposition dispatchImportTraced(const IopImportCall &call, CpuState &cpu)
        {
            // VENDETTA_IOPLOG=1: every IOP module call into the kernel / SIF / sound libraries, with its result.
            static const bool trace = std::getenv("VENDETTA_IOPLOG") != nullptr;
            // (Checked before the name compares below, which only matter for the log.)
            if (!trace)
                return dispatchImportUntraced(call, cpu);
            // Left out: string / print helpers and the calls that repeat every few hundred cycles (thread
            // yield / sleep / wake, libsd status polls), which would bury everything else.
            const bool noisy = iequals(call.library, "sysclib") || iequals(call.library, "stdio") ||
                               (iequals(call.library, "thbase") && (call.ordinal >= 24u && call.ordinal <= 26u)) ||
                               (iequals(call.library, "thbase") && (call.ordinal == 34u || call.ordinal == 40u)) ||
                               (iequals(call.library, "libsd") && (call.ordinal == 10u || call.ordinal == 20u));
            if (noisy)
                return dispatchImportUntraced(call, cpu);
            const uint32_t pc = cpu.pc, ra = cpu.gpr[31];
            const uint32_t a0 = cpu.gpr[4], a1 = cpu.gpr[5], a2 = cpu.gpr[6], a3 = cpu.gpr[7];
            const ImportDisposition disposition = dispatchImportUntraced(call, cpu);
            static uint32_t traced = 0u;
            if (traced < 200000u)
            {
                ++traced;
                std::ostringstream out;
                out << "[iop-call] " << call.library << ':' << std::dec << call.ordinal << std::hex
                    << " pc=0x" << pc << " ra=0x" << ra << " a0=0x" << a0 << " a1=0x" << a1
                    << " a2=0x" << a2 << " a3=0x" << a3;
                if (disposition == ImportDisposition::JumpToGuest)
                    out << " -> guest 0x" << cpu.pc;
                else
                    out << " v0=0x" << cpu.gpr[2] << (disposition == ImportDisposition::Missing ? " (missing)" : "");
                out << " cyc=" << std::dec << totalCycles;
                log(LogLevel::Info, out.str());
            }
            return disposition;
        }

        ImportDisposition dispatchImportUntraced(const IopImportCall &call, CpuState &cpu)
        {
            const uint32_t a0 = cpu.gpr[4];
            auto setV0 = [&](uint32_t value)
            {
                cpu.gpr[2] = value;
            };

            if (iequals(call.library, "sysmem") && sysmem.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "cdvdman") && cdvd.dispatchImport(call.ordinal, cpu))
            {
                if (const auto callback = cdvd.takeCompletionCallback())
                {
                    idleKnown = false;
                    pendingGuestCallbacks.emplace(
                        totalCycles + kCdvdCompletionCycles,
                        ScheduledGuestCallback{
                            callback->address,
                            callback->gp,
                            callback->reason,
                        });
                }
                return ImportDisposition::Handled;
            }

            if (iequals(call.library, "loadcore") && loadcore.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "thbase") || iequals(call.library, "threadman"))
            {
                return kernel.dispatchThreadImport(call.ordinal, cpu, totalCycles)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thsemap"))
            {
                return kernel.dispatchSemaphoreImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thevent"))
            {
                return kernel.dispatchEventImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "sifcmd"))
            {
                return rpc.dispatchSifCmdImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "intrman") && intrman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "secrman"))
            {
                switch (call.ordinal)
                {
                case 4: // SecrSetMcCommandHandler
                    secrMcCommandHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                case 5: // SecrSetMcDevIDHandler
                    secrMcDevIdHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                default:
                    break;
                }
            }
            if (iequals(call.library, "modload") && call.ordinal == 13u)
            {
                checkKelfPathCallback = {a0, cpu.gpr[28]};
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "ioman") && ioman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sifman"))
            {
                return rpc.dispatchSifManImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "vblank") && vblank.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "timrman") && timrman.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "dmacman"))
            {
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "stdio") && stdio.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sysclib"))
            {
                return sysclib.dispatchImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "heaplib") && heaplib.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            const uint32_t target = imports.resolve(call.library, call.ordinal, call.version);
            if (target != 0u)
            {
                cpu.pc = target;
                cpu.branchPending = false;
                return ImportDisposition::JumpToGuest;
            }

            std::ostringstream out;
            out << "[IOP] unhandled import " << call.library << ':' << call.ordinal
                << " version=0x" << std::hex << call.version << " pc=0x" << cpu.pc;
            log(LogLevel::Warning, out.str());
            setV0(0);
            return ImportDisposition::Missing;
        }

        // VENDETTA_IOPPROF=1: samples the IOP pc every 64 instructions; every 64M instructions logs the busiest
        // 32-byte blocks (with the share of samples) and clears the counts.
        void profileSample(uint32_t pc)
        {
            static const bool enabled = std::getenv("VENDETTA_IOPPROF") != nullptr;
            if (!enabled || (totalInstructions & 63u) != 0u)
                return;
            ++profileCounts[pc & ~31u];
            if (++profileSamples < (64u * 1024u * 1024u) / 64u)
                return;
            std::vector<std::pair<uint32_t, uint64_t>> top(profileCounts.begin(), profileCounts.end());
            std::sort(top.begin(), top.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
            std::ostringstream out;
            const double hostMs = static_cast<double>(profileHostNs) / 1.0e6;
            const auto now = std::chrono::steady_clock::now();
            const double wallMs = profileWallStart.time_since_epoch().count() == 0
                                      ? 0.0
                                      : std::chrono::duration<double, std::milli>(now - profileWallStart).count();
            // runCycles calls (one per EE checkpoint) and how many of them the idle skip answered, since the last line.
            const uint64_t calls = runCalls - profileRunCallsMark;
            const uint64_t skips = idleSkips - profileIdleSkipsMark;
            out << "[iop-prof] " << (64u * 1024u * 1024u) / 1000000u << "M instr in " << static_cast<uint64_t>(hostMs)
                << " ms inside the IOP of " << static_cast<uint64_t>(wallMs) << " ms wall ("
                << static_cast<uint64_t>(hostMs > 0.0 ? 64.0 * 1024.0 * 1024.0 / hostMs / 1000.0 : 0.0)
                << "M instr/s while running); calls " << calls << " ("
                << static_cast<uint64_t>(wallMs > 0.0 ? static_cast<double>(calls) * 1000.0 / wallMs : 0.0)
                << "/s), idle skips " << skips << " ("
                << static_cast<uint64_t>(wallMs > 0.0 ? static_cast<double>(skips) * 1000.0 / wallMs : 0.0)
                << "/s); top blocks:";
            profileHostNs = 0u;
            profileWallStart = now;
            profileRunCallsMark = runCalls;
            profileIdleSkipsMark = idleSkips;
            for (size_t i = 0; i < top.size() && i < 12u; ++i)
                out << " 0x" << std::hex << top[i].first << '=' << std::dec << (top[i].second * 100u / profileSamples) << '%';
            log(LogLevel::Info, out.str());
            profileCounts.clear();
            profileSamples = 0u;
        }

        class CpuNativeCall final : public ps2x::iop::NativeCall
        {
        public:
            CpuNativeCall(CpuState &cpu, IopMemory &memory) noexcept : m_cpu(cpu), m_memory(memory) {}
            [[nodiscard]] uint32_t reg(uint32_t index) const override { return index < 32u ? m_cpu.gpr[index] : 0u; }
            void setReg(uint32_t index, uint32_t value) override
            {
                if (index != 0u && index < 32u)
                    m_cpu.gpr[index] = value;
            }
            [[nodiscard]] bool readRam(uint32_t address, void *destination, size_t size) const override
            {
                return m_memory.readRam(address, destination, size);
            }
            bool writeRam(uint32_t address, const void *source, size_t size) override
            {
                return m_memory.writeRam(address, source, size);
            }

        private:
            CpuState &m_cpu;
            IopMemory &m_memory;
        };

        // VENDETTA_IOP_NATIVE_VERIFY: the routine being interpreted for a check, until it returns to $ra
        struct NativeWatch
        {
            CpuState *cpu = nullptr;
            const ps2x::iop::NativeHook *hook = nullptr;
            uint32_t ra = 0;
            uint32_t sp = 0;
            uint64_t instructions = 0;
        };
        NativeWatch nativeWatch;
        uint64_t nativeVerifyChecks = 0;
        uint64_t nativeVerifyDifferences = 0;

        // At a hooked routine's first instruction (a clean entry: no branch or load in flight, the code as hashed):
        // run it natively and continue at $ra; true = done. In the verify mode it is interpreted and checked instead.
        bool runNativeHook(CpuState &cpu)
        {
            if (cpu.branchPending || cpu.pendingLoad)
                return false;
            const uint32_t phys = physicalAddress(cpu.pc);
            for (const ps2x::iop::NativeHook &hook : nativeHooks())
            {
                if (physicalAddress(hook.address) != phys)
                    continue;
                const std::span<const uint8_t> ram = memory.ram();
                if (hook.codeBytes > ram.size() - phys ||
                    ps2x::iop::nativeCodeHash(ram.data() + phys, hook.codeBytes) != hook.codeHash)
                    return false;
                CpuNativeCall call(cpu, memory);
                static const bool verify = []
                {
                    const char *value = std::getenv("VENDETTA_IOP_NATIVE_VERIFY");
                    return value && *value && *value != '0';
                }();
                if (verify && hook.check)
                {
                    if (nativeWatch.cpu == nullptr)
                    {
                        hook.check(call, true, 0u);
                        nativeWatch = NativeWatch{&cpu, &hook, cpu.gpr[31], cpu.gpr[29], 1u};
                    }
                    return false;
                }
                const uint32_t instructions = hook.run ? hook.run(call) : 0u;
                if (instructions == 0u)
                    return false;
                static bool announced = false;
                if (!announced)
                {
                    announced = true;
                    std::ostringstream out;
                    out << "[iop-native] running " << hook.name << " natively (0x" << std::hex << hook.address << ")";
                    log(LogLevel::Info, out.str());
                }
                cpu.pc = cpu.gpr[31];
                totalInstructions += instructions;
                totalCycles += instructions;
                return true;
            }
            return false;
        }

        void watchNativeHook(CpuState &cpu)
        {
            if (cpu.pc != nativeWatch.ra || cpu.gpr[29] != nativeWatch.sp || cpu.branchPending)
            {
                ++nativeWatch.instructions;
                return;
            }
            CpuNativeCall call(cpu, memory);
            nativeVerifyDifferences += nativeWatch.hook->check(call, false, nativeWatch.instructions);
            if ((++nativeVerifyChecks & 255u) == 0u)
            {
                std::ostringstream out;
                out << "[iop-native] verify: " << nativeVerifyChecks << " calls of " << nativeWatch.hook->name
                    << " checked, " << nativeVerifyDifferences << " differences";
                log(LogLevel::Info, out.str());
            }
            nativeWatch = NativeWatch{};
        }

        // VENDETTA_IOPPROF: tested here, so profileSample is not called at every instruction when it is off
        const bool profileEnabled = std::getenv("VENDETTA_IOPPROF") != nullptr;

        bool step(CpuState &cpu)
        {
            if (profileEnabled)
                profileSample(cpu.pc);
            if (cpu.stopped)
                return false;
            if (cpu.pc == kThreadReturnSentinel || cpu.pc == kCallReturnSentinel)
            {
                cpu.stopped = true;
                return false;
            }
            if (physicalAddress(cpu.pc) >= kRamSize)
            {
                std::ostringstream out;
                out << "[IOP] execution outside RAM pc=0x" << std::hex << cpu.pc;
                log(LogLevel::Error, out.str());
                cpu.stopped = true;
                return false;
            }
            if (checkInterrupt(cpu))
                return true;
            if (nativeWatch.cpu == &cpu)
                watchNativeHook(cpu);
            if (g_nativeHookPages[physicalAddress(cpu.pc) >> 12] != 0u && runNativeHook(cpu))
                return !cpu.stopped;

            // An import stub is "jr $ra; addiu $0, $0, ordinal": only a jr $ra can be one, so the registry (which reads
            // the instruction again through IopMemory) is asked only then — the pc is in RAM (checked above).
            const uint32_t phys = physicalAddress(cpu.pc);
            uint32_t word = 0u;
            if ((phys & 3u) == 0u)
                std::memcpy(&word, memory.ram().data() + phys, sizeof(word));
            if (const auto import = ((phys & 3u) != 0u || word == 0x03E00008u) ? imports.decode(cpu.pc) : decltype(imports.decode(cpu.pc)){})
            {
                const ImportDisposition disposition = dispatchImport(*import, cpu);
                ++totalInstructions;
                ++totalCycles;
                if (disposition == ImportDisposition::JumpToGuest)
                    return true;
                cpu.pc = cpu.gpr[31];
                cpu.branchPending = false;
                return !cpu.stopped;
            }

            g_hctpIopPc = cpu.pc;
            const bool running = cpuCore.executeInstruction(cpu);
            g_hctpIopPc = 0xEEEEEEEEu;
            schedulePendingDma();
            ++totalInstructions;
            ++totalCycles;
            return running;
        }

        uint32_t runCpu(CpuState &cpu, uint32_t instructionBudget)
        {
            // Guest code can change anything the idle skip relies on (threads, DMA, callbacks, timers).
            idleKnown = false;
            CpuState *previous = activeCpu;
            activeCpu = &cpu;
            const uint64_t start = totalInstructions;
            while (!cpu.stopped && !cpu.yielded && totalInstructions - start < instructionBudget)
            {
                if (!step(cpu))
                    break;
                // After every instruction, as before - but only called when something can be due: a call with
                // nothing due changes nothing. No DMA interrupt completes before dmaDueFloor, and the guest
                // callbacks are kept in cycle order, so the first one is the earliest.
                if (!servicingDmaInterrupts && totalCycles >= dmaDueFloor)
                    servicePendingDmaInterrupts();
                if (!servicingGuestCallbacks && !pendingGuestCallbacks.empty() &&
                    pendingGuestCallbacks.begin()->first <= totalCycles)
                    servicePendingGuestCallbacks();
            }
            activeCpu = previous;
            idleKnown = false;
            return static_cast<uint32_t>(totalInstructions - start);
        }

        uint32_t callFunction(uint32_t address,
                              uint32_t a0,
                              uint32_t a1,
                              uint32_t a2,
                              uint32_t a3,
                              uint32_t gp,
                              uint32_t budget = kMaxCallInstructions,
                              uint32_t extraRounds = 0u)
        {
            struct CallDepthGuard
            {
                uint32_t &depth;
                ~CallDepthGuard() { --depth; }
            };

            const uint32_t depth = callDepth++;
            const CallDepthGuard depthGuard{callDepth};
            CpuState cpu{};
            cpu.pc = address;
            cpu.gpr[4] = a0;
            cpu.gpr[5] = a1;
            cpu.gpr[6] = a2;
            cpu.gpr[7] = a3;
            cpu.gpr[28] = gp;
            if (depth < kCallStackCapacity)
            {
                const uint32_t stackTop = kCallStackLimit - depth * kCallStackSize;
                cpu.gpr[29] = stackTop - 32u;
            }
            else if (activeCpu && activeCpu->gpr[29] > kCallStackBase + kStackGuardBytes)
            {
                // Extremely deep re-entrancy borrows unused space below the
                // suspended caller's live frame. Stack growth remains away
                // from the caller, so its saved registers stay intact.
                cpu.gpr[29] = (activeCpu->gpr[29] - kStackGuardBytes) & ~15u;
            }
            else
            {
                cpu.gpr[29] = kCallStackBase - 32u;
            }
            cpu.gpr[31] = kCallReturnSentinel;
            runCpu(cpu, budget);
            // A module start routine (run here, not as an IOP thread) may outlast one budget: libsd's sceSdInit
            // polls the SPU2, or it waits for a vblank. Real modload runs it to completion, so keep going.
            for (uint32_t round = 0; round < extraRounds && !cpu.stopped && cpu.pc != kCallReturnSentinel; ++round)
            {
                std::fprintf(stderr, "[iop-start] call 0x%x not finished (pc 0x%x yielded %d), round %u\n", address,
                             cpu.pc, cpu.yielded ? 1 : 0, round + 1u);
                cpu.yielded = false;
                runCpu(cpu, budget);
            }
            return cpu.gpr[2];
        }

        uint32_t executeGuestFunction(uint32_t address,
                                      uint32_t a0,
                                      uint32_t a1,
                                      uint32_t a2,
                                      uint32_t a3,
                                      uint32_t gp) override
        {
            return callFunction(address, a0, a1, a2, a3, gp);
        }

        uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                uint32_t a0,
                                                uint32_t a1,
                                                uint32_t a2,
                                                uint32_t a3,
                                                uint32_t gp,
                                                uint32_t instructionBudget) override
        {
            return callFunction(address, a0, a1, a2, a3, gp, instructionBudget);
        }

        // Not that good to use exception handling for control flow but will do for now
        void servicePendingDmaInterrupts()
        {
            {
                // HCTP_IOPDUMP_SEC=N: write IOP RAM to iop_ram.bin once, N seconds after the first call (diag)
                static const int dumpSec = std::getenv("HCTP_IOPDUMP_SEC") ? std::atoi(std::getenv("HCTP_IOPDUMP_SEC")) : 0;
                static const auto t0 = std::chrono::steady_clock::now();
                static bool dumped = false;
                if (dumpSec > 0 && !dumped && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(dumpSec))
                {
                    dumped = true;
                    const std::span<const uint8_t> ram = memory.ram();
                    if (FILE *f = std::fopen("iop_ram.bin", "wb"))
                    {
                        std::fwrite(ram.data(), 1, ram.size(), f);
                        std::fclose(f);
                    }
                    std::fprintf(stderr, "[iopdump] wrote iop_ram.bin (%zu bytes)%c", ram.size(), 10);
                }
                // HCTP: a file "iop_dump_now" in the game folder -> iop_ram_N.bin + every IOP thread's state (diag)
                static uint32_t pollTick = 0u;
                static int dumpIndex = 0;
                if ((++pollTick & 0xFFFu) == 0u)
                {
                    if (FILE *flag = std::fopen("iop_dump_now", "rb"))
                    {
                        std::fclose(flag);
                        std::remove("iop_dump_now");
                        char name[32];
                        std::snprintf(name, sizeof(name), "iop_ram_%d.bin", dumpIndex++);
                        const std::span<const uint8_t> ram = memory.ram();
                        if (FILE *f = std::fopen(name, "wb"))
                        {
                            std::fwrite(ram.data(), 1, ram.size(), f);
                            std::fclose(f);
                        }
                        std::fprintf(stderr, "[iopdump] %s at cycle %llu%c", name, static_cast<unsigned long long>(totalCycles), 10);
                        for (const auto &[id, t] : kernel.threads())
                            std::fprintf(stderr, "[iopthr] id %d state %d pc 0x%x ra 0x%x entry 0x%x prio %u wait %d bits 0x%x wake %llu%c",
                                         id, static_cast<int>(t.state), t.cpu.pc, t.cpu.gpr[31], t.entry, t.priority, t.waitId,
                                         t.waitBits, static_cast<unsigned long long>(t.wakeCycle), 10);
                    }
                }
            }
            // Before dmaDueFloor nothing in the list is due, and a walk that finds nothing due changes nothing.
            if (servicingDmaInterrupts || totalCycles < dmaDueFloor)
                return;
            if (pendingDmaInterrupts.empty())
            {
                dmaDueFloor = UINT64_MAX;
                return;
            }

            servicingDmaInterrupts = true;

            std::vector<int> completed;
            uint64_t remainingFloor = UINT64_MAX;
            for (auto it = pendingDmaInterrupts.begin(); it != pendingDmaInterrupts.end();)
            {
                if (it->second > totalCycles)
                {
                    remainingFloor = std::min(remainingFloor, it->second);
                    ++it;
                    continue;
                }
                completed.push_back(it->first);
                it = pendingDmaInterrupts.erase(it);
            }
            // Exact for what is left; a handler below that starts a new DMA lowers it again (schedulePendingDma).
            dmaDueFloor = remainingFloor;
            try
            {
                for (const int irq : completed)
                    (void)intrman.dispatchInterrupt(irq, *this);
            }
            catch (...)
            {
                servicingDmaInterrupts = false;
                throw;
            }
            servicingDmaInterrupts = false;
        }

        void servicePendingGuestCallbacks()
        {
            // In cycle order: when the first one is not due, none is, and the walk below would take nothing.
            if (servicingGuestCallbacks || pendingGuestCallbacks.empty() ||
                pendingGuestCallbacks.begin()->first > totalCycles)
                return;

            std::vector<ScheduledGuestCallback> callbacks;
            for (auto it = pendingGuestCallbacks.begin(); it != pendingGuestCallbacks.end();)
            {
                if (it->first > totalCycles)
                    break;
                callbacks.push_back(it->second);
                it = pendingGuestCallbacks.erase(it);
            }
            if (callbacks.empty())
                return;

            servicingGuestCallbacks = true;
            try
            {
                for (const ScheduledGuestCallback &callback : callbacks)
                {
                    if (callback.function != 0u)
                    {
                        (void)callFunction(callback.function,
                                           callback.argument,
                                           0u,
                                           0u,
                                           0u,
                                           callback.gp,
                                           100000u);
                    }
                }
            }
            catch (...)
            {
                servicingGuestCallbacks = false;
                throw;
            }
            servicingGuestCallbacks = false;
        }

        // Called at every EE checkpoint (millions of times a second), mostly with 1-4 IOP cycles, and most of
        // those calls find the IOP idle: every thread waiting and nothing due within them.
        void runCycles(uint64_t cycles) noexcept
        {
            // The IOP runs up to the total of the cycles it has been given, not "now + cycles": a step can take more
            // than one cycle (a native hook charges a whole routine at once) and pass the target; counted from where
            // the IOP stood, that overshoot was never paid back and the IOP's clock — the SPU2's 48 kHz output with it
            // — ran ahead of the EE's (the host's audio queue overflowed; the movies, timed by their sound, stuttered).
            // Time the IOP spent outside runCycles (a synchronous RPC, a module load) counts as given, as before.
            if (totalCycles > cyclesAfterLastRun)
                grantedCycles += totalCycles - cyclesAfterLastRun;
            grantedCycles += cycles;
            const uint64_t target = std::max(grantedCycles, totalCycles);
            ++runCalls;
            // The idle skip. idleKnown means the last pass of runCyclesFull ended in its idle branch and nothing
            // has changed since (every change clears it - see idleKnown). That branch found no thread ready and
            // nothing - delay wake, DMA completion, guest callback, timer - due before idleUntil. Running the loop
            // again with a target at or before idleUntil, its one pass would service nothing (all of those are due
            // at idleUntil or later, after the current cycle, and a service call with nothing due changes nothing),
            // find no thread (none was ready and no delay ends by then), take the idle branch with the same
            // earliest cycle and set totalCycles to min(target, idleUntil) = target. Setting it here is that same
            // result, and idleKnown / idleUntil are what that pass would have left.
            if (idleKnown && target <= idleUntil)
            {
                ++idleSkips;
                static const bool verify = std::getenv("VENDETTA_IOP_VERIFY") != nullptr;
                if (!verify)
                    totalCycles = target;
                else
                    verifyIdleSkip(target);
            }
            else
            {
                runCyclesFull(target);
            }
            if (totalCycles >= memory.spu2().nextSampleCycle())
                runSoundChip();
            cyclesAfterLastRun = totalCycles;
        }

        // The SPU2 plays up to the IOP's cycle (a 48 kHz sample every 768 cycles); its output goes to the host in
        // blocks of 256 frames (~5 ms).
        void runSoundChip() noexcept
        {
            try
            {
                memory.spu2().advanceTo(totalCycles, soundOutput);
                if (soundOutput.size() >= 512u)
                {
                    host.submitAudio(soundOutput.data(), soundOutput.size() / 2u);
                    soundOutput.clear();
                }
            }
            catch (...)
            {
                soundOutput.clear();
            }
        }

        void runCyclesFull(uint64_t target) noexcept
        {
            try
            {
                while (totalCycles < target)
                {
                    // This pass may change what the idle skip relies on (services, wakes, runs a thread); only
                    // its idle branch, the last thing a pass does, sets it again.
                    idleKnown = false;
                    ++loopPasses;
                    servicePendingDmaInterrupts();
                    servicePendingGuestCallbacks();
                    timrman.serviceDue(totalCycles, *this);
                    uint64_t nextDelayWake = UINT64_MAX;
                    IopThread *next = kernel.beginNextReady(totalCycles, &nextDelayWake);
                    if (!next)
                    {
                        // The earliest cycle anything is due. (min(target, this) is what the loop always used:
                        // nextWakeCycle(target) only folded target in earlier.)
                        uint64_t nextEvent = nextDelayWake;
                        for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                            nextEvent = std::min(nextEvent, completionCycle);
                        if (!pendingGuestCallbacks.empty())
                            nextEvent = std::min(nextEvent, pendingGuestCallbacks.begin()->first);
                        nextEvent = timrman.nextEventCycle(nextEvent);
                        totalCycles = std::max(totalCycles + 1u, std::min(target, nextEvent));
                        idleUntil = nextEvent;
                        // Only at the top level: nested inside running guest code (activeCpu set), a kernel call
                        // after this could make a thread ready without clearing the flag before runCpu returns.
                        idleKnown = activeCpu == nullptr;
                        continue;
                    }
                    ++threadSlices;
                    const uint64_t before = totalCycles;
                    runCpu(next->cpu, static_cast<uint32_t>(std::min<uint64_t>(kDefaultSlice, target - totalCycles)));
                    kernel.endTimeslice(*next, kThreadReturnSentinel);
                    if (totalCycles == before)
                        ++totalCycles;
                }
            }
            catch (...)
            {
                // Runtime scheduling must never throw through EeScheduler::accountCycles().
            }
        }

        // VENDETTA_IOP_VERIFY=1: where the idle skip would answer, run the full loop instead (its result is the
        // one kept) and check it did only what the skip assumes: one pass, nothing serviced, no thread run, no
        // instruction, the same delays / DMA list / callbacks / timers, ending at the target with the same idleUntil.
        void verifyIdleSkip(uint64_t target) noexcept
        {
            const uint64_t cycleBefore = totalCycles;
            const uint64_t untilBefore = idleUntil;
            const uint64_t instructionsBefore = totalInstructions;
            const uint64_t slicesBefore = threadSlices;
            const uint64_t passesBefore = loopPasses;
            const size_t dmaBefore = pendingDmaInterrupts.size();
            const size_t callbacksBefore = pendingGuestCallbacks.size();
            const uint64_t delayBefore = kernel.nextWakeCycle(UINT64_MAX);
            const uint64_t timerBefore = timrman.nextEventCycle(UINT64_MAX);
            runCyclesFull(target);
            ++verifyChecks;
            const bool same = totalCycles == target && totalInstructions == instructionsBefore &&
                              threadSlices == slicesBefore && loopPasses == passesBefore + 1u &&
                              pendingDmaInterrupts.size() == dmaBefore &&
                              pendingGuestCallbacks.size() == callbacksBefore &&
                              kernel.nextWakeCycle(UINT64_MAX) == delayBefore &&
                              timrman.nextEventCycle(UINT64_MAX) == timerBefore &&
                              idleKnown && idleUntil == untilBefore;
            if (!same && ++verifyMismatches <= 40u)
            {
                std::cerr << "[iopverify] idle skip differs from the full loop: cycle " << cycleBefore << " -> target "
                          << target << " (idle until " << untilBefore << "); the loop reached " << totalCycles
                          << " in " << (loopPasses - passesBefore) << " pass(es), ran " << (threadSlices - slicesBefore)
                          << " thread slice(s) and " << (totalInstructions - instructionsBefore)
                          << " instruction(s); DMA " << dmaBefore << "->" << pendingDmaInterrupts.size()
                          << ", callbacks " << callbacksBefore << "->" << pendingGuestCallbacks.size()
                          << ", idle " << (idleKnown ? "known" : "unknown") << " until " << idleUntil << std::endl;
            }
            if ((verifyChecks & ((1ull << 25u) - 1u)) == 0u)
                std::cerr << "[iopverify] " << verifyChecks << " idle skips checked against the full loop, "
                          << verifyMismatches << " differed" << std::endl;
        }

        ModuleLoadResult loadImage(std::string path, std::span<const uint8_t> image, const void *arguments, uint32_t argumentSize)
        {
            ModuleLoadResult result{true, -1, -1};
            const IopImageLoadResult loaded = IopModuleLoader::load(image, memory, moduleCursor);
            moduleCursor = loaded.nextModuleCursor;
            if (!loaded)
            {
                if (loaded.error == IopImageLoadError::InvalidElf)
                    log(LogLevel::Error, "[IOP] rejected invalid/non-MIPS IRX ELF");
                else if (loaded.error == IopImageLoadError::ArenaExhausted)
                    log(LogLevel::Error, "[IOP] module arena exhausted");
                return result;
            }
            if (!loaded.relocationsComplete)
                log(LogLevel::Warning, "[IOP] one or more IRX relocations were unsupported");

            Module module;
            module.id = nextModuleId++;
            module.path = std::move(path);
            const size_t slash = module.path.find_last_of("/\\:");
            module.name = slash == std::string::npos ? module.path : module.path.substr(slash + 1u);
            module.base = loaded.base;
            module.size = loaded.size;
            module.entry = loaded.entry;
            module.gp = loaded.gp;

            uint32_t args = 0u;
            if (arguments && argumentSize)
            {
                args = allocate(argumentSize + 1u, 16u);
                if (args)
                {
                    writeRam(args, arguments, argumentSize);
                    write8(args + argumentSize, 0u);
                }
            }
            const uint32_t startResult = callFunction(module.entry, argumentSize, args, 0u, 0u, module.gp,
                                                      kMaxCallInstructions, 50u);
            if (args)
                freeAllocation(args);
            module.resident = startResult == 0u || startResult == 2u;
            result.moduleId = module.id;
            result.startResult = static_cast<int32_t>(startResult);
            modules[module.id] = std::move(module);

            std::ostringstream out;
            out << "[IOP] loaded IRX id=" << result.moduleId
                << " entry=0x" << std::hex << modules[result.moduleId].entry
                << " base=0x" << modules[result.moduleId].base
                << " start=" << std::dec << result.startResult;
            log(LogLevel::Info, out.str());
            return result;
        }

        ModuleLoadResult loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readWholeHostFile(host, path, image))
            {
                log(LogLevel::Warning, std::string("[IOP] failed to open IRX '") + std::string(path) + "'");
                return {true, -1, -1};
            }
            return loadImage(std::string(path), image, arguments, argumentSize);
        }

        ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readElfFromGuest(host, guestAddress, image) &&
                !IopModuleLoader::readElfFromIop(memory, guestAddress, image))
                return {true, -1, -1};
            std::ostringstream tag;
            tag << "buffer@0x" << std::hex << guestAddress;
            return loadImage(tag.str(), image, arguments, argumentSize);
        }

        bool stopModule(int32_t moduleId, int32_t *result)
        {
            idleKnown = false; // threads are ended here without running guest code
            auto it = modules.find(moduleId);
            if (it == modules.end())
                return false;
            // A removable IRX normally exposes a stop entry through module metadata. We do not guess it; terminate owned execution and release the image cleanly.
            kernel.terminateThreadsInRange(it->second.base, it->second.size);
            rpc.removeServersInRange(it->second.base, it->second.size);
            imports.eraseRange(it->second.base, it->second.size);
            modules.erase(it);
            kernel.cleanupDeadThreads();
            if (result)
                *result = 0;
            return true;
        }

        IopHost &host;
        IopMemory memory;
        IopSysmem sysmem;
        IopKernel kernel;
        IopCdvd cdvd;
        IopVblank vblank;
        IopRpcBridge rpc;
        IopSysclib sysclib;
        IopStdio stdio;
        IopHeaplib heaplib;
        IopIntrman intrman;
        IopTimrman timrman;
        IopIoman ioman;
        IopCpuCore cpuCore;
        IopImportRegistry imports;
        IopLoadcore loadcore;
        std::map<int, Module> modules;
        std::map<int, uint64_t> pendingDmaInterrupts;
        std::unordered_map<uint32_t, uint64_t> profileCounts;
        uint64_t profileSamples = 0u;
        uint64_t profileHostNs = 0u;
        std::chrono::steady_clock::time_point profileWallStart{};
        std::multimap<uint64_t, ScheduledGuestCallback> pendingGuestCallbacks;
        // A lower bound of every completion cycle in pendingDmaInterrupts (UINT64_MAX: none due ever): kept by
        // schedulePendingDma (min with each new one) and set exactly by each walk in servicePendingDmaInterrupts.
        uint64_t dmaDueFloor = UINT64_MAX;
        // The idle skip in runCycles. Set only by runCyclesFull's idle branch: then no thread was ready and nothing
        // was due before idleUntil. Cleared by everything that can change that: each pass of runCyclesFull, runCpu
        // (all guest code: threads, RPC servers, SIF command handlers, interrupt / timer / cdvd callbacks, module
        // starts) on entry and exit, a new DMA (schedulePendingDma), a new cdvd callback, reset, stopModule, and
        // every IopEmulator call from the EE side that changes IOP state.
        bool idleKnown = false;
        uint64_t idleUntil = 0u;
        uint64_t runCalls = 0u;      // runCycles calls
        uint64_t idleSkips = 0u;     // ... answered by the idle skip
        uint64_t threadSlices = 0u;  // runCyclesFull passes that ran a thread
        uint64_t loopPasses = 0u;    // runCyclesFull passes
        uint64_t verifyChecks = 0u;
        uint64_t verifyMismatches = 0u;
        uint64_t profileRunCallsMark = 0u;
        uint64_t profileIdleSkipsMark = 0u;
        uint32_t nextModuleId = 1;
        uint32_t moduleCursor = kModuleLoadBase;
        uint64_t grantedCycles = 0;      // the cycles runCycles has handed the IOP so far (see there)
        uint64_t cyclesAfterLastRun = 0; // totalCycles when runCycles last returned
        uint64_t totalCycles = 0;
        std::vector<int16_t> soundOutput; // SPU2 frames not yet handed to the host
        uint64_t totalInstructions = 0;
        uint64_t eeCycleCarry = 0;
        CpuState *activeCpu = nullptr;
        std::string lastError;
        bool servicingDmaInterrupts = false;
        bool servicingGuestCallbacks = false;
        uint32_t callDepth = 0u;
        GuestCallback secrMcCommandHandler;
        GuestCallback secrMcDevIdHandler;
        GuestCallback checkKelfPathCallback;
    };

    IopEmulator::IopEmulator(IopHost &host)
        : m_impl(std::make_unique<Impl>(host))
    {
    }

    IopEmulator::~IopEmulator() = default;

    void IopEmulator::reset()
    {
        m_impl->reset();
    }

    // Every call from the EE side that can change IOP state clears the idle skip first (Impl::idleKnown): most of
    // them run guest code, which clears it anyway, but this way none can be missed.
    ModuleLoadResult IopEmulator::loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
    {
        m_impl->idleKnown = false;
        return m_impl->loadModule(path, arguments, argumentSize);
    }

    ModuleLoadResult IopEmulator::loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
    {
        m_impl->idleKnown = false;
        return m_impl->loadModuleBuffer(guestAddress, arguments, argumentSize);
    }

    bool IopEmulator::stopModule(int32_t moduleId, int32_t *result)
    {
        m_impl->idleKnown = false;
        return m_impl->stopModule(moduleId, result);
    }

    void IopEmulator::runEeCycles(uint64_t eeCycles) noexcept
    {
        // The EE hands its time over at every checkpoint, 8-32 EE cycles = 1-4 IOP cycles. While an IOP thread is
        // ready, each such call is a whole pass of runCyclesFull (DMA and callback servicing, timers, picking a thread)
        // for 1-4 instructions — the busier the EE, the finer the IOP's time is cut and the more it costs (a sound
        // driver's 2 M instructions a second took 625,000 passes a second while the EE was busy, 3.3 instructions each,
        // against ~20,000 when it idles), until a game thread that falls behind never catches up. So while the IOP has
        // work, its time is handed over in batches of kIopBatchCycles (~14 us at 36.864 MHz — the EE and the IOP run side
        // by side on the console, and nothing the EE can see is that fine); an IOP known to be idle through the cycles
        // takes them at once (the idle skip, as cheap as before).
        constexpr uint64_t kIopBatchCycles = 512u;
        const uint64_t total = m_impl->eeCycleCarry + eeCycles;
        const uint64_t iopCycles = total / 8u;
        if (!iopCycles)
        {
            m_impl->eeCycleCarry = total;
            return;
        }
        if (iopCycles < kIopBatchCycles &&
            !(m_impl->idleKnown && m_impl->totalCycles + iopCycles <= m_impl->idleUntil))
        {
            m_impl->eeCycleCarry = total;
            return;
        }
        m_impl->eeCycleCarry = total % 8u;
        spu2NativeService(m_impl->memory.spu2(), m_impl->totalCycles);
        static const bool profile = std::getenv("VENDETTA_IOPPROF") != nullptr;
        if (!profile)
        {
            m_impl->runCycles(iopCycles);
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        m_impl->runCycles(iopCycles);
        m_impl->profileHostNs += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    }

    RpcResult IopEmulator::handleRpc(const RpcRequest &request)
    {
        m_impl->idleKnown = false;
        return m_impl->rpc.handleRpc(request, *m_impl);
    }

    bool IopEmulator::handleSifCommand(uint32_t commandId, const void *packet, size_t packetSize)
    {
        m_impl->idleKnown = false;
        return m_impl->rpc.handleSifCommand(commandId, packet, packetSize, *m_impl);
    }

    bool IopEmulator::hasRpcServer(uint32_t sid) const noexcept
    {
        return m_impl->rpc.hasServer(sid);
    }

    void IopEmulator::onSifTransfer(const SifTransfer &transfer)
    {
        m_impl->idleKnown = false;
        m_impl->rpc.onSifTransfer(transfer);
    }

    uint32_t IopEmulator::allocateMemory(uint32_t size, uint32_t alignment)
    {
        m_impl->idleKnown = false;
        return m_impl->memory.allocate(size, alignment);
    }

    bool IopEmulator::freeMemory(uint32_t address)
    {
        m_impl->idleKnown = false;
        return m_impl->memory.freeAllocation(address);
    }

    bool IopEmulator::readMemory(uint32_t address, void *destination, size_t size) const
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.readRam(address, destination, size);
    }

    bool IopEmulator::writeMemory(uint32_t address, const void *source, size_t size)
    {
        m_impl->idleKnown = false;
        return isMemoryRange(address, size) &&
               m_impl->memory.writeRam(address, source, size);
    }

    bool IopEmulator::zeroMemory(uint32_t address, size_t size)
    {
        m_impl->idleKnown = false;
        return isMemoryRange(address, size) &&
               m_impl->memory.zeroRam(address, size);
    }

    bool IopEmulator::isMemoryRange(uint32_t address, size_t size) const
    {
        const bool physicalSegment = address < IopMemory::RamSize;
        const bool cachedSegment = address >= 0x80000000u && address < 0x80200000u;
        const bool uncachedSegment = address >= 0xA0000000u && address < 0xA0200000u;
        if (!physicalSegment && !cachedSegment && !uncachedSegment)
            return false;
        const uint32_t physical = IopMemory::physicalAddress(address);
        return physical <= IopMemory::RamSize && size <= IopMemory::RamSize - physical;
    }

    uint64_t IopEmulator::cycles() const noexcept
    {
        return m_impl->totalCycles;
    }

    uint64_t IopEmulator::instructions() const noexcept
    {
        return m_impl->totalInstructions;
    }

    uint32_t IopEmulator::loadedModuleCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->modules.size());
    }

    uint32_t IopEmulator::threadCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->kernel.threadCount());
    }

    uint32_t IopEmulator::rpcServerCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->rpc.serverCount());
    }

}

namespace ps2x::iop
{
    void registerNativeHook(const NativeHook &hook)
    {
        detail::nativeHooks().push_back(hook);
        const uint32_t phys = detail::physicalAddress(hook.address);
        if (phys < detail::kRamSize)
            detail::g_nativeHookPages[phys >> 12] = 1u;
    }
}
