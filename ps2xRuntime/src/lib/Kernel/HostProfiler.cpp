// Host-side code-address sampling for the profiler (ps2_host_profile.h): which host function (and source line) each
// recording thread — the game thread, the GS thread, the chain worker — is in, sampled by the profiler's thread
// (VENDETTA_HOSTPROF=2). Windows only: a thread is suspended for a moment, its instruction pointer read, and resumed;
// nothing is allocated while it is stopped (it may hold the heap lock).
// The addresses are named afterwards with DbgHelp from the exe's PDB (link with /DEBUG). Code built with debug info
// (the runtime library and the compiled VU1 code: /Z7 in ps2xRuntime/CMakeLists.txt) is named down to its source
// lines; other code (the generated game functions) by its public symbols (FUN_xxxxxxxx).
// VENDETTA_PCPROF_DUMP=<file>: every report also appends all its per-line counts there (tab-separated: thread, count,
// file:line, function) for offline analysis.
#include "runtime/ps2_host_profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

namespace ps2_host_profile
{
    namespace
    {
        std::atomic<uint32_t> g_recordingThreadOsIds[TrackCount]{};

        struct PcSample
        {
            uint64_t pc;
            uint8_t phase;
        };

        struct TrackSampler
        {
            HANDLE thread = nullptr;
            uint32_t threadId = 0;
            std::vector<PcSample> samples;
        };

        struct PcSampler
        {
            TrackSampler tracks[TrackCount];
            bool symbolsReady = false;
            std::unordered_map<uint64_t, std::string> names; // address -> function name
            std::unordered_map<uint64_t, std::string> lines; // address -> "file:line" (or "")
        };

        PcSampler &sampler()
        {
            static PcSampler instance;
            return instance;
        }

        const std::string &nameOf(PcSampler &state, uint64_t pc)
        {
            auto found = state.names.find(pc);
            if (found != state.names.end())
                return found->second;
            alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512];
            SYMBOL_INFO *symbol = reinterpret_cast<SYMBOL_INFO *>(buffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = 511;
            DWORD64 displacement = 0;
            std::string name;
            if (state.symbolsReady && SymFromAddr(GetCurrentProcess(), pc, &displacement, symbol))
                name.assign(symbol->Name, symbol->NameLen);
            else
            {
                char text[48];
                const DWORD64 base = state.symbolsReady ? SymGetModuleBase64(GetCurrentProcess(), pc) : 0;
                std::snprintf(text, sizeof(text), "?+0x%llx", static_cast<unsigned long long>(pc - base));
                name = text;
            }
            return state.names.emplace(pc, std::move(name)).first->second;
        }

        const std::string &lineOf(PcSampler &state, uint64_t pc)
        {
            auto found = state.lines.find(pc);
            if (found != state.lines.end())
                return found->second;
            std::string text;
            IMAGEHLP_LINE64 line{};
            line.SizeOfStruct = sizeof(line);
            DWORD displacement = 0;
            if (state.symbolsReady && SymGetLineFromAddr64(GetCurrentProcess(), pc, &displacement, &line) && line.FileName)
            {
                const char *file = line.FileName;
                for (const char *p = line.FileName; *p; ++p)
                    if (*p == '\\' || *p == '/')
                        file = p + 1;
                text = std::string(file) + ":" + std::to_string(line.LineNumber);
            }
            return state.lines.emplace(pc, std::move(text)).first->second;
        }

        bool openTrackThread(TrackSampler &state, uint32_t track)
        {
            const uint32_t id = g_recordingThreadOsIds[track].load(std::memory_order_relaxed);
            if (id == 0u)
                return false;
            if (state.threadId != id)
            {
                if (state.thread)
                    CloseHandle(state.thread);
                state.thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, id);
                state.threadId = state.thread ? id : 0u;
                if (!state.thread)
                    return false;
                state.samples.reserve(1u << 16);
            }
            return true;
        }

        // this exe's image, for telling return addresses from other stack words
        struct ImageRange
        {
            uint64_t begin = 0;
            uint64_t end = 0;
        };

        const ImageRange &exeImage()
        {
            static const ImageRange range = []
            {
                ImageRange r;
                const auto *base = reinterpret_cast<const uint8_t *>(GetModuleHandleW(nullptr));
                const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
                const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
                r.begin = reinterpret_cast<uint64_t>(base) + 0x1000u;
                r.end = reinterpret_cast<uint64_t>(base) + nt->OptionalHeader.SizeOfImage;
                return r;
            }();
            return range;
        }

        // does the code just before this address end in a call (rel32, or indirect through a register / memory)?
        bool followsCall(uint64_t address)
        {
            const ImageRange &image = exeImage();
            if (address < image.begin + 8u || address >= image.end)
                return false;
            const auto *b = reinterpret_cast<const uint8_t *>(address);
            if (b[-5] == 0xE8u)
                return true; // call rel32
            if (b[-6] == 0xFFu && (b[-5] & 0x38u) == 0x10u && ((b[-5] & 0xC0u) == 0x80u || (b[-5] & 0xC7u) == 0x05u))
                return true; // call [rip+disp32] / [reg+disp32]
            if (b[-2] == 0xFFu && (b[-1] & 0xF8u) == 0xD0u)
                return true; // call reg
            if (b[-3] == 0xFFu && (b[-2] & 0xF8u) == 0x50u)
                return true; // call [reg+disp8]
            if (b[-3] == 0xFFu && (b[-2] & 0xF8u) == 0x10u)
                return true; // call [reg] (+ a REX prefix before)
            if (b[-4] == 0xFFu && (b[-3] & 0x38u) == 0x10u && (b[-3] & 0xC7u) == 0x44u)
                return true; // call [reg+index+disp8]
            return false;
        }

        const char *trackName(uint32_t track)
        {
            static const char *const names[TrackCount] = {"game thread", "GS thread", "chain worker"};
            return track < TrackCount ? names[track] : "?";
        }
    }

    void noteRecordingThread(Track track) noexcept
    {
        if (track < TrackCount)
            g_recordingThreadOsIds[track].store(GetCurrentThreadId(), std::memory_order_relaxed);
    }

    bool samplePc(Track track, uint8_t phase) noexcept
    {
        if (track >= TrackCount)
            return false;
        TrackSampler &state = sampler().tracks[track];
        if (!openTrackThread(state, track))
            return false;
        if (state.samples.size() >= state.samples.capacity())
            return false; // never allocate while sampling
        if (SuspendThread(state.thread) == static_cast<DWORD>(-1))
            return false;
        CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL;
        const bool ok = GetThreadContext(state.thread, &context) != 0;
        ResumeThread(state.thread);
        if (!ok)
            return false;
#if defined(_M_X64) || defined(__x86_64__)
        state.samples.push_back({context.Rip, phase});
#elif defined(_M_ARM64)
        state.samples.push_back({context.Pc, phase});
#else
        state.samples.push_back({context.Eip, phase});
#endif
        return true;
    }

    size_t stackScan(Track track, uint64_t *out, size_t maxOut) noexcept
    {
        if (track >= TrackCount || maxOut == 0u)
            return 0u;
        TrackSampler &state = sampler().tracks[track];
        if (!openTrackThread(state, track))
            return 0u;
        static uint64_t copy[8192]; // 64 KB of the stack, read while the thread is stopped (only the profiler thread scans)
        (void)exeImage();
        if (SuspendThread(state.thread) == static_cast<DWORD>(-1))
            return 0u;
        CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL;
        size_t words = 0u;
        uint64_t pc = 0u;
        if (GetThreadContext(state.thread, &context))
        {
#if defined(_M_X64) || defined(__x86_64__)
            pc = context.Rip;
            const uint64_t sp = context.Rsp;
            MEMORY_BASIC_INFORMATION region{};
            if (VirtualQuery(reinterpret_cast<const void *>(sp), &region, sizeof(region)) == sizeof(region))
            {
                const uint64_t regionEnd = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
                words = static_cast<size_t>(std::min<uint64_t>((regionEnd - sp) / 8u, sizeof(copy) / 8u));
                std::memcpy(copy, reinterpret_cast<const void *>(sp), words * 8u);
            }
#endif
        }
        ResumeThread(state.thread);
        if (pc == 0u)
            return 0u;
        size_t count = 0u;
        out[count++] = pc;
        for (size_t i = 0; i < words && count < maxOut; ++i)
            if (followsCall(copy[i]))
                out[count++] = copy[i];
        return count;
    }

    std::string describeStack(const uint64_t *pcs, size_t count)
    {
        PcSampler &state = sampler();
        if (!state.symbolsReady)
        {
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
            state.symbolsReady = SymInitialize(GetCurrentProcess(), nullptr, TRUE) != 0;
        }
        std::string out;
        std::string previous;
        for (size_t i = 0; i < count; ++i)
        {
            const std::string &name = nameOf(state, pcs[i]);
            if (name == previous)
                continue;
            previous = name;
            if (!out.empty())
                out += " <- ";
            out += name;
        }
        return out;
    }

    std::string pcReport(uint32_t topOverall, uint32_t topPerPhase)
    {
        PcSampler &state = sampler();
        if (!state.symbolsReady)
        {
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
            state.symbolsReady = SymInitialize(GetCurrentProcess(), nullptr, TRUE) != 0;
        }
        const auto top = [](const std::unordered_map<std::string, uint32_t> &counts, uint32_t limit)
        {
            std::vector<std::pair<std::string, uint32_t>> sorted(counts.begin(), counts.end());
            std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b)
                      { return a.second != b.second ? a.second > b.second : a.first < b.first; });
            if (sorted.size() > limit)
                sorted.resize(limit);
            return sorted;
        };
        std::string out;
        char line[640];
        const char *dumpPath = std::getenv("VENDETTA_PCPROF_DUMP");
        FILE *dump = dumpPath && *dumpPath ? std::fopen(dumpPath, "a") : nullptr;
        for (uint32_t track = 0; track < TrackCount; ++track)
        {
            std::vector<PcSample> &samples = state.tracks[track].samples;
            const size_t total = samples.size();
            if (total == 0u)
                continue;
            std::unordered_map<std::string, uint32_t> overall;
            std::unordered_map<std::string, uint32_t> byPhase[Count];
            std::unordered_map<std::string, uint32_t> byLine;
            uint32_t phaseTotals[Count] = {};
            for (const PcSample &sample : samples)
            {
                const std::string &name = nameOf(state, sample.pc);
                ++overall[name];
                const std::string &source = lineOf(state, sample.pc);
                if (!source.empty())
                    ++byLine[source + "\t" + name];
                if (sample.phase < Count)
                {
                    ++byPhase[sample.phase][name];
                    ++phaseTotals[sample.phase];
                }
            }
            samples.clear();

            std::snprintf(line, sizeof(line), "[pcprof] %s: %zu samples; top functions:\n", trackName(track), total);
            out += line;
            for (const auto &[name, count] : top(overall, topOverall))
            {
                std::snprintf(line, sizeof(line), "[pcprof]   %5.1f%%  %s\n", 100.0 * count / total, name.c_str());
                out += line;
            }
            if (dump)
                for (const auto &[where, count] : byLine)
                    std::fprintf(dump, "%s\t%u\t%s\n", trackName(track), count, where.c_str());
            if (!byLine.empty())
            {
                std::snprintf(line, sizeof(line), "[pcprof] %s: top source lines:\n", trackName(track));
                out += line;
                for (const auto &[where, count] : top(byLine, topOverall))
                {
                    std::snprintf(line, sizeof(line), "[pcprof]   %5.1f%%  %s\n", 100.0 * count / total, where.c_str());
                    out += line;
                }
            }
            if (track != GameTrack)
                continue; // the phase breakdown is for the game thread (the others mostly run one kind of work)
            for (uint8_t phase = 0; phase < Count; ++phase)
            {
                if (phaseTotals[phase] * 50u < total) // phases under 2% of the samples: skip
                    continue;
                std::snprintf(line, sizeof(line), "[pcprof] within %s (%.1f%% of samples):\n", name(phase), 100.0 * phaseTotals[phase] / total);
                out += line;
                for (const auto &[fn, count] : top(byPhase[phase], topPerPhase))
                {
                    std::snprintf(line, sizeof(line), "[pcprof]   %5.1f%%  %s\n", 100.0 * count / total, fn.c_str());
                    out += line;
                }
            }
        }
        if (dump)
            std::fclose(dump);
        if (out.empty())
            return "[pcprof] no samples (VENDETTA_HOSTPROF=2 samples the game thread's, the GS thread's and the chain worker's code addresses)\n";
        return out;
    }
}

#else

namespace ps2_host_profile
{
    void noteRecordingThread(Track) noexcept {}
    bool samplePc(Track, uint8_t) noexcept { return false; }
    size_t stackScan(Track, uint64_t *, size_t) noexcept { return 0u; }
    std::string describeStack(const uint64_t *, size_t) { return {}; }
    std::string pcReport(uint32_t, uint32_t) { return "[pcprof] not available on this platform\n"; }
}

#endif
