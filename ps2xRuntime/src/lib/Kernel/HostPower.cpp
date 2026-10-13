// Windows power throttling ("EcoQoS"): Windows 11 may run a process whose window is not in front (or a locked /
// remote session) at reduced speed, on the efficiency cores of a hybrid CPU (Intel 12th gen and later). A game wants
// full speed whatever its window state, so the runtime opts out at start-up. VENDETTA_ECOQOS=1 leaves Windows' choice.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

void ps2HostPreferPerformance()
{
#if defined(_WIN32) && defined(PROCESS_POWER_THROTTLING_CURRENT_VERSION)
    const char *keep = std::getenv("VENDETTA_ECOQOS");
    if (keep && *keep == '1')
        return;
    PROCESS_POWER_THROTTLING_STATE state{};
    state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask = 0u; // execution-speed throttling off
    if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state)))
        std::fprintf(stderr, "[host] could not opt out of power throttling (error %lu)\n", GetLastError());
#endif
}

// On a hybrid CPU (performance + efficiency cores, e.g. Intel 12th gen), Windows may run a busy thread on an
// efficiency core at about half the speed — measured: the chain worker's VU1 rate halved for 10 s of a match. The
// threads that set the pace (the game thread, the chain worker, the GS thread) ask for the performance cores; the
// drawing helpers stay free to use any core. Not hybrid: nothing changes. VENDETTA_PCORES=0 leaves Windows' choice.
void ps2HostPreferPerformanceCores()
{
#if defined(_WIN32)
    static const std::vector<ULONG> performanceSets = []()
    {
        std::vector<ULONG> ids;
        const char *setting = std::getenv("VENDETTA_PCORES");
        if (setting && *setting == '0')
            return ids;
        ULONG length = 0;
        GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
        if (length == 0)
            return ids;
        std::vector<uint8_t> buffer(length);
        auto *first = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data());
        if (!GetSystemCpuSetInformation(first, length, &length, GetCurrentProcess(), 0))
            return ids;
        BYTE best = 0, lowest = 0xFF;
        for (ULONG offset = 0; offset < length;)
        {
            const auto *info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION *>(buffer.data() + offset);
            if (info->Type == CpuSetInformation)
            {
                best = std::max(best, info->CpuSet.EfficiencyClass);
                lowest = std::min(lowest, info->CpuSet.EfficiencyClass);
            }
            offset += info->Size;
        }
        if (best == lowest)
            return ids; // one kind of core
        for (ULONG offset = 0; offset < length;)
        {
            const auto *info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION *>(buffer.data() + offset);
            if (info->Type == CpuSetInformation && info->CpuSet.EfficiencyClass == best)
                ids.push_back(info->CpuSet.Id);
            offset += info->Size;
        }
        std::fprintf(stderr, "[host] hybrid CPU: the game, chain and GS threads prefer the %zu performance-core processors\n", ids.size());
        return ids;
    }();
    if (!performanceSets.empty())
        SetThreadSelectedCpuSets(GetCurrentThread(), performanceSets.data(), static_cast<ULONG>(performanceSets.size()));
#endif
}
