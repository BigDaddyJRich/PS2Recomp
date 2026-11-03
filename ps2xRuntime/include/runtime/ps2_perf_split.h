#pragma once

// HCTP_PERF_SPLIT=1: per-second wall-clock split of the render pipeline, one stderr line a second:
//   [perfsplit] fps=<presents> chain busy=<ms> (vu1 <ms>) idle=<ms> | gs-thread busy=<ms> idle=<ms> |
//               ee fence=<ms> (n) chain-throttle=<ms> gs-wait=<ms>
// chain busy = ChainThread running records (VIF1 unpack, VU1, GIF parse, and - with HCTP_GS_THREAD=0 - the D3D11
// work too); vu1 = inside the VU1 MSCAL / MSCNT callbacks (native or interpreted, incl. the XGKICKs they make);
// gs-thread busy = GSThread running renderer commands (D3D11); ee fence = the game thread waiting in
// ChainWorker::fence (GIF_STAT / CSR ...); chain-throttle = the present backlog wait; gs-wait = any producer waiting
// on the GS thread (drain for a readback, ring full, flip backlog). Off: one relaxed load per scope, no timestamps.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ps2_perf_split
{
    enum Bucket
    {
        ChainBusy,
        Vu1,
        GsBusy,
        EeFence,
        ChainThrottle,
        GsWait,
        BucketCount
    };

    // HCTP_PERF_PHASES=1 (with HCTP_PERF_SPLIT=1): exclusive time per phase on the thread inside a ChainBusy batch
    // (nested phases are not charged to their parent), printed as a second [perfphase] line.
    enum Phase
    {
        PhOther,   // chain batch time not inside a phase below (ring loop, DrainArbiter, register writes)
        PhVif,     // runVif1: VIF1 decode / unpack
        PhVu1Nat,  // VU1 MSCAL/MSCNT callback, native program
        PhVu1Int,  // VU1 interpreter (Vu1 execute)
        PhGif,     // GS front end: processGIFPacket (GIFtag / register / vertex parse)
        PhD3d,     // D3D11 backend Submit/LoadClut/Flush/Sync/Clear (draw build + D3D11 calls)
        PhTexDec,  // D3D11 decodedTexture (CPU VRAM mirror read + upload)
        PhRtSync,  // RT / depth readback into the CPU VRAM mirror (syncRtToVram / syncDepthRect)
        PhXfer,    // BeginTransfer / UploadImage (host->local into the CPU VRAM mirror)
        PhPresent, // runPresent
        // parts of vu1nat (hctp_native_vu1.cpp), exclusive like the others
        PhNatS1,    // stage-1 skinning + lighting (p01-p04 MSCAL)
        PhNatChunk, // p08/p18/p11/p19... chunk maths: clip flags, ADC, corners (computeChunk)
        PhNatGpu,   // GPU-character draw build (corner records -> GSNativeVtx) + submit
        PhNatGen,   // generic native programs (p17 crowd, p14, p16 ...) via MscalGeneric
        PhNatP07,   // p07 morph weights
        PhNatSub,   // GS::submitNativeTris (front end state + ring copy), inside n-gpu
        PhaseCount
    };
    inline const char *const kPhaseNames[PhaseCount] = {"other", "vif", "vu1nat", "vu1int", "gif",
                                                        "d3d", "texdec", "rtsync", "xfer", "present",
                                                        "n-s1", "n-chunk", "n-gpu", "n-gen", "n-p07", "n-sub"};

    inline bool enabled()
    {
        static const bool on = [] {
            const char *v = std::getenv("HCTP_PERF_SPLIT");
            return v && *v && *v != '0';
        }();
        return on;
    }

    inline std::atomic<uint64_t> g_ns[BucketCount]{};
    inline std::atomic<uint64_t> g_fences{0};

    inline uint64_t nowNs()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }

    inline bool phasesEnabled()
    {
        static const bool on = [] {
            const char *v = std::getenv("HCTP_PERF_PHASES");
            return enabled() && v && *v && *v != '0';
        }();
        return on;
    }
    inline std::atomic<uint64_t> g_phaseNs[PhaseCount]{};
    inline thread_local int t_phase = -1; // -1 = not inside a chain batch
    inline thread_local uint64_t t_phaseStart = 0;

    // Charges the time since the last switch to the current phase and makes `p` current; returns the old phase.
    inline int switchPhase(int p)
    {
        const int old = t_phase;
        const uint64_t now = nowNs();
        if (old >= 0)
            g_phaseNs[old].fetch_add(now - t_phaseStart, std::memory_order_relaxed);
        t_phase = p;
        t_phaseStart = now;
        return old;
    }

    // Root (chain batch) or nested phase. Nested scopes only count inside a root.
    struct PhaseScope
    {
        int prev = -2;
        explicit PhaseScope(Phase p, bool root = false)
        {
            if (phasesEnabled() && (root || t_phase >= 0))
                prev = switchPhase(p);
        }
        ~PhaseScope()
        {
            if (prev != -2)
                switchPhase(prev);
        }
        PhaseScope(const PhaseScope &) = delete;
        PhaseScope &operator=(const PhaseScope &) = delete;
    };

    inline void add(Bucket b, uint64_t ns) { g_ns[b].fetch_add(ns, std::memory_order_relaxed); }

    struct Scope
    {
        Bucket bucket;
        uint64_t start;
        explicit Scope(Bucket b) : bucket(b), start(enabled() ? nowNs() : 0u) {}
        ~Scope()
        {
            if (start)
                add(bucket, nowNs() - start);
        }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
    };

    // Nested VU1 callbacks (MSCNT inside a slice of MSCAL) count once.
    struct Vu1Scope
    {
        static inline thread_local int t_depth = 0;
        uint64_t start = 0;
        Vu1Scope()
        {
            if (enabled() && t_depth++ == 0)
                start = nowNs();
        }
        ~Vu1Scope()
        {
            if (!enabled())
                return;
            if (--t_depth == 0 && start)
                add(Vu1, nowNs() - start);
        }
    };

    // Called once per presentation by the game thread; prints and resets once a second.
    inline void tick()
    {
        if (!enabled())
            return;
        static uint64_t s_last = nowNs();
        static uint32_t s_presents = 0;
        ++s_presents;
        const uint64_t now = nowNs();
        const uint64_t span = now - s_last;
        if (span < 1000000000ull)
            return;
        uint64_t v[BucketCount];
        for (int i = 0; i < BucketCount; ++i)
            v[i] = g_ns[i].exchange(0u, std::memory_order_relaxed);
        const uint64_t fences = g_fences.exchange(0u, std::memory_order_relaxed);
        // fraction of the span x 1000 = ms per second
        const auto ms = [&](uint64_t ns) { return static_cast<double>(ns) * 1000.0 / static_cast<double>(span); };
        const double wall = 1000.0;
        std::fprintf(stderr,
                     "[perfsplit] fps=%.1f chain busy=%.0f (vu1 %.0f) idle=%.0f | gs-thread busy=%.0f idle=%.0f | ee fence=%.0f (%llu) chain-throttle=%.0f gs-wait=%.0f ms/s%c",
                     s_presents * 1e9 / static_cast<double>(span), ms(v[ChainBusy]), ms(v[Vu1]), wall - ms(v[ChainBusy]),
                     ms(v[GsBusy]), v[GsBusy] ? wall - ms(v[GsBusy]) : 0.0, ms(v[EeFence]),
                     static_cast<unsigned long long>(fences), ms(v[ChainThrottle]), ms(v[GsWait]), 10);
        if (phasesEnabled())
        {
            char line[512];
            int n = std::snprintf(line, sizeof(line), "[perfphase]");
            for (int i = 0; i < PhaseCount && n > 0 && n < static_cast<int>(sizeof(line)); ++i)
                n += std::snprintf(line + n, sizeof(line) - n, " %s=%.0f", kPhaseNames[i],
                                   ms(g_phaseNs[i].exchange(0u, std::memory_order_relaxed)));
            std::fprintf(stderr, "%s ms/s%c", line, 10);
        }
        s_presents = 0;
        s_last = now;
    }
}
