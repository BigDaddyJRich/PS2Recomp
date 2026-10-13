#pragma once

#include <atomic>
#include <cstdint>
#include <string>

// What the game (EE) thread and the GS thread are busy with, for a sampling profiler (VENDETTA_HOSTPROF, sampled by a
// separate thread). A Scope marks a stretch of host work; anything unmarked is the game's own recompiled code (game
// track) or command handling (GS track). Plain relaxed loads / stores so the marks cost next to nothing when no one
// samples.
namespace ps2_host_profile
{
    enum Phase : uint8_t
    {
        Game = 0,  // the game's recompiled code (and runtime calls not marked below)
        Iop,       // the emulated IOP (interpreter + its devices)
        Gs,        // GIF packets: GS register writes (not counted below)
        GsLock,    // waiting for the GS state lock (held by presentation / debug readers)
        GsDraw,    // the software rasteriser (a primitive batch)
        GsUpload,  // image uploads into GS memory
        Vu1,       // VU1 microprograms
        Mpeg,      // movie decoding (FFmpeg)
        Wait,      // the EE scheduler waiting for an event (nothing to run)
        GsSubmit,  // handing a primitive / upload to the renderer (queueing it, or a ring record for the GS thread)
        GsWait,    // the game thread blocked on the GS thread (ring full, a reply, a drain, the flip throttle)
        GsPresent, // a presentation: GS memory copied for the screen
        GsCmd,     // the GS thread handling a command (not drawing / uploading / presenting)
        Idle,      // the GS thread / the chain worker with nothing to do
        ChainWait, // the game thread blocked on the chain worker (a fence, the ring full, the throttle)
        ChainCmd,  // the chain worker handling a record (not in VU1 / the GS front end)
        Pace,      // the game thread held at a checkpoint until an event's real-time moment (EE time ahead of the clock)
        Count
    };

    // One phase variable per recording thread. Band workers and other threads record nothing.
    enum Track : uint8_t
    {
        GameTrack = 0,
        GsTrack,
        ChainTrack, // the VIF1 -> VU1 -> GS front end chain worker (VENDETTA_CHAIN_ASYNC=1)
        TrackCount
    };
    inline std::atomic<uint8_t> g_phases[TrackCount] = {{Game}, {Idle}, {Idle}};
    inline std::atomic<uint8_t> &g_phase = g_phases[GameTrack];
    inline thread_local std::atomic<uint8_t> *t_phase = nullptr;
    // The game thread only.
    inline thread_local bool t_recording = false;
    // remembers the OS thread of a track for the code-address sampler (Kernel/HostProfiler.cpp)
    void noteRecordingThread(Track track) noexcept;
    inline void recordThisThread(Track track = GameTrack) noexcept
    {
        t_phase = &g_phases[track];
        if (track == GameTrack)
            t_recording = true;
        noteRecordingThread(track);
    }
    inline void setPhase(Phase phase) noexcept
    {
        if (std::atomic<uint8_t> *slot = t_phase)
            slot->store(phase, std::memory_order_relaxed);
    }

    // VENDETTA_HOSTPROF=2: sample a track's thread's host code address (called by the sampling thread each tick, with
    // the phase it just read), and name / rank the samples since the last report, per track, by function and by
    // source line (Kernel/HostProfiler.cpp).
    bool samplePc(Track track, uint8_t phase) noexcept;
    std::string pcReport(uint32_t topOverall, uint32_t topPerPhase);
    // The stall report (VENDETTA_HOSTPROF): a rough host call stack of a track's thread — its instruction pointer, then
    // the words on its stack that point just after a call instruction in this exe (no unwinding while it is stopped,
    // nothing allocated) — and the names of such a stack, innermost first.
    size_t stackScan(Track track, uint64_t *out, size_t maxOut) noexcept;
    std::string describeStack(const uint64_t *pcs, size_t count);

    struct Scope
    {
        std::atomic<uint8_t> *slot;
        uint8_t previous;
        explicit Scope(Phase phase) noexcept : slot(t_phase), previous(slot ? slot->load(std::memory_order_relaxed) : uint8_t{0})
        {
            if (slot)
                slot->store(phase, std::memory_order_relaxed);
        }
        ~Scope()
        {
            if (slot)
                slot->store(previous, std::memory_order_relaxed);
        }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
    };

    // Work counters for the same report (relaxed; read and reset by the sampler)
    enum Counter : uint8_t
    {
        Sprites = 0,
        SpritePixels,
        Triangles,
        TriangleBoxPixels, // pixels the triangle loop visits (its bounding box)
        TrianglePixels,    // pixels inside the triangle
        Vu1Pairs,          // VU1 instruction pairs executed
        Vu1Cycles,         // VU1 cycles stepped (pairs + stalls)
        GsFlushes,         // draw-queue flushes
        GsParallelFlushes, // ... of them drawn on several cores
        GsHazardFlushes,   // ... forced by a texture / target page overlap
        Vu1CompiledPairs,  // VU1 instruction pairs run by the compiled VU1 code (of Vu1Pairs)
        // why the draw queue was flushed (the causes add up to GsFlushes)
        GsFlushFull,       // the queue reached its limit
        GsFlushClut,       // a palette load that changes the palette, or reads pages a queued primitive draws to
        GsFlushUpload,     // an image upload onto pages a queued primitive reads or draws to
        GsFlushTransfer,   // a local -> local / local -> host transfer
        GsFlushExplicit,   // Flush() (FINISH, a reset)
        GsFlushReadback,   // ReadVram / SnapshotVram / local -> host bytes
        GsFlushPresent,    // a presentation (GS memory copied for the screen)
        GsFlushOther,      // Initialize / Reset / a framebuffer clear / WriteVram / a check
        // the GS thread and presentation
        GsCmds,            // commands handed to the renderer
        GsRingBytes,       // ... their bytes in the GS thread's ring
        GsUploadBytes,     // image bytes uploaded
        GsRingFullWaits,   // the game thread waited for room in the ring
        GsDrains,          // the game thread waited for the GS thread to finish everything
        GsReplyCalls,      // calls that return a value from the renderer (ReadVram, SnapshotVram, local -> host)
        GsPresents,        // presentations
        GsPresentsSkipped, // ... skipped: same picture as the last one (same display registers, no GS memory write)
        GsPresentForced,   // ... taken before the vblank handlers finished (the next vblank came first)
        GsFlips,           // presentations whose display registers changed
        GsThrottleWaits,   // the game thread waited because the GS thread was > 2 flips behind
        GsProducerChanges, // a different thread started handing work to the renderer
        GsThreadExceptions,
        GsFpEnvFixups,     // drawing found the rounding mode not round-to-nearest and set it
        GsVerifyChecks,    // VENDETTA_GS_VERIFY: GS memory compared with the 1-band shadow renderer
        GsVerifyMismatch,
        GsStreamMismatch,  // VENDETTA_GS_VERIFY: the GS thread did not get the commands the game thread sent
        GsProducerMxcsr,   // VENDETTA_GS_VERIFY: commands sent with a rounding mode other than round-to-nearest
        // the chain worker (runtime/ps2_chain.h)
        ChainRecords,      // records handed to the chain (VIF1 / GIF data, VIF1 registers, presentations)
        ChainBytes,        // ... their data bytes
        ChainFences,       // the game thread waited for the chain to finish everything handed to it
        ChainFenceCsr,     // ... of them: a CSR read / write
        ChainFenceOther,   // ... of them: SIGLBLID, GIF_STAT, VU1 memory, a GS call, a D / T kick, SYNC mode
        ChainRingFullWaits,
        ChainThrottleWaits,// the game thread waited because the chain was > CHAIN_LAG vblanks behind
        ChainOwnerViolations, // worker-owned state touched from the wrong thread
        ChainExceptions,
        ChainDtKicks,      // VIF1 kicks with D / T bits enabled (run synchronously)
        ChainKickMxcsr,    // records run in a rounding mode other than round-to-nearest
        ChainCsrExempt,    // CSR reads at an exempt site (no fence: the site uses only CSR.FIELD)
        ChainCsrExemptDiff,// VENDETTA_CHAIN_VERIFY: ... where SIGNAL / FINISH differed from a fenced read (informational)
        ChainVerifyMismatch,// VENDETTA_CHAIN_VERIFY: ... where FIELD differed (must be 0)
        // the VU1 code memory and the decode cache
        Vu1CodeUploads,    // VIF1 MPG uploads
        Vu1CodeUploadsSame,// ... of them with the bytes already in place (nothing changes)
        Vu1DecodeFull,     // the decode cache rebuilt for all of code memory
        Vu1DecodePartial,  // ... re-decoded only the pairs written since
        Vu1DecodeVerifyMismatch, // VENDETTA_VU1_DECODE_VERIFY: the cache differed from a full rebuild (must be 0)
        // the window
        WindowFrames,      // game pictures the window showed
        WindowSkips,       // ... pictures it never showed (a newer one had come before the window took it)
        WindowRepeats,     // window refreshes that showed no new picture
        CounterCount
    };
    inline std::atomic<uint64_t> g_counters[CounterCount]{};
    inline void count(Counter counter, uint64_t amount = 1u) noexcept
    {
        g_counters[counter].fetch_add(amount, std::memory_order_relaxed);
    }
    inline const char *counterName(uint8_t counter) noexcept
    {
        static const char *const names[CounterCount] = {
            "sprites", "sprite-px", "tris", "tri-box-px", "tri-px", "vu1-pairs", "vu1-cycles", "gs-flushes",
            "gs-par-flushes", "gs-hazard-flushes", "vu1-compiled-pairs", "gs-flush-full", "gs-flush-clut",
            "gs-flush-upload", "gs-flush-transfer", "gs-flush-explicit", "gs-flush-readback", "gs-flush-present",
            "gs-flush-other", "gs-cmds", "gs-ring-bytes", "gs-upload-bytes", "gs-ring-full-waits", "gs-drains",
            "gs-reply-calls", "gs-presents", "gs-presents-skipped", "gs-present-forced", "gs-flips",
            "gs-throttle-waits", "gs-producer-changes", "gs-thread-exceptions", "gs-fpenv-fixups", "gs-verify-checks",
            "gs-verify-mismatch", "gs-stream-mismatch", "gs-producer-mxcsr", "chain-records", "chain-bytes",
            "chain-fences", "chain-fence-csr", "chain-fence-other", "chain-ring-full-waits", "chain-throttle-waits",
            "chain-owner-violations", "chain-exceptions", "chain-dt-kicks", "chain-kick-mxcsr", "chain-csr-exempt",
            "chain-csr-exempt-diff", "chain-verify-mismatch", "vu1-code-uploads", "vu1-code-uploads-same",
            "vu1-decode-full", "vu1-decode-partial", "vu1-decode-verify-mismatch", "window-frames", "window-skips",
            "window-repeats"};
        return counter < CounterCount ? names[counter] : "?";
    }

    inline const char *name(uint8_t phase) noexcept
    {
        static const char *const names[Count] = {"game", "iop", "gs", "gs-lock", "gs-draw", "gs-upload", "vu1", "mpeg", "wait",
                                                 "gs-submit", "gs-wait", "gs-present", "gs-cmd", "idle", "chain-wait", "chain-cmd", "pace"};
        return phase < Count ? names[phase] : "?";
    }
}
