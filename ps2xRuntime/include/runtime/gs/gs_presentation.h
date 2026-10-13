#pragma once

#include "runtime/gs/gs_types.h"

#include <atomic>
#include <cstdint>
#include <vector>

// The display registers (privileged GS registers) a presentation is built from: all 15 low ones, in GS order (the
// CPU renderer reads 8 of them; the graphics card renderer's video circuit reads them all).
struct GSDisplayRegs
{
    uint64_t pmode = 0;
    uint64_t smode1 = 0;
    uint64_t smode2 = 0;
    uint64_t srfsh = 0;
    uint64_t synch1 = 0;
    uint64_t synch2 = 0;
    uint64_t syncv = 0;
    uint64_t dispfb1 = 0;
    uint64_t display1 = 0;
    uint64_t dispfb2 = 0;
    uint64_t display2 = 0;
    uint64_t extbuf = 0;
    uint64_t extdata = 0;
    uint64_t extwrite = 0;
    uint64_t bgcolor = 0;

    bool operator==(const GSDisplayRegs &other) const noexcept
    {
        return pmode == other.pmode && smode1 == other.smode1 && smode2 == other.smode2 && srfsh == other.srfsh &&
               synch1 == other.synch1 && synch2 == other.synch2 && syncv == other.syncv && dispfb1 == other.dispfb1 &&
               display1 == other.display1 && dispfb2 == other.dispfb2 && display2 == other.display2 &&
               extbuf == other.extbuf && extdata == other.extdata && extwrite == other.extwrite &&
               bgcolor == other.bgcolor;
    }
    bool operator!=(const GSDisplayRegs &other) const noexcept { return !(*this == other); }
};

struct GSPresentTicket
{
    uint64_t seq = 0;  // unique, increasing
    bool flip = false; // the display registers changed since the last presentation
};

// One presented picture: a copy of GS memory as of a point in the command order, plus how to read the screen from it.
struct GSPresentSlot
{
    uint64_t seq = 0;
    GSPresentationRequest request{};
    std::vector<uint8_t> vram;
    bool valid = false;
};

// A lock-free triple buffer between the one thread that presents (the renderer's owner: the GS thread, or the game
// thread when drawing inline) and the one thread that shows pictures (the main thread). The writer fills writeSlot()
// and publishes it; the reader takes the newest published slot, which stays untouched until its next acquire().
class GSPresentMailbox
{
public:
    // writer
    GSPresentSlot &writeSlot() noexcept { return m_slot[m_back]; }
    void publish() noexcept
    {
        const uint64_t seq = m_slot[m_back].seq;
        m_back = m_middle.exchange(m_back | kFresh, std::memory_order_acq_rel) & kIndexMask;
        m_seq.store(seq, std::memory_order_release);
    }

    // reader: the newest published slot (or the last one taken when nothing new was published)
    const GSPresentSlot *acquire() noexcept
    {
        if ((m_middle.load(std::memory_order_acquire) & kFresh) != 0u)
            m_front = m_middle.exchange(m_front, std::memory_order_acq_rel) & kIndexMask;
        return &m_slot[m_front];
    }

    // any thread: the sequence number of the newest published slot
    uint64_t sequence() const noexcept { return m_seq.load(std::memory_order_acquire); }

private:
    static constexpr uint32_t kFresh = 4u;
    static constexpr uint32_t kIndexMask = 3u;
    GSPresentSlot m_slot[3];
    std::atomic<uint32_t> m_middle{1u};
    uint32_t m_back = 0u;  // writer only
    uint32_t m_front = 2u; // reader only
    std::atomic<uint64_t> m_seq{0u};
};
