#pragma once

#include "runtime/gs/gs_presentation.h"
#include "runtime/gs/gs_types.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

// The renderer behind the GS front end. The front end (register state, GIF parsing, CSR / SIGNAL / FINISH) calls it in
// order from one thread at a time (the "producer": whoever holds GS::m_stateMutex). A renderer may run the calls later
// on another thread (GSThreadedBackend), but it must give exactly the result of running them in this order:
// - calls that return something (ReadVram, SnapshotVram, ConsumeLocalToHostBytes, Initialize's exception) see every
//   earlier call done; ClearFramebuffer's result depends only on its arguments and whether VRAM is set;
// - Sync(Reset / DebugReadback / LocalToHost) returns once every earlier call is done;
// - PresentToMailbox copies GS memory as of its place in the order.
class GSRasterBackend
{
public:
    virtual ~GSRasterBackend() = default;

    virtual void Initialize(uint8_t *vram, uint32_t vramSize) = 0;
    virtual void Reset() = 0;

    virtual void Submit(const GSPrimitiveBatch &batch) = 0;
    virtual void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) = 0;

    virtual void BeginTransfer(const GSTransferCommand &command) = 0;
    virtual void UploadImage(const uint8_t *data, uint32_t sizeBytes) = 0;

    virtual void Flush() = 0;
    virtual void TextureFlush() = 0;
    virtual void Sync(GSSyncReason reason) = 0;
    virtual PresentationFrame Present(const GSPresentationRequest &request) = 0;

    virtual bool ClearFramebuffer(const GSContext &context, uint32_t rgba) = 0;
    virtual uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) = 0;

    virtual uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const = 0;
    virtual void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) = 0;
    virtual void SnapshotVram(std::vector<uint8_t> &out) const = 0;
    virtual GSTransferSnapshot GetTransferSnapshot() const = 0;

    // Draw everything submitted so far, copy GS memory as of this point in the call order into box.writeSlot() and
    // publish it.
    virtual void PresentToMailbox(const GSPresentationRequest &request, GSPresentTicket ticket, GSPresentMailbox &box)
    {
        GSPresentSlot &slot = box.writeSlot();
        SnapshotVram(slot.vram);
        slot.request = request;
        slot.seq = ticket.seq;
        slot.valid = !slot.vram.empty();
        box.publish();
    }
    // Publish "no picture" (after a reset).
    virtual void ResetPresentation(GSPresentMailbox &box, uint64_t seq)
    {
        GSPresentSlot &slot = box.writeSlot();
        slot.valid = false;
        slot.seq = seq;
        box.publish();
    }
    // VENDETTA_GS_VERIFY: compare GS memory (8 KiB pages, CRC) with what the 1-band shadow renderer drew, at this
    // point in the call order.
    virtual void VerifyPages(const std::array<uint64_t, 512> &expected, uint64_t check, uint64_t streamHash)
    {
        (void)expected;
        (void)check;
        (void)streamHash;
    }
    // The end of a front-end entry point: queued work may start now.
    virtual void Kick() {}
    // The flip throttle: wait (holding no lock) while more than maxAhead flips are still to be drawn; false = stopped.
    virtual bool WaitFlipBacklog(uint32_t maxAhead, const std::atomic<bool> &stop)
    {
        (void)maxAhead;
        (void)stop;
        return true;
    }
    // requestStop: release a throttle wait.
    virtual void WakeWaiters() {}
    // Stop a renderer thread (idempotent; after the game thread has ended).
    virtual void Shutdown() {}
};
