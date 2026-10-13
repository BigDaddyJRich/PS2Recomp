#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_gpu.h"

// VENDETTA_GS=gpu: the renderer behind the GS front end when the graphics card draws everything. Nothing is drawn on
// the CPU; the front end still parses every packet (CSR, SIGNAL / FINISH / LABEL and every value the game can read come
// from it), and the few calls that return GS memory are answered by the graphics card renderer after everything sent
// before them (GS memory as of that point in the order).
class GSGpuBackend final : public GSRasterBackend
{
public:
    explicit GSGpuBackend(GSGpuRenderer &gpu) : m_gpu(gpu) {}

    void Initialize(uint8_t *, uint32_t) override {}
    void Reset() override {}

    void Submit(const GSPrimitiveBatch &) override {}
    void LoadClut(const GSTex0Reg &, const GSTexClutReg &) override {}
    void BeginTransfer(const GSTransferCommand &) override {}
    void UploadImage(const uint8_t *, uint32_t) override {}

    void Flush() override {}
    void TextureFlush() override {}
    void Sync(GSSyncReason) override {}
    PresentationFrame Present(const GSPresentationRequest &) override { return {}; }

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) override {}
    void SnapshotVram(std::vector<uint8_t> &out) const override { m_gpu.snapshotVram(out); }
    GSTransferSnapshot GetTransferSnapshot() const override { return {}; }

    // The window shows the graphics card's picture; the CPU picture mailbox only ever says "no picture".
    void PresentToMailbox(const GSPresentationRequest &, GSPresentTicket ticket, GSPresentMailbox &box) override
    {
        ResetPresentation(box, ticket.seq);
    }

private:
    GSGpuRenderer &m_gpu;
};
