#pragma once

// Hardware Direct3D 11 GS rasterizer.
//
// GS primitives are rasterized natively on the GPU into upscaled render targets (one per
// frame-buffer base pointer); VRAM transfers, CLUT loads and read-backs are serviced by an
// embedded GSCpuBackend that owns the authoritative 4 MB local-memory image. Render targets
// are synced back to local memory only when the guest reads them through a transfer or as
// a reinterpreted texture.

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_cpu_backend.h"

#include <cstdint>
#include <memory>

class GSD3D11Backend final : public GSRasterBackend
{
public:
    GSD3D11Backend();
    ~GSD3D11Backend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    // the slot for HCTP's native model draws (not used yet)
    void SubmitNative(const GSDrawState &state, const void *job, uint32_t jobSize);
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

    // HCTP frontend: a presentation latches the display frame on the GPU (no GS memory snapshot); the window loop
    // takes it with TakeFreshFrame + PresentLatchedToHost.
    void PresentToMailbox(const GSPresentationRequest &request, GSPresentTicket ticket, GSPresentMailbox &box) override;
    void ResetPresentation(GSPresentMailbox &box, uint64_t seq) override;
    // true once per newly latched frame (window thread)
    static bool TakeFreshFrame();
    // frames the game has presented so far (the FPS counter's GAME line)
    static uint64_t LatchedFrameCount();

    // Host presentation: blits the most recently latched display frame to the swap chain.
    // Returns false when no hardware frame is available (caller falls back to CPU pixels).
    static bool PresentLatchedToHost();
    static bool IsActive();

    struct Impl;

private:
    std::unique_ptr<Impl> m_impl;
};
