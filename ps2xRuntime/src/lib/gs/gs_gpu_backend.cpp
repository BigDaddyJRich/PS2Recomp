#include "runtime/gs/gs_gpu_backend.h"
#include "runtime/gs/gs_cpu_backend.h"

bool GSGpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    (void)rgba;
    return GSCpuBackend::ClearFramebufferWritesPixels(context, true);
}

uint32_t GSGpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    return m_gpu.readTransfer(dst, maxBytes);
}

uint32_t GSGpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    // Rare (debugging, other games' stand-ins): a copy of GS memory from the graphics card, read the CPU renderer's way.
    std::vector<uint8_t> vram;
    m_gpu.snapshotVram(vram);
    if (vram.empty())
        return 0u;
    GSCpuBackend reader(1u);
    reader.Initialize(vram.data(), static_cast<uint32_t>(vram.size()));
    const uint32_t value = reader.ReadVram(psm, base, bw, x, y);
    reader.Initialize(nullptr, 0u);
    return value;
}
