#include "runtime/gs/gs_gpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace
{
    // Deliberately a small ABI declaration: the optional DLL owns all Vulkan
    // objects and is loaded only when HCTP_NATIVE_GS_MODULE names it.
    struct NativeGsScanout
    {
        uint64_t pmode, smode1, smode2, dispfb1, display1, dispfb2, display2, bgcolor, vsyncTick;
        uint8_t *rgba;
        uint32_t stride, maxHeight, width, height;
        void (*unlock)(void *);
        void (*relock)(void *);
        void *lockContext;
    };

    struct NativeGsApi
    {
        uint32_t version;
        void *(*create)(const char *, const uint8_t *);
        void (*destroy)(void *);
        void (*reset)(void *);
        void (*gif)(void *, uint32_t, const uint8_t *, uint32_t);
        void (*reg)(void *, uint8_t, uint64_t);
        void (*image)(void *, const uint8_t *, uint32_t);
        void (*flush)(void *);
        void (*wait)(void *);
        void (*read)(void *, uint32_t, uint8_t *, uint32_t);
        void (*write)(void *, uint32_t, const uint8_t *, uint32_t);
        void (*fifo)(void *, uint8_t *, uint32_t);
        void (*clear)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
        int (*scanout)(void *, NativeGsScanout *);
        int (*attachWindow)(void *, void *);
    };

    using NativeGsGetApi = const NativeGsApi *(*)();
    constexpr uint32_t kNativeGsApiVersion = 6u;
    constexpr uint32_t kNativeGsVramBytes = 4u * 1024u * 1024u;
    constexpr uint32_t kScanoutStride = 2048u;
    constexpr uint32_t kScanoutMaxHeight = 2048u;

    class NativeGsModuleRenderer final : public GSGpuRenderer
    {
    public:
        explicit NativeGsModuleRenderer(const char *path) : m_path(path ? path : "") {}
        ~NativeGsModuleRenderer() override { shutdown(); }

        bool start(std::string &whyNot)
        {
            m_module = LoadLibraryA(m_path.c_str());
            if (!m_module)
            {
                whyNot = "could not load HCTP_NATIVE_GS_MODULE (Win32 error " + std::to_string(GetLastError()) + ")";
                return false;
            }
            const auto getApi = reinterpret_cast<NativeGsGetApi>(GetProcAddress(m_module, "black_parallel_gs_api"));
            if (!getApi)
            {
                whyNot = "HCTP_NATIVE_GS_MODULE does not export black_parallel_gs_api";
                shutdown();
                return false;
            }
            m_api = getApi();
            if (!m_api || m_api->version != kNativeGsApiVersion || !m_api->create || !m_api->destroy || !m_api->gif ||
                !m_api->reg || !m_api->reset || !m_api->flush || !m_api->wait || !m_api->read || !m_api->fifo || !m_api->scanout)
            {
                whyNot = "HCTP_NATIVE_GS_MODULE has an incompatible GS module API";
                shutdown();
                return false;
            }
            // Four MiB cannot live on the host stack. This is merely the
            // module's initial guest-visible VRAM image, not a host-resource
            // allocation limit.
            std::vector<uint8_t> initial(kNativeGsVramBytes, 0u);
            m_handle = m_api->create("HCTP", initial.data());
            if (!m_handle)
            {
                whyNot = "HCTP_NATIVE_GS_MODULE could not create its Vulkan renderer";
                shutdown();
                return false;
            }
            m_name = "ps2-native-gs (" + m_path + ")";
            return true;
        }

        const char *deviceName() const override { return m_name.c_str(); }

        bool attachWindow(void *window) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_handle || !m_api || !m_api->attachWindow || !window)
                return false;
            m_directPresentation = m_api->attachWindow(m_handle, window) != 0;
            if (m_directPresentation)
                std::fprintf(stderr, "[native-gs] direct Vulkan window presentation enabled\n");
            return m_directPresentation;
        }

        bool directPresentation() const override { return m_directPresentation; }

        void gif(uint8_t path, const uint8_t *data, uint32_t bytes) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_handle && data && bytes)
                m_api->gif(m_handle, path, data, bytes & ~15u);
        }

        void reg(uint8_t addr, uint64_t value) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_handle)
                m_api->reg(m_handle, addr, value);
        }

        void reset() override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_handle)
                m_api->reset(m_handle);
        }

        void present(const uint64_t (&privRegs)[kPrivRegCount], uint64_t vsyncTick, bool) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_handle)
                return;

            if (m_scanout.size() != size_t(kScanoutStride) * kScanoutMaxHeight * 4u)
                m_scanout.resize(size_t(kScanoutStride) * kScanoutMaxHeight * 4u);

            NativeGsScanout request{};
            request.pmode = privRegs[0];
            request.smode1 = privRegs[1];
            request.smode2 = privRegs[2];
            request.dispfb1 = privRegs[7];
            request.display1 = privRegs[8];
            request.dispfb2 = privRegs[9];
            request.display2 = privRegs[10];
            request.bgcolor = privRegs[14];
            request.vsyncTick = vsyncTick;

            // SMODE2 INT=1 + FFMD=1 (HCTP: smode2=3): every field is its own frame buffer of DH/2 lines (HCTP draws
            // 640x224 fields at FBP 0 and FBP 70; 70 pages = 640*224*4 bytes), shown on all DH output lines. The
            // module derives the scanout height from DISPLAY.DH alone and ignores FFMD and MAGV, so it read 448 rows
            // and the lower half came from the other field buffer / stale VRAM: the lower band on every screen.
            // Same rule as PCSX2 and our CPU backend (gs_cpu_backend.cpp halfHeightFields). Hand the module the
            // real buffer height; the host stretches it to the window. HCTP_NATIVE_GS_FULL_FIELDS=1 restores the
            // old behaviour for A/B.
            static const bool fullFields = std::getenv("HCTP_NATIVE_GS_FULL_FIELDS") != nullptr;
            if (!fullFields)
            {
                const bool halfHeightFields = (request.smode2 & 3u) == 3u;
                auto fixDisplay = [halfHeightFields](uint64_t display) -> uint64_t
                {
                    const uint32_t dh = uint32_t(display >> 44) & 0x7FFu;
                    const uint32_t magv = uint32_t(display >> 27) & 0x3u;
                    uint32_t lines = (dh + 1u) / (magv + 1u);
                    if (halfHeightFields && lines > 1u)
                        lines /= 2u;
                    if (lines == 0u)
                        lines = 1u;
                    display &= ~(uint64_t(0x7FFu) << 44);
                    display &= ~(uint64_t(0x3u) << 27);
                    return display | (uint64_t(lines - 1u) << 44);
                };
                request.display1 = fixDisplay(request.display1);
                request.display2 = fixDisplay(request.display2);
            }

            // The native module's direct path renders to its Vulkan swapchain.
            // A null RGBA pointer is intentional: it avoids getBytes() and the
            // subsequent CPU-to-D3D11 upload in the old mailbox path.
            if (m_directPresentation)
            {
                request.rgba = nullptr;
                request.stride = 0u;
                request.maxHeight = 0u;
                m_api->scanout(m_handle, &request);
                return;
            }

            request.rgba = m_scanout.data();
            request.stride = kScanoutStride * 4u;
            request.maxHeight = kScanoutMaxHeight;

            if (m_api->scanout(m_handle, &request) == 0 || request.width == 0u || request.height == 0u)
                return;

            const uint32_t width = std::min(request.width, kScanoutStride);
            const uint32_t height = std::min(request.height, kScanoutMaxHeight);
            GSGpuFrame &frame = m_frames.writeSlot();
            frame.seq = ++m_sequence;
            frame.width = width;
            frame.height = height;
            frame.internalWidth = width;
            frame.internalHeight = height;
            frame.modeWidth = width;
            frame.modeHeight = height;
            frame.highResolution = width > 1024u || height > 1024u;
            frame.shared = false;
            frame.rgba.resize(size_t(width) * height * 4u);
            // The module uses request.stride as the capacity of a destination
            // scanline.  Its returned image is nevertheless tightly packed:
            // every row begins after request.width * 4 bytes (including the
            // internally scaled scanout case).  Stepping by the capacity here
            // skipped large regions of the module output and produced the
            // horizontal strips/black frame seen by the host presenter.
            for (uint32_t y = 0; y < height; ++y)
                std::memcpy(frame.rgba.data() + size_t(y) * width * 4u,
                            m_scanout.data() + size_t(y) * width * 4u,
                            size_t(width) * 4u);
            m_lastWidth = width;
            m_lastHeight = height;
            m_frames.publish();
        }

        void snapshotVram(std::vector<uint8_t> &out) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            out.assign(kNativeGsVramBytes, 0u);
            if (m_handle)
            {
                m_api->wait(m_handle);
                m_api->read(m_handle, 0u, out.data(), static_cast<uint32_t>(out.size()));
            }
        }

        uint32_t readTransfer(uint8_t *dst, uint32_t maxBytes) override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_handle || !dst || maxBytes == 0u)
                return 0u;
            const uint32_t bytes = maxBytes & ~15u;
            if (!bytes)
                return 0u;
            m_api->fifo(m_handle, dst, bytes);
            // The menu transition makes several GS-local-to-EE-memory
            // transfers.  Keep an opt-in, bounded signature trace so the
            // native module's materialized pixels can be matched against a
            // reference run without retaining a huge binary dump in the log.
            static const bool traceReadback = std::getenv("HCTP_NATIVE_GS_READBACK_DIAG") != nullptr;
            static uint32_t traceCount = 0u;
            if (traceReadback && traceCount++ < 64u)
            {
                uint64_t hash = 1469598103934665603ull;
                for (uint32_t i = 0; i < bytes; ++i)
                {
                    hash ^= dst[i];
                    hash *= 1099511628211ull;
                }
                const uint32_t head = bytes >= 4u ? uint32_t(dst[0]) | uint32_t(dst[1]) << 8u |
                                                       uint32_t(dst[2]) << 16u | uint32_t(dst[3]) << 24u
                                                  : 0u;
                std::fprintf(stderr, "[native-gs-readback] #%u bytes=%u fnv=%016llx head=%08x\n", traceCount, bytes,
                             static_cast<unsigned long long>(hash), head);
            }
            return bytes;
        }

        void kick() override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_handle)
                m_api->flush(m_handle);
        }

        bool waitPresentBacklog(uint32_t, const std::atomic<bool> &stop) override { return !stop.load(std::memory_order_acquire); }
        void wake() override {}
        const GSGpuFrame *takeFrame(uint64_t &lastSeq) override { return m_frames.take(lastSeq); }
        void setOutputSize(uint32_t, uint32_t) override {}
        void setSuperSampling(uint32_t) override {}
        void setHighResolution(bool) override {}
        void setFsr(bool) override {}
        void setSharpness(float) override {}

        GSGpuStatus status() const override
        {
            GSGpuStatus status{};
            status.pictureWidth = m_lastWidth;
            status.pictureHeight = m_lastHeight;
            return status;
        }

        void shutdown() override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_handle && m_api)
                m_api->destroy(m_handle);
            m_handle = nullptr;
            m_api = nullptr;
            if (m_module)
                FreeLibrary(m_module);
            m_module = nullptr;
        }

    private:
        std::string m_path;
        std::string m_name = "ps2-native-gs";
        HMODULE m_module = nullptr;
        const NativeGsApi *m_api = nullptr;
        void *m_handle = nullptr;
        mutable std::mutex m_mutex;
        std::vector<uint8_t> m_scanout;
        GSGpuFrameMailbox m_frames;
        uint64_t m_sequence = 0u;
        uint32_t m_lastWidth = 0u;
        uint32_t m_lastHeight = 0u;
        bool m_directPresentation = false;
    };
}

std::unique_ptr<GSGpuRenderer> makeNativeGsModuleRenderer(std::string &whyNot)
{
    whyNot.clear();
    const char *path = std::getenv("HCTP_NATIVE_GS_MODULE");
    if (!path || !*path)
        return nullptr;
    auto renderer = std::make_unique<NativeGsModuleRenderer>(path);
    if (!renderer->start(whyNot))
        return nullptr;
    return renderer;
}

#else

std::unique_ptr<GSGpuRenderer> makeNativeGsModuleRenderer(std::string &whyNot)
{
    if (const char *path = std::getenv("HCTP_NATIVE_GS_MODULE"); path && *path)
        whyNot = "HCTP_NATIVE_GS_MODULE is supported only by the Windows host";
    else
        whyNot.clear();
    return nullptr;
}

#endif
