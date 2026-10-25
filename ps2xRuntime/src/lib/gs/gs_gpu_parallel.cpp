// The GS on the graphics card: paraLLEl-GS (Arntzen Software AS, LGPL-3.0+) on Granite's Vulkan backend.
//
// The GS front end hands over the raw GS input in its own order (GIF data per path, register writes, presentations);
// it goes into a single-producer / single-consumer ring and one thread, GpuGsThread, feeds it to paraLLEl-GS. That
// thread owns every Vulkan object: it creates the instance, the device and the GS interface, runs all the work and
// destroys them before it ends (Granite and paraLLEl-GS are single-threaded APIs). At each presentation it scans out
// the picture, copies it back and publishes it to the main thread's mailbox.
// Nothing here includes raylib (windows.h and raylib share names such as CloseWindow / Rectangle / LoadImage).

#include "runtime/gs/gs_gpu.h"

#include "context.hpp"
#include "device.hpp"
#include "gs_interface.hpp"
#include "logging.hpp"
#include "muglm/muglm_impl.hpp"
#include "shaders/slangmosh.hpp"
#include "gs_vulkan_shared.h"
#include "gs_vulkan_present.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
    std::atomic<bool> g_sharedStopped{false};
}

void stopSharedPictures() { g_sharedStopped.store(true, std::memory_order_release); }

bool vulkanPresentWanted()
{
    static const bool wanted = []
    {
        const char *present = std::getenv("VENDETTA_PRESENT");
        const char *renderer = std::getenv("VENDETTA_GS");
        const std::string mode = renderer ? renderer : "";
        if (mode == "cpu")
            return false;
        if (present && *present)
            return std::string(present) == "vulkan";
        return mode.empty() || mode == "gpu";
    }();
    return wanted;
}

namespace
{
    // Granite's log lines, prefixed; its information lines (layers, extensions, devices) only with VENDETTA_GPU_LOG=1.
    class GpuLog final : public Util::LoggingInterface
    {
    public:
        explicit GpuLog(bool info) : m_info(info) {}

        bool log(const char *tag, const char *fmt, va_list va) override
        {
            if (!m_info && std::strcmp(tag, "[INFO]: ") == 0)
                return true;
            char line[1024];
            std::vsnprintf(line, sizeof(line), fmt, va);
            std::fprintf(stderr, "[gpu-gs] %s%s", tag, line);
            return true;
        }

    private:
        bool m_info;
    };

    std::string lowerCase(std::string text)
    {
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    // VENDETTA_GPU: a device index (as Vulkan lists them) or part of its name, e.g. "nvidia" / "intel".
    VkPhysicalDevice pickPhysicalDevice(VkInstance instance, const std::string &selector)
    {
        if (selector.empty())
            return VK_NULL_HANDLE; // the library's choice: a discrete GPU first
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0)
            return VK_NULL_HANDLE;
        std::vector<VkPhysicalDevice> gpus(count);
        if (vkEnumeratePhysicalDevices(instance, &count, gpus.data()) != VK_SUCCESS)
            return VK_NULL_HANDLE;

        const bool numeric = std::all_of(selector.begin(), selector.end(), [](unsigned char c)
                                         { return std::isdigit(c) != 0; });
        if (numeric)
        {
            const unsigned long index = std::strtoul(selector.c_str(), nullptr, 10);
            if (index < count)
                return gpus[index];
        }
        else
        {
            const std::string want = lowerCase(selector);
            for (VkPhysicalDevice gpu : gpus)
            {
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(gpu, &props);
                if (lowerCase(props.deviceName).find(want) != std::string::npos)
                    return gpu;
            }
        }
        std::cerr << "[gpu-gs] VENDETTA_GPU=" << selector << " matches no Vulkan device; using the default" << std::endl;
        return VK_NULL_HANDLE;
    }

    // AMD FSR 1's constants for the EASU upscale and the RCAS sharpen passes paraLLEl-GS ships (shaders upscale_* and
    // sharpen_*), as its PCSX2 integration computes them.
    void fsrEasuConstants(float *con0, float *con1, float *con2, float *con3, float inputWidth, float inputHeight,
                          float outputWidth, float outputHeight)
    {
        con0[0] = inputWidth / outputWidth;
        con0[1] = inputHeight / outputHeight;
        con0[2] = 0.5f * inputWidth / outputWidth - 0.5f;
        con0[3] = 0.5f * inputHeight / outputHeight - 0.5f;
        con1[0] = 1.0f / inputWidth;
        con1[1] = 1.0f / inputHeight;
        con1[2] = 1.0f / inputWidth;
        con1[3] = -1.0f / inputHeight;
        con2[0] = -1.0f / inputWidth;
        con2[1] = 2.0f / inputHeight;
        con2[2] = 1.0f / inputWidth;
        con2[3] = 2.0f / inputHeight;
        con3[0] = 0.0f;
        con3[1] = 4.0f / inputHeight;
        con3[2] = con3[3] = 0.0f;
    }

    // sharpness in stops: 0 = the most, each 1 halves it
    void fsrRcasConstants(float *con, float sharpness)
    {
        sharpness = std::exp2(-sharpness);
        const uint32_t half = muglm::floatToHalf(sharpness);
        con[0] = sharpness;
        const uint32_t halves = half | (half << 16);
        std::memcpy(&con[1], &halves, sizeof(halves));
        con[2] = 0.0f;
        con[3] = 0.0f;
    }

    constexpr uint64_t align16(uint64_t bytes) { return (bytes + 15u) & ~uint64_t{15u}; }
    constexpr uint32_t kMaxGifChunk = 256u * 1024u;         // larger GIF transfers go in pieces (paths keep their tag)
    constexpr uint64_t kPublishBytes = 16u * 1024u;         // hand records over at least this often
    constexpr auto kSpin = std::chrono::microseconds(20);   // before sleeping

    inline void cpuRelax() { _mm_pause(); }

    class ParallelGsRenderer final : public GSGpuRenderer
    {
        using Clock = std::chrono::steady_clock;

    public:
        explicit ParallelGsRenderer(const GSGpuOptions &options) : m_options(options)
        {
            m_profile = std::getenv("VENDETTA_HOSTPROF") != nullptr;
            m_wantSsaa.store(options.superSampling);
            m_wantHires.store(options.highResolutionScanout);
            m_wantFsr.store(options.fsr);
            m_wantSharpness.store(options.sharpness);
            size_t size = size_t{1} << 20u;
            while (size < options.ringBytes)
                size <<= 1u;
            m_size = size;
            m_mask = size - 1u;
            m_ringStorage.reset(new uint8_t[size + 64u]);
            m_ring = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(m_ringStorage.get()) + 63u) & ~uintptr_t{63u});
        }

        ~ParallelGsRenderer() override { shutdown(); }

        // Starts the thread and waits until it has a device (or failed). False → whyNot.
        bool start(std::string &whyNot)
        {
            std::future<std::string> ready = m_ready.get_future();
            m_thread = std::thread([this]
                                   { threadMain(); });
            std::string error = ready.get();
            if (!error.empty())
            {
                whyNot = error;
                shutdown();
                return false;
            }
            return true;
        }

        const char *deviceName() const override { return m_deviceName.c_str(); }

        // ---- producer ----------------------------------------------------------------------------------------

        void gif(uint8_t path, const uint8_t *data, uint32_t bytes) override
        {
            bytes &= ~15u;
            while (bytes != 0u)
            {
                const uint32_t chunk = std::min(bytes, kMaxGifChunk);
                uint8_t *dst = reserve(Cmd::Gif, chunk);
                reinterpret_cast<Header *>(dst)[-1].flags = path;
                std::memcpy(dst, data, chunk);
                commit();
                data += chunk;
                bytes -= chunk;
            }
        }

        void reg(uint8_t addr, uint64_t value) override
        {
            uint8_t *dst = reserve(Cmd::Reg, 0u);
            Header &header = reinterpret_cast<Header *>(dst)[-1];
            header.flags = addr;
            header.aux = value;
            commit();
        }

        void reset() override
        {
            reserve(Cmd::Reset, 0u);
            commit();
        }

        void present(const uint64_t (&privRegs)[kPrivRegCount], uint64_t vsyncTick, bool flip) override
        {
            PresentRecord record{};
            std::memcpy(record.regs, privRegs, sizeof(record.regs));
            record.vsyncTick = vsyncTick;
            record.flip = flip ? 1u : 0u;
            uint8_t *dst = reserve(Cmd::Present, sizeof(record));
            std::memcpy(dst, &record, sizeof(record));
            m_presentsIssued.fetch_add(1u, std::memory_order_relaxed);
            commit();
            publish();
        }

        void snapshotVram(std::vector<uint8_t> &out) override
        {
            CallRecord call{};
            call.kind = CallRecord::Snapshot;
            call.out = &out;
            runCall(call);
        }

        uint32_t readTransfer(uint8_t *dst, uint32_t maxBytes) override
        {
            CallRecord call{};
            call.kind = CallRecord::ReadTransfer;
            call.dst = dst;
            call.bytes = maxBytes;
            runCall(call);
            return call.result;
        }

        void kick() override
        {
            if (m_options.sync)
                drain();
            else
                publish();
        }

        // ---- game thread ---------------------------------------------------------------------------------------

        bool waitPresentBacklog(uint32_t maxAhead, const std::atomic<bool> &stop) override
        {
            auto caughtUp = [&]()
            {
                return m_presentsIssued.load(std::memory_order_relaxed) -
                           m_presentsDone.load(std::memory_order_acquire) <=
                       maxAhead;
            };
            if (caughtUp())
                return true;
            waitProgress("the graphics card to draw the frames it was given", [&]()
                         { return stop.load(std::memory_order_acquire) || caughtUp(); });
            return !stop.load(std::memory_order_acquire);
        }

        void wake() override
        {
            {
                std::lock_guard<std::mutex> lock(m_progressMutex);
            }
            m_progressCv.notify_all();
        }

        // ---- main thread ---------------------------------------------------------------------------------------

        const GSGpuFrame *takeFrame(uint64_t &lastSeq) override { return m_frames.take(lastSeq); }

        void setOutputSize(uint32_t width, uint32_t height) override
        {
            m_outputSize.store((uint64_t{width} << 32u) | height, std::memory_order_relaxed);
        }

        void setSharedReading(int image) override { m_frames.sharedReading.store(image, std::memory_order_release); }

        void setSuperSampling(uint32_t rate) override
        {
            if (rate == 1u || rate == 2u || rate == 4u || rate == 8u || rate == 16u)
                m_wantSsaa.store(rate, std::memory_order_relaxed);
        }
        void setHighResolution(bool on) override { m_wantHires.store(on, std::memory_order_relaxed); }
        void setFsr(bool on) override { m_wantFsr.store(on, std::memory_order_relaxed); }
        void setSharpness(float stops) override
        {
            m_wantSharpness.store(std::clamp(stops, 0.0f, 2.0f), std::memory_order_relaxed);
        }

        GSGpuStatus status() const override
        {
            std::lock_guard<std::mutex> lock(m_statusMutex);
            return m_status;
        }

        void shutdown() override
        {
            if (m_shutDown)
                return;
            m_shutDown = true;
            m_stop.store(true, std::memory_order_seq_cst);
            m_wakeEpoch.fetch_add(1u, std::memory_order_seq_cst);
            m_wakeEpoch.notify_one();
            if (m_thread.joinable())
                m_thread.join();
            m_readPos.store(m_localWrite, std::memory_order_release);
            m_presentsDone.store(m_presentsIssued.load(std::memory_order_relaxed), std::memory_order_release);
            wake();
        }

    private:
        enum class Cmd : uint16_t
        {
            Wrap = 0, // fills the rest of the ring; the next record starts at its beginning
            Gif,      // flags = path, payload = whole quadwords
            Reg,      // flags = register address, aux = value
            Reset,
            Present,  // payload = PresentRecord
            Call,     // payload = a CallRecord * (on the waiting producer's stack)
        };
        struct alignas(16) Header
        {
            uint16_t cmd = 0;
            uint16_t flags = 0;
            uint32_t bytes = 0; // payload bytes (the record is padded to 16)
            uint64_t aux = 0;
        };
        static_assert(sizeof(Header) == 16u);
        struct CallRecord
        {
            enum Kind : uint32_t
            {
                Snapshot,
                ReadTransfer,
            };
            Kind kind = Snapshot;
            std::vector<uint8_t> *out = nullptr;
            uint8_t *dst = nullptr;
            uint32_t bytes = 0u;
            uint32_t result = 0u;
            std::atomic<bool> done{false};
        };
        struct PresentRecord
        {
            uint64_t regs[kPrivRegCount];
            uint64_t vsyncTick;
            uint32_t flip;
            uint32_t pad;
        };

        // ---- the ring, producer side ----------------------------------------------------------------------------

        void runCall(CallRecord &call)
        {
            CallRecord *pointer = &call;
            uint8_t *dst = reserve(Cmd::Call, sizeof(pointer));
            std::memcpy(dst, &pointer, sizeof(pointer));
            commit();
            publish();
            waitProgress("the graphics card to read GS memory back", [&]()
                         { return m_stop.load(std::memory_order_relaxed) || call.done.load(std::memory_order_acquire); });
        }

        uint8_t *reserve(Cmd cmd, uint32_t payloadBytes)
        {
            const uint64_t recordBytes = sizeof(Header) + align16(payloadBytes);
            const uint64_t offset = m_localWrite & m_mask;
            const uint64_t tail = m_size - offset;
            const bool wrap = recordBytes > tail;
            const uint64_t needed = recordBytes + (wrap ? tail : 0u);
            if (m_size - (m_localWrite - m_cachedRead) < needed)
            {
                m_cachedRead = m_readPos.load(std::memory_order_acquire);
                if (m_size - (m_localWrite - m_cachedRead) < needed)
                {
                    publish();
                    waitProgress("room in the graphics card's ring", [&]()
                                 {
                                     m_cachedRead = m_readPos.load(std::memory_order_acquire);
                                     return m_stop.load(std::memory_order_relaxed) ||
                                            m_size - (m_localWrite - m_cachedRead) >= needed; });
                }
            }
            if (wrap)
            {
                Header *filler = reinterpret_cast<Header *>(m_ring + offset);
                *filler = Header{};
                filler->cmd = static_cast<uint16_t>(Cmd::Wrap);
                filler->bytes = static_cast<uint32_t>(tail - sizeof(Header));
                m_localWrite += tail;
            }
            Header *header = reinterpret_cast<Header *>(m_ring + (m_localWrite & m_mask));
            *header = Header{};
            header->cmd = static_cast<uint16_t>(cmd);
            header->bytes = payloadBytes;
            m_pendingRecordBytes = recordBytes;
            return reinterpret_cast<uint8_t *>(header + 1);
        }

        void commit()
        {
            m_localWrite += m_pendingRecordBytes;
            if (m_options.sync)
                drain();
            else if (m_localWrite - m_published >= kPublishBytes)
                publish();
        }

        void publish()
        {
            if (m_published == m_localWrite)
                return;
            m_published = m_localWrite;
            m_writePos.store(m_localWrite, std::memory_order_seq_cst);
            if (m_consumerSleeping.load(std::memory_order_seq_cst) != 0u)
            {
                m_wakeEpoch.fetch_add(1u, std::memory_order_seq_cst);
                m_wakeEpoch.notify_one();
            }
        }

        void drain()
        {
            publish();
            if (m_readPos.load(std::memory_order_acquire) >= m_localWrite)
                return;
            waitProgress("the graphics card thread to finish its commands", [&]()
                         { return m_stop.load(std::memory_order_relaxed) ||
                                  m_readPos.load(std::memory_order_acquire) >= m_localWrite; });
        }

        template <typename Pred>
        void waitProgress(const char *what, Pred pred)
        {
            const auto start = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - start < kSpin)
            {
                if (pred())
                    return;
                cpuRelax();
            }
            bool reported = false;
            std::unique_lock<std::mutex> lock(m_progressMutex);
            m_producerWaiting.fetch_add(1u, std::memory_order_seq_cst);
            while (!pred())
            {
                if (m_progressCv.wait_for(lock, std::chrono::milliseconds(100)) == std::cv_status::timeout && !reported &&
                    std::chrono::steady_clock::now() - start > std::chrono::seconds(2))
                {
                    reported = true;
                    std::fprintf(stderr, "[gpu-gs] waited over 2 s for %s: read=%llu write=%llu%c", what,
                                 static_cast<unsigned long long>(m_readPos.load(std::memory_order_relaxed)),
                                 static_cast<unsigned long long>(m_localWrite), 10);
                }
            }
            m_producerWaiting.fetch_sub(1u, std::memory_order_seq_cst);
        }

        // ---- the graphics card thread -------------------------------------------------------------------------

        void publishProgress(uint64_t readPos)
        {
            m_readPos.store(readPos, std::memory_order_seq_cst);
            if (m_producerWaiting.load(std::memory_order_seq_cst) != 0u)
                wake();
        }

        void threadMain()
        {
            SetThreadDescription(GetCurrentThread(), L"GpuGsThread");
            GpuLog logger(m_options.libraryLog);
            Util::set_thread_logging_interface(&logger);

            bool reported = false;
            auto fail = [&](const std::string &why)
            {
                if (!reported)
                {
                    reported = true;
                    m_ready.set_value(why);
                }
            };

            try
            {
                if (!Vulkan::Context::init_loader(nullptr))
                    return fail("no Vulkan loader (vulkan-1.dll)");

                // Closed in reverse order: the GS interface, the device (both explicitly, below), then the context.
                Vulkan::Context context;
                context.set_num_thread_indices(1);
                const Vulkan::ContextCreationFlags flags = Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT |
                                                           Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_HEAP_BIT |
                                                           Vulkan::CONTEXT_CREATION_ENABLE_DESCRIPTOR_BUFFER_BIT;
                if (!context.init_instance(nullptr, 0, flags))
                    return fail("no Vulkan instance");
                const VkPhysicalDevice chosen = pickPhysicalDevice(context.get_instance(), m_options.gpuSelector);
                if (!context.init_device(chosen, VK_NULL_HANDLE, nullptr, 0, flags))
                    return fail("no usable Vulkan device");

                auto deviceHolder = std::make_unique<Vulkan::Device>();
                Vulkan::Device &device = *deviceHolder;
                device.set_context(context);
                device.init_frame_contexts(12);

                auto gsHolder = std::make_unique<ParallelGS::GSInterface>();
                ParallelGS::GSInterface &gs = *gsHolder;
                ParallelGS::GSOptions gsOptions{};
                gsOptions.super_sampling = static_cast<ParallelGS::SuperSampling>(m_options.superSampling);
                gsOptions.ordered_super_sampling = m_options.orderedGrid;
                gsOptions.super_sampled_textures = m_options.superSampledTextures;
                gsOptions.dynamic_super_sampling = true;
                if (!gs.init(&device, gsOptions))
                    return fail(std::string("paraLLEl-GS could not start on ") + context.get_gpu_props().deviceName +
                                " (it needs descriptor indexing, timeline semaphores, 8/16-bit storage and subgroup "
                                "size control)");

                {
                    // 2-D detail: VENDETTA_UPSCALE_SPRITES=1 (settings.toml graphics.sharp_2d) super-samples sprites
                    // (HUD, text, menus) like the 3-D; at 16x the picture is scanned out at 4x (each super-sample its
                    // own pixel; VENDETTA_HIRES4=0: 2x as before). VENDETTA_PROMOTE=1 (a test): show a final blit's
                    // source instead (paraLLEl-GS's backbuffer promotion).
                    auto on = [](const char *name) { const char *v = std::getenv(name); return v && *v && *v != '0'; };
                    ParallelGS::Hacks hacks{};
                    hacks.upscale_sprites = on("VENDETTA_UPSCALE_SPRITES");
                    hacks.backbuffer_promotion = on("VENDETTA_PROMOTE");
                    gs.set_hacks(hacks);
                    const char *hires4 = std::getenv("VENDETTA_HIRES4");
                    m_quadScanout = !(hires4 && *hires4 == '0');
                    std::cerr << "[gpu-gs] 2-D: upscale-sprites=" << hacks.upscale_sprites << " promote="
                              << hacks.backbuffer_promotion << " hires4=" << m_quadScanout << std::endl;
                }
                {
                    Vulkan::ResourceLayout layout;
                    ParallelGS::Shaders<> suite(device, layout, 0);
                    m_upscaleProgram = device.request_program(suite.upscale_vert, suite.upscale_frag);
                    m_sharpenProgram = device.request_program(suite.sharpen_vert, suite.sharpen_frag);
                }
                m_currentSsaa = m_options.superSampling;

                const VkPhysicalDeviceProperties &props = context.get_gpu_props();
                m_deviceName = props.deviceName;
                std::cerr << "[gpu-gs] device=\"" << props.deviceName << "\" vulkan=" << VK_VERSION_MAJOR(props.apiVersion)
                          << '.' << VK_VERSION_MINOR(props.apiVersion) << '.' << VK_VERSION_PATCH(props.apiVersion)
                          << " ssaa=" << m_options.superSampling << (m_options.orderedGrid ? " ordered" : " rotated")
                          << " ssaa-textures=" << (m_options.superSampledTextures ? 1 : 0)
                          << " hires=" << (m_options.highResolutionScanout ? 1 : 0)
                          << " external-memory=" << (context.get_enabled_device_features().supports_external ? "yes" : "no")
                          << std::endl;
                reported = true;
                m_ready.set_value(std::string());

                {
                    // shown through Vulkan (vulkanPresentWanted): pictures shared with the window's own Vulkan
                    // device (no read back)
                    m_sharedPresent = vulkanPresentWanted() && context.get_enabled_device_features().supports_external;
                    if (m_sharedPresent)
                    {
                        VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                        props2.pNext = &ids;
                        vkGetPhysicalDeviceProperties2(context.get_gpu(), &props2);
                        std::memcpy(m_deviceUuid, ids.deviceUUID, sizeof(m_deviceUuid));
                        std::cerr << "[gpu-gs] pictures shared with the window's device" << std::endl;
                    }
                }

                m_device = &device;
                m_gs = &gs;
                consume();
                const auto stopStart = Clock::now();
                auto stepMs = [&]()
                { return std::chrono::duration<double, std::milli>(Clock::now() - stopStart).count(); };
                m_gs = nullptr;
                m_device = nullptr;
                m_lastScanout = {};
                m_readback.reset();
                m_fsrUpscaled.reset();
                m_fsrSharpened.reset();
                for (auto &image : m_sharedImages)
                    image.reset();
                m_sharedTimeline.reset();
                closeSharedHandles(m_sharedHandles);
                closeSharedHandles(m_sharedOldHandles);
                if (m_sharedTimelineHandle)
                    CloseHandle(m_sharedTimelineHandle);
                m_sharedTimelineHandle = nullptr;
                // The GS interface first: it submits the command buffers it holds (waiting for the device while it
                // holds one never ends), then the device.
                gsHolder.reset();
                device.wait_idle();
                deviceHolder.reset();
                if (m_profile)
                    std::fprintf(stderr, "[gpu-gs] closed in %.1f ms%c", stepMs(), 10);
            }
            catch (const std::exception &e)
            {
                if (!reported)
                    fail(std::string("Vulkan error: ") + e.what());
                else
                    std::cerr << "[gpu-gs] the graphics card thread stopped: " << e.what() << std::endl;
            }
            Util::set_thread_logging_interface(nullptr);
        }

        void consume()
        {
            uint64_t read = m_readPos.load(std::memory_order_relaxed);
            for (;;)
            {
                const uint64_t write = m_writePos.load(std::memory_order_acquire);
                if (read == write)
                {
                    publishProgress(read);
                    if (m_stop.load(std::memory_order_acquire))
                        return;
                    const auto spinStart = std::chrono::steady_clock::now();
                    while (m_writePos.load(std::memory_order_acquire) == read && !m_stop.load(std::memory_order_relaxed) &&
                           std::chrono::steady_clock::now() - spinStart < kSpin)
                        cpuRelax();
                    if (m_writePos.load(std::memory_order_acquire) == read && !m_stop.load(std::memory_order_relaxed))
                    {
                        const auto sleepStart = Clock::now();
                        m_consumerSleeping.store(1u, std::memory_order_seq_cst);
                        const uint32_t epoch = m_wakeEpoch.load(std::memory_order_seq_cst);
                        if (m_writePos.load(std::memory_order_seq_cst) == read && !m_stop.load(std::memory_order_seq_cst))
                            m_wakeEpoch.wait(epoch, std::memory_order_seq_cst);
                        m_consumerSleeping.store(0u, std::memory_order_relaxed);
                        m_stats.idle += Clock::now() - sleepStart;
                    }
                    reportStats();
                    continue;
                }

                uint32_t sinceProgress = 0u;
                while (read != write)
                {
                    if (m_stop.load(std::memory_order_relaxed))
                    {
                        publishProgress(read);
                        return; // what is left is dropped
                    }
                    const Header header = *reinterpret_cast<const Header *>(m_ring + (read & m_mask));
                    const uint8_t *payload = m_ring + (read & m_mask) + sizeof(Header);
                    try
                    {
                        execute(header, payload);
                    }
                    catch (const std::exception &e)
                    {
                        if (!m_reportedException)
                        {
                            m_reportedException = true;
                            std::cerr << "[gpu-gs] command " << header.cmd << " failed: " << e.what() << std::endl;
                        }
                    }
                    read += sizeof(Header) + align16(header.bytes);
                    if (++sinceProgress >= 64u)
                    {
                        sinceProgress = 0u;
                        publishProgress(read);
                    }
                }
                publishProgress(read);
                reportStats();
            }
        }

        void execute(const Header &header, const uint8_t *payload)
        {
            switch (static_cast<Cmd>(header.cmd))
            {
            case Cmd::Wrap:
                break;
            case Cmd::Gif:
            {
                const auto start = Clock::now();
                m_gs->gif_transfer(header.flags, payload, header.bytes);
                m_stats.gif += Clock::now() - start;
                m_stats.gifBytes += header.bytes;
                break;
            }
            case Cmd::Reg:
                m_gs->write_register(static_cast<ParallelGS::RegisterAddr>(header.flags), header.aux);
                break;
            case Cmd::Reset:
                m_gs->reset_context_state();
                break;
            case Cmd::Call:
            {
                CallRecord *call = nullptr;
                std::memcpy(&call, payload, sizeof(call));
                try
                {
                    doCall(*call);
                }
                catch (...)
                {
                    call->done.store(true, std::memory_order_release);
                    wake();
                    throw;
                }
                call->done.store(true, std::memory_order_release);
                wake();
                break;
            }
            case Cmd::Present:
            {
                PresentRecord record;
                std::memcpy(&record, payload, sizeof(record));
                doPresent(record);
                m_presentsDone.fetch_add(1u, std::memory_order_release);
                if (m_producerWaiting.load(std::memory_order_seq_cst) != 0u)
                    wake();
                break;
            }
            }
        }

        void doCall(CallRecord &call)
        {
            constexpr size_t kVramBytes = size_t{4} << 20u;
            switch (call.kind)
            {
            case CallRecord::Snapshot:
            {
                const auto *vram = static_cast<const uint8_t *>(m_gs->map_vram_read(0, kVramBytes));
                call.out->assign(vram, vram + kVramBytes);
                break;
            }
            case CallRecord::ReadTransfer:
                m_gs->read_transfer_fifo(call.dst, call.bytes / 16u);
                call.result = call.bytes & ~15u;
                break;
            }
        }

        // VENDETTA_HOSTPROF: every 5 s, where the graphics card thread's time went.
        // Every second: the debug panel's status (from what was counted since the last one).
        void updateStatus(Clock::time_point now)
        {
            if (m_statusSince == Clock::time_point{})
                m_statusSince = now;
            const double seconds = std::chrono::duration<double>(now - m_statusSince).count();
            if (seconds < 1.0)
                return;
            const double idle = std::chrono::duration<double>(m_stats.idle - m_statusBase.idle).count();
            GSGpuStatus status;
            status.superSampling = m_currentSsaa;
            status.highResolution = m_lastScanout.high_resolution_scanout;
            status.fsr = m_lastFrameFsr;
            status.sharpness = m_wantSharpness.load(std::memory_order_relaxed);
            status.pictureWidth = m_lastFrameWidth;
            status.pictureHeight = m_lastFrameHeight;
            status.busyPercent = static_cast<float>(100.0 * (1.0 - idle / seconds));
            status.presentsPerSecond = static_cast<float>((m_stats.presents - m_statusBase.presents) / seconds);
            status.gifMegabytesPerSecond =
                static_cast<float>((m_stats.gifBytes - m_statusBase.gifBytes) / seconds / 1048576.0);
            status.renderPassesPerSecond =
                static_cast<float>((m_stats.renderPasses - m_statusBase.renderPasses) / seconds);
            {
                std::lock_guard<std::mutex> lock(m_statusMutex);
                m_status = status;
            }
            m_statusBase = m_stats;
            m_statusSince = now;
        }

        // VENDETTA_HOSTPROF: every 5 s, where the graphics card thread's time went.
        void reportStats()
        {
            const auto now = Clock::now();
            updateStatus(now);
            if (m_stats.since == Clock::time_point{})
                m_stats.since = now;
            const double seconds = std::chrono::duration<double>(now - m_stats.since).count();
            if (seconds < 5.0)
                return;
            if (!m_profile)
            {
                m_stats = Stats{};
                m_stats.since = now;
                m_statusBase = Stats{};
                return;
            }
            auto pct = [&](Clock::duration d)
            { return static_cast<int>(100.0 * std::chrono::duration<double>(d).count() / seconds + 0.5); };
            std::fprintf(stderr,
                         "[hostprof-gpu] busy=%d%% gif=%d%% flush=%d%% vsync=%d%% readback=%d%% presents/s=%.0f scanouts/s=%.0f "
                         "gif-MB/s=%.1f render-passes/s=%.0f prims/s=%.0f%c",
                         100 - pct(m_stats.idle), pct(m_stats.gif), pct(m_stats.flush), pct(m_stats.vsync), pct(m_stats.readback),
                         m_stats.presents / seconds, m_stats.scanouts / seconds, m_stats.gifBytes / seconds / 1048576.0,
                         m_stats.renderPasses / seconds, m_stats.primitives / seconds, 10);
            m_stats = Stats{};
            m_stats.since = now;
            m_statusBase = Stats{};
        }

        void doPresent(const PresentRecord &record)
        {
            ParallelGS::GSInterface &gs = *m_gs;
            const auto flushStart = Clock::now();
            gs.flush();
            m_stats.flush += Clock::now() - flushStart;
            ParallelGS::PrivRegisterState &priv = gs.get_priv_register_state();
            for (uint32_t i = 0; i < kPrivRegCount; ++i)
                priv.qwords_lo[2u * i] = record.regs[i];

            // A display circuit switched on with an all-zero size (DISPLAY DW = DH = 0: right after a GS reset, before
            // the game sets its display up again) shows nothing; paraLLEl-GS would take MAGH 0 as a 2560-wide mode and
            // scan out a huge black picture (5120 / 10240 wide at 2x / 4x). Black at the window's size instead.
            {
                const uint64_t pmode = record.regs[0];
                const bool shows1 = (pmode & 1u) != 0u && (record.regs[8] >> 32u) != 0u;
                const bool shows2 = (pmode & 2u) != 0u && (record.regs[10] >> 32u) != 0u;
                if (!shows1 && !shows2)
                {
                    ++m_stats.presents;
                    m_lastScanout = {};
                    readBack(m_lastScanout, 0u);
                    return;
                }
            }

            {
                // HCTP_DISPLOG=1: DISPFB/DISPLAY changes per vsync (first 400) - field-mode shimmy diagnosis.
                static const bool dispLog = std::getenv("HCTP_DISPLOG") != nullptr;
                static uint64_t s_prev[4] = {~0ull, ~0ull, ~0ull, ~0ull};
                static int s_lines = 0;
                const uint64_t cur[4] = {record.regs[7], record.regs[8], record.regs[9], record.regs[10]};
                if (dispLog && s_lines < 400 && std::memcmp(cur, s_prev, sizeof(cur)) != 0)
                {
                    ++s_lines;
                    std::memcpy(s_prev, cur, sizeof(cur));
                    std::fprintf(stderr, "[displog] tick=%llu flip=%u SMODE2=%llx DISPFB1=%llx DISPLAY1=%llx DISPFB2=%llx DISPLAY2=%llx\n",
                                 (unsigned long long)record.vsyncTick, (unsigned)record.flip,
                                 (unsigned long long)record.regs[2], (unsigned long long)cur[0],
                                 (unsigned long long)cur[1], (unsigned long long)cur[2], (unsigned long long)cur[3]);
                }
            }
            ParallelGS::VSyncInfo info{};
            // Field phase follows the game's field buffers, not the vsync count: HCTP draws two 640x224 field
            // buffers and swaps DISPFB each vsync. When the game misses a vsync the same buffer stays up, and
            // a tick-parity phase would shift that unchanged picture by a line (the whole-screen shimmy). Hold
            // the phase until DISPFB1/2 change. HCTP_FIELD_TICK=1 = old tick parity; HCTP_FIELD_INVERT=1 flips.
            static const bool fieldTick = std::getenv("HCTP_FIELD_TICK") != nullptr;
            static const uint32_t fieldInvert = std::getenv("HCTP_FIELD_INVERT") ? 1u : 0u;
            static uint64_t s_fieldFb[2] = {~0ull, ~0ull};
            static uint32_t s_fieldPhase = 0u;
            if (fieldTick || record.regs[7] != s_fieldFb[0] || record.regs[9] != s_fieldFb[1])
            {
                s_fieldFb[0] = record.regs[7];
                s_fieldFb[1] = record.regs[9];
                s_fieldPhase = static_cast<uint32_t>((record.vsyncTick + fieldInvert) & 1u);
            }
            info.phase = s_fieldPhase;
            info.force_progressive = true;
            info.anti_blur = true;
            info.overscan = false;
            info.crtc_offsets = false;
            info.adapt_to_internal_horizontal_resolution = true;
            info.raw_circuit_scanout = true;
            // live settings (the debug panel)
            bool settingsChanged = false;
            const uint32_t wantSsaa = m_wantSsaa.load(std::memory_order_relaxed);
            if (wantSsaa != m_currentSsaa)
            {
                gs.set_super_sampling_rate(static_cast<ParallelGS::SuperSampling>(wantSsaa), m_options.orderedGrid,
                                           m_options.superSampledTextures);
                m_currentSsaa = wantSsaa;
                settingsChanged = true;
            }
            const bool hires = m_wantHires.load(std::memory_order_relaxed) && m_currentSsaa >= 4u;
            const uint64_t output = m_outputSize.load(std::memory_order_relaxed);
            const bool fsr = m_wantFsr.load(std::memory_order_relaxed) && output != 0u;
            const float sharpness = m_wantSharpness.load(std::memory_order_relaxed);
            settingsChanged |= hires != m_lastHires || fsr != m_lastFsr || output != m_lastOutput || sharpness != m_lastSharpness;
            m_lastHires = hires;
            m_lastFsr = fsr;
            m_lastOutput = output;
            m_lastSharpness = sharpness;

            info.high_resolution_scanout = hires;
            info.quad_resolution_scanout = hires && m_quadScanout;
            // FSR samples the picture in a shader; the plain read-back copies it
            info.dst_layout = fsr ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            info.dst_stage = fsr ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_COPY_BIT;
            info.dst_access = fsr ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_TRANSFER_READ_BIT;

            const ParallelGS::FlushStats stats = gs.consume_flush_stats();
            ++m_stats.presents;
            m_stats.renderPasses += stats.num_render_passes;
            m_stats.primitives += stats.num_primitives;
            const bool duplicate = !settingsChanged && record.flip == 0u && stats.num_render_passes == 0u &&
                                   stats.num_copies == 0u &&
                                   m_lastScanout.image && gs.vsync_can_skip(info);
            if (duplicate)
                return; // the same picture: the main thread keeps the last one

            const auto vsyncStart = Clock::now();
            m_lastScanout = gs.vsync(info);
            if (m_lastScanout.image && m_lastScanout.mode_width > 1024u)
            {
                // a scanout wider than any real picture: show the display registers behind it (once per value)
                static uint64_t s_lastDisplay = ~0ull;
                const uint64_t display = record.regs[8] ^ (record.regs[10] << 1u) ^ record.regs[0];
                if (display != s_lastDisplay)
                {
                    s_lastDisplay = display;
                    std::fprintf(stderr, "[gpu-gs] wide scanout %ux%u: PMODE=%llx DISPFB1=%llx DISPLAY1=%llx DISPFB2=%llx DISPLAY2=%llx SMODE2=%llx\n",
                                 m_lastScanout.mode_width, m_lastScanout.mode_height,
                                 (unsigned long long)record.regs[0], (unsigned long long)record.regs[7],
                                 (unsigned long long)record.regs[8], (unsigned long long)record.regs[9],
                                 (unsigned long long)record.regs[10], (unsigned long long)record.regs[2]);
                }
            }
            const auto readStart = Clock::now();
            readBack(m_lastScanout, fsr ? output : 0u);
            m_stats.vsync += readStart - vsyncStart;
            m_stats.readback += Clock::now() - readStart;
            ++m_stats.scanouts;
        }

        // Copies the scanout picture back to host memory and publishes it (alpha forced opaque: the window blends).
        // fsrOutput != 0: upscaled + sharpened to that size first (packed width << 32 | height).
        void readBack(const ParallelGS::ScanoutResult &scanout, uint64_t fsrOutput)
        {
            GSGpuFrame &frame = m_frames.writeSlot();
            frame.seq = ++m_frameSeq;
            if (!scanout.image)
            {
                frame.width = frame.height = 0u;
                frame.rgba.clear();
                m_frames.publish();
                m_lastFrameWidth = m_lastFrameHeight = 0u;
                return;
            }
            uint32_t width = scanout.image->get_width();
            uint32_t height = scanout.image->get_height();
            {
                // the picture paraLLEl-GS scans out, once per change (its size shows the 1x / 2x / 4x scanout)
                static uint32_t s_loggedWidth = 0u, s_loggedHeight = 0u;
                if (width != s_loggedWidth || height != s_loggedHeight)
                {
                    s_loggedWidth = width;
                    s_loggedHeight = height;
                    std::cerr << "[gpu-gs] scanout " << width << "x" << height << " (mode " << scanout.mode_width << "x"
                              << scanout.mode_height << (scanout.high_resolution_scanout ? ", high resolution" : "")
                              << (scanout.interlaced ? ", interlaced" : "") << ")" << std::endl;
                }
            }
            const Vulkan::Image *source = scanout.image.get();
            Vulkan::CommandBufferHandle cmd = m_device->request_command_buffer();
            m_lastFrameFsr = false;
            if (fsrOutput != 0u)
            {
                const uint32_t outWidth = static_cast<uint32_t>(fsrOutput >> 32u);
                const uint32_t outHeight = static_cast<uint32_t>(fsrOutput & 0xFFFFFFFFu);
                // FSR only upscales; at or below the picture's own size the plain copy is as good
                if (outWidth > width && outHeight > height && outWidth <= 8192u && outHeight <= 8192u)
                {
                    runFsr(*cmd, scanout.image->get_view(), outWidth, outHeight);
                    source = m_fsrSharpened.get();
                    width = outWidth;
                    height = outHeight;
                    m_lastFrameFsr = true;
                }
                else
                {
                    // the picture is in the sampling layout; make it copyable
                    cmd->image_barrier(*scanout.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                                       VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                }
            }
            m_lastFrameWidth = width;
            m_lastFrameHeight = height;
            if (m_sharedPresent && g_sharedStopped.load(std::memory_order_acquire))
                m_sharedPresent = false; // the window fell back to raylib's: read back as before
            if (m_sharedPresent && shareFrame(frame, cmd, *source, width, height))
            {
                frame.internalWidth = scanout.internal_width;
                frame.internalHeight = scanout.internal_height;
                frame.modeWidth = scanout.mode_width;
                frame.modeHeight = scanout.mode_height;
                frame.highResolution = scanout.high_resolution_scanout;
                m_frames.publish();
                return;
            }
            frame.shared = false;
            const VkDeviceSize bytes = VkDeviceSize{width} * height * 4u;
            if (!m_readback || m_readback->get_create_info().size < bytes)
            {
                Vulkan::BufferCreateInfo info{};
                info.size = bytes;
                info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                info.domain = Vulkan::BufferDomain::CachedHost;
                m_readback = m_device->create_buffer(info);
            }

            cmd->copy_image_to_buffer(*m_readback, *source, 0, {0, 0, 0}, {width, height, 1}, 0, 0,
                                      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
            cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
            Vulkan::Fence fence;
            m_device->submit(cmd, &fence);
            fence->wait();

            const auto *src = static_cast<const uint32_t *>(
                m_device->map_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT, 0, bytes));
            frame.width = width;
            frame.height = height;
            frame.internalWidth = scanout.internal_width;
            frame.internalHeight = scanout.internal_height;
            frame.modeWidth = scanout.mode_width;
            frame.modeHeight = scanout.mode_height;
            frame.highResolution = scanout.high_resolution_scanout;
            frame.rgba.resize(static_cast<size_t>(bytes));
            auto *dst = reinterpret_cast<uint32_t *>(frame.rgba.data());
            const size_t pixels = static_cast<size_t>(width) * height;
            for (size_t i = 0; i < pixels; ++i)
                dst[i] = src[i] | 0xFF000000u;
            m_device->unmap_host_buffer(*m_readback, Vulkan::MEMORY_ACCESS_READ_BIT, 0, bytes);
            m_frames.publish();
        }

        // shown through Vulkan: the picture copied into one of four images shared with the window's own Vulkan device
        // (no read back): one the window isn't drawing from and that isn't one of the two newest pictures. It is handed
        // over in the sampling layout, released to the "external" queue family; the shared timeline semaphore reaches the
        // frame's number once it is there. The images are made again at a new size (a new generation of handles).
        bool shareFrame(GSGpuFrame &frame, Vulkan::CommandBufferHandle &cmd, const Vulkan::Image &source,
                        uint32_t width, uint32_t height)
        {
            if (!m_sharedTimeline)
            {
                m_sharedTimeline = m_device->request_semaphore_external(VK_SEMAPHORE_TYPE_TIMELINE,
                                                                        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT);
                if (m_sharedTimeline)
                    m_sharedTimelineHandle = m_sharedTimeline->export_to_handle().handle;
                if (!m_sharedTimeline || !m_sharedTimelineHandle)
                {
                    std::cerr << "[gpu-gs] no shareable timeline semaphore: pictures read back as before" << std::endl;
                    m_sharedPresent = false;
                    return false;
                }
            }
            if (!m_sharedImages[0] || m_sharedImages[0]->get_width() != width || m_sharedImages[0]->get_height() != height)
            {
                // the old handles stay open one generation more: the window may still be importing them
                closeSharedHandles(m_sharedOldHandles);
                std::copy(std::begin(m_sharedHandles), std::end(m_sharedHandles), std::begin(m_sharedOldHandles));
                std::fill(std::begin(m_sharedHandles), std::end(m_sharedHandles), nullptr);
                for (int i = 0; i < 4; ++i)
                {
                    Vulkan::ImageCreateInfo info = sharedImageInfo(width, height);
                    m_sharedImages[i] = m_device->create_image(info);
                    if (m_sharedImages[i])
                        m_sharedHandles[i] = m_sharedImages[i]->export_handle().handle;
                    if (!m_sharedImages[i] || !m_sharedHandles[i])
                    {
                        std::cerr << "[gpu-gs] no shareable " << width << "x" << height
                                  << " image: pictures read back as before" << std::endl;
                        m_sharedPresent = false;
                        return false;
                    }
                }
                ++m_sharedGeneration;
                m_sharedNewest[0] = m_sharedNewest[1] = -1;
            }
            const int reading = m_frames.sharedReading.load(std::memory_order_acquire);
            int target = 0;
            for (int i = 0; i < 4; ++i)
                if (i != reading && i != m_sharedNewest[0] && i != m_sharedNewest[1])
                {
                    target = i;
                    break;
                }
            const Vulkan::Image &image = *m_sharedImages[target];
            cmd->image_barrier(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            cmd->copy_image(image, source, {0, 0, 0}, {0, 0, 0}, {width, height, 1},
                            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1});
            VkImageMemoryBarrier2 release{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            release.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            release.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            release.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            release.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            release.srcQueueFamilyIndex = m_device->get_queue_info().family_indices[Vulkan::QUEUE_INDEX_GRAPHICS];
            release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            release.image = image.get_image();
            release.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &release;
            cmd->barrier(dep);
            m_device->submit(cmd);
            auto signal = m_device->request_timeline_semaphore_as_binary(*m_sharedTimeline, frame.seq);
            m_device->submit_empty(Vulkan::CommandBuffer::Type::Generic, nullptr, signal.get());

            frame.shared = true;
            frame.width = width;
            frame.height = height;
            frame.rgba.clear();
            frame.sharedImage = target;
            frame.sharedGeneration = m_sharedGeneration;
            std::copy(std::begin(m_sharedHandles), std::end(m_sharedHandles), std::begin(frame.sharedHandles));
            frame.sharedTimeline = m_sharedTimelineHandle;
            std::memcpy(frame.sharedDeviceUuid, m_deviceUuid, sizeof(m_deviceUuid));
            m_sharedNewest[1] = m_sharedNewest[0];
            m_sharedNewest[0] = target;
            return true;
        }

        static void closeSharedHandles(void *(&handles)[4])
        {
            for (void *&handle : handles)
            {
                if (handle)
                    CloseHandle(handle);
                handle = nullptr;
            }
        }

        void ensureTarget(Vulkan::ImageHandle &image, uint32_t width, uint32_t height)
        {
            if (image && image->get_width() == width && image->get_height() == height)
                return;
            Vulkan::ImageCreateInfo info = Vulkan::ImageCreateInfo::render_target(width, height, VK_FORMAT_R8G8B8A8_UNORM);
            info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            image = m_device->create_image(info);
        }

        // AMD FSR 1 on the picture: EASU upscales it to the window's size, RCAS sharpens it (both in the picture's own
        // gamma space, as AMD recommends). The result ends in m_fsrSharpened, ready to copy.
        void runFsr(Vulkan::CommandBuffer &cmd, const Vulkan::ImageView &picture, uint32_t width, uint32_t height)
        {
            ensureTarget(m_fsrUpscaled, width, height);
            ensureTarget(m_fsrSharpened, width, height);
            const float vertices[] = {-1.0f, -1.0f, -1.0f, 3.0f, 3.0f, -1.0f};
            struct Push
            {
                float width, height;
            } push{static_cast<float>(width), static_cast<float>(height)};

            cmd.image_barrier(*m_fsrUpscaled, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            Vulkan::RenderPassInfo pass{};
            pass.num_color_attachments = 1;
            pass.color_attachments[0] = &m_fsrUpscaled->get_view();
            pass.store_attachments = 1u << 0;
            cmd.begin_render_pass(pass);
            {
                struct Constants
                {
                    float params[4][4];
                } constants;
                fsrEasuConstants(constants.params[0], constants.params[1], constants.params[2], constants.params[3],
                                 static_cast<float>(picture.get_image().get_width()),
                                 static_cast<float>(picture.get_image().get_height()), static_cast<float>(width),
                                 static_cast<float>(height));
                *cmd.allocate_typed_constant_data<Constants>(1, 0, 1) = constants;
                cmd.push_constants(&push, 0, sizeof(push));
                std::memcpy(cmd.allocate_vertex_data(0, sizeof(vertices), 2 * sizeof(float)), vertices, sizeof(vertices));
                cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32_SFLOAT, 0);
                cmd.set_texture(0, 0, picture, Vulkan::StockSampler::NearestClamp);
                cmd.set_program(m_upscaleProgram);
                cmd.set_opaque_state();
                cmd.set_depth_test(false, false);
                cmd.draw(3);
            }
            cmd.end_render_pass();
            cmd.image_barrier(*m_fsrUpscaled, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            cmd.image_barrier(*m_fsrSharpened, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            pass.color_attachments[0] = &m_fsrSharpened->get_view();
            cmd.begin_render_pass(pass);
            {
                struct Constants
                {
                    float params[4];
                    int32_t range[4];
                } constants;
                fsrRcasConstants(constants.params, m_lastSharpness);
                constants.range[0] = 0;
                constants.range[1] = 0;
                constants.range[2] = static_cast<int32_t>(width) - 1;
                constants.range[3] = static_cast<int32_t>(height) - 1;
                *cmd.allocate_typed_constant_data<Constants>(1, 0, 1) = constants;
                cmd.push_constants(&push, 0, sizeof(push));
                std::memcpy(cmd.allocate_vertex_data(0, sizeof(vertices), 2 * sizeof(float)), vertices, sizeof(vertices));
                cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32_SFLOAT, 0);
                cmd.set_texture(0, 0, m_fsrUpscaled->get_view(), Vulkan::StockSampler::NearestClamp);
                cmd.set_program(m_sharpenProgram);
                cmd.set_opaque_state();
                cmd.set_depth_test(false, false);
                cmd.draw(3);
            }
            cmd.end_render_pass();
            cmd.image_barrier(*m_fsrSharpened, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                              VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        }

        GSGpuOptions m_options;
        std::string m_deviceName;
        std::promise<std::string> m_ready;
        std::thread m_thread;

        std::unique_ptr<uint8_t[]> m_ringStorage;
        uint8_t *m_ring = nullptr;
        uint64_t m_size = 0u;
        uint64_t m_mask = 0u;

        alignas(64) std::atomic<uint64_t> m_writePos{0u};
        alignas(64) std::atomic<uint64_t> m_readPos{0u};

        // producer only
        alignas(64) uint64_t m_localWrite = 0u;
        uint64_t m_published = 0u;
        uint64_t m_cachedRead = 0u;
        uint64_t m_pendingRecordBytes = 0u;
        alignas(64) std::atomic<uint64_t> m_presentsIssued{0u};

        // the graphics card thread only
        alignas(64) Vulkan::Device *m_device = nullptr;
        ParallelGS::GSInterface *m_gs = nullptr;
        bool m_quadScanout = false;
        ParallelGS::ScanoutResult m_lastScanout{};
        Vulkan::BufferHandle m_readback;
        uint64_t m_frameSeq = 0u;
        bool m_reportedException = false;
        alignas(64) std::atomic<uint64_t> m_presentsDone{0u};

        // waking the consumer (atomic wait) and the producers (condition variable, with a stall report)
        alignas(64) std::atomic<uint32_t> m_consumerSleeping{0u};
        std::atomic<uint32_t> m_wakeEpoch{0u};
        std::atomic<uint32_t> m_producerWaiting{0u};
        std::mutex m_progressMutex;
        std::condition_variable m_progressCv;
        std::atomic<bool> m_stop{false};
        bool m_shutDown = false;

        GSGpuFrameMailbox m_frames;

        struct Stats
        {
            Clock::time_point since{};
            Clock::duration idle{}, gif{}, flush{}, vsync{}, readback{};
            uint64_t gifBytes = 0, presents = 0, scanouts = 0, renderPasses = 0, primitives = 0;
        };
        bool m_profile = false;
        Stats m_stats; // the graphics card thread only
        Stats m_statusBase;
        Clock::time_point m_statusSince{};

        // live settings: written by any thread, taken at a presentation
        std::atomic<uint32_t> m_wantSsaa{8u};
        std::atomic<bool> m_wantHires{true};
        std::atomic<bool> m_wantFsr{true};
        std::atomic<float> m_wantSharpness{0.5f};
        std::atomic<uint64_t> m_outputSize{0u};
        // the graphics card thread only
        uint32_t m_currentSsaa = 0u;
        bool m_lastHires = false;
        bool m_lastFsr = false;
        uint64_t m_lastOutput = 0u;
        float m_lastSharpness = -1.0f;
        bool m_lastFrameFsr = false;
        uint32_t m_lastFrameWidth = 0u;
        uint32_t m_lastFrameHeight = 0u;
        Vulkan::Program *m_upscaleProgram = nullptr;
        Vulkan::Program *m_sharpenProgram = nullptr;
        // shown through Vulkan: the pictures shared with the window's device
        bool m_sharedPresent = false; // the drawing thread's own; also off once g_sharedStopped
        uint8_t m_deviceUuid[16] = {};
        Vulkan::ImageHandle m_sharedImages[4];
        void *m_sharedHandles[4] = {};
        void *m_sharedOldHandles[4] = {};
        uint64_t m_sharedGeneration = 0u;
        int m_sharedNewest[2] = {-1, -1};
        Vulkan::Semaphore m_sharedTimeline;
        void *m_sharedTimelineHandle = nullptr;
        Vulkan::ImageHandle m_fsrUpscaled;
        Vulkan::ImageHandle m_fsrSharpened;
        mutable std::mutex m_statusMutex;
        GSGpuStatus m_status;
    };
}

std::unique_ptr<GSGpuRenderer> makeGpuRenderer(const GSGpuOptions &options, std::string &whyNot)
{
    if (auto native = makeNativeGsModuleRenderer(whyNot))
        return native;
    if (!whyNot.empty())
        return nullptr;
    auto renderer = std::make_unique<ParallelGsRenderer>(options);
    if (!renderer->start(whyNot))
        return nullptr;
    return renderer;
}
