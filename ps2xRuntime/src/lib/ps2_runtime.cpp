#include "runtime/ps2_host_profile.h"
#include "runtime/ps2_host_settings.h"
#include "runtime/ps2_pad_rumble.h"
#include "runtime/ps2_widescreen.h"
#include "ps2_runtime.h"
#include "ps2_log.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "game_overrides.h"
#include "ps2_runtime_macros.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ee_scheduler.h"
#include "ThreadNaming.h"
#include "Kernel/Stubs/Audio.h"
#include "Kernel/Stubs/GS.h"
#include "Kernel/Stubs/MPEG.h"
#include "ps2_host_backend.h"
#if defined(PS2X_NATIVE_HOST)
#include "runtime/gs/gs_d3d11_backend.h"
#endif
#include "ps2_iop_host.h"
#include "hctp_native_vu1.h"
#include "runtime/ps2_vu1_compiled.h"
#include "ps2x/iop/iop_subsystem.h"

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>
#include <limits>
#include <chrono>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <sstream>
#if defined(PS2X_HAS_GLFW)
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "runtime/ps2_host_window.h"
#endif
#if defined(PS2X_HAS_GLFW) && defined(PS2X_HAS_PARALLEL_GS)
#include "gs/gs_vulkan_present.h"
#include "ps2_host_input.h"
#define PS2X_VULKAN_PRESENT 1
#endif

namespace ps2_stubs
{
    void resetSifState();
}

#define ELF_MAGIC 0x464C457F // "\x7FELF" in little endian
#define ET_EXEC 2            // Executable file
#define EM_MIPS 8            // MIPS architecture
#define PT_LOAD 1            // Loadable segment

static constexpr int FB_WIDTH = 640;
static constexpr int FB_HEIGHT = 512;
static constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
static constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;
static constexpr uint32_t DEFAULT_FB_ADDR = (PS2_RAM_SIZE - DEFAULT_FB_SIZE - 0x10000u);
#if defined(PLATFORM_VITA)
static constexpr int HOST_WINDOW_WIDTH = 960;
static constexpr int HOST_WINDOW_HEIGHT = 544;
#else
static constexpr int HOST_WINDOW_WIDTH = FB_WIDTH;
static constexpr int HOST_WINDOW_HEIGHT = DEFAULT_DISPLAY_HEIGHT;
#endif
struct ElfHeader
{
    uint32_t magic;
    uint8_t elf_class;
    uint8_t endianness;
    uint8_t version;
    uint8_t os_abi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
};

struct ProgramHeader
{
    uint32_t type;
    uint32_t offset;
    uint32_t vaddr;
    uint32_t paddr;
    uint32_t filesz;
    uint32_t memsz;
    uint32_t flags;
    uint32_t align;
};

void ps2RecordDispatch(char what, uint32_t pc, uint32_t sp, uint32_t ra, int thread); // defined below
bool ps2DispatchRecording() noexcept;                                                  // defined below
namespace
{
    // VENDETTA_DISPATCHLOG=1: every call, return and thread dispatch into a small ring (the "ended at pc 0" report
    // prints it). Off by default: two records per guest call were ~2% of the game thread.
    const bool g_recordDispatches = []
    {
        const char *value = std::getenv("VENDETTA_DISPATCHLOG");
        return value && *value && *value != '0';
    }();
}

namespace
{
    constexpr uint32_t kGuestHeapDefaultBase = 0x00100000u;
    constexpr uint32_t kGuestHeapDefaultAlignment = 16u;
    constexpr uint32_t kGuestHeapSafetyPad = 0x1000u;
    constexpr uint32_t kGuestHeapHardLimit = 0x01FF0000u; // HCTP: ps2_main mallocs up to 0x1FF0000 (main stack above); old-fork CAM fix
    constexpr uint32_t kGuestLowPoolBase = 0x000A0000u; // see resetGuestHeapLocked
    constexpr uint32_t kGuestLowPoolSize = 0x0005F000u;
    // kernel RAM (see Helpers/State.h): interrupt / callback stacks and the runtime's small buffers
    constexpr uint32_t kKernelCallbackStackFloor = 0x00048000u;
    constexpr uint32_t kKernelCallbackStackTop = 0x00080000u;
    constexpr uint32_t kKernelArenaBase = 0x00080000u; // HCTP: was 0x41000-0x48000 (28 KB): full -> 16 KB GS uploads spilled into the game heap
    constexpr uint32_t kKernelArenaLimit = 0x000A0000u; // 128 KB, below the guest low pool (0xA0000)

    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_GENERAL = 0x80000080u;
    constexpr uint32_t EXCEPTION_VECTOR_TLB_REFILL = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT = 0xBFC00200u;

    struct DispatchHistory
    {
        std::array<uint32_t, 64> pcs{};
        uint32_t next = 0u;
        bool wrapped = false;
    };

    thread_local DispatchHistory g_dispatchHistory;

    bool computeFileCrc32(const std::string &path, uint32_t &crcOut)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        static const std::array<uint32_t, 256> table = []
        {
            std::array<uint32_t, 256> values{};
            for (uint32_t i = 0; i < values.size(); ++i)
            {
                uint32_t value = i;
                for (uint32_t bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1u) ? (0xEDB88320u ^ (value >> 1u)) : (value >> 1u);
                }
                values[i] = value;
            }
            return values;
        }();

        uint32_t crc = 0xFFFFFFFFu;
        std::array<uint8_t, 16 * 1024> buffer{};
        while (file.good())
        {
            file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = file.gcount();
            for (std::streamsize i = 0; i < count; ++i)
            {
                crc = table[(crc ^ buffer[static_cast<size_t>(i)]) & 0xFFu] ^ (crc >> 8u);
            }
        }
        if (file.bad())
        {
            return false;
        }
        crcOut = ~crc;
        return true;
    }

    void pushDispatchPc(uint32_t pc)
    {
        DispatchHistory &h = g_dispatchHistory;
        h.pcs[h.next] = pc;
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.pcs.size());
        if (h.next == 0u)
        {
            h.wrapped = true;
        }
    }

    std::string formatDispatchHistory()
    {
        const DispatchHistory &h = g_dispatchHistory;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return "(empty)";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t idx = (h.next + h.pcs.size() - count + i) % static_cast<uint32_t>(h.pcs.size());
            if (!first)
            {
                oss << " -> ";
            }
            first = false;
            oss << "0x" << std::hex << h.pcs[idx];
        }
        return oss.str();
    }

    uint32_t selectExceptionVector(const R5900Context *ctx, bool tlbRefill)
    {
        if (ctx->cop0_status & COP0_STATUS_BEV)
        {
            return EXCEPTION_VECTOR_BOOT;
        }
        return tlbRefill ? EXCEPTION_VECTOR_TLB_REFILL : EXCEPTION_VECTOR_GENERAL;
    }

    void seedVu0IdleSuccess(R5900Context *ctx)
    {
        if (!ctx)
        {
            return;
        }

        ctx->vu0_clip_flags = 0;
        ctx->vu0_clip_flags2 = 0;
        ctx->vu0_mac_flags = 0;
        ctx->vu0_status = 0;
        ctx->vu0_q = 1.0f;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;
    }

    void copyVu0ContextToState(const R5900Context *ctx, VU1State &state)
    {
        std::memset(&state, 0, sizeof(state));

        for (uint32_t i = 0; i < 32u; ++i)
        {
            _mm_storeu_ps(state.vf[i], ctx->vu0_vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            state.vi[i] = static_cast<int16_t>(ctx->vi[i]);
        }

        _mm_storeu_ps(state.acc, ctx->vu0_acc);
        state.q = ctx->vu0_q;
        state.p = ctx->vu0_p;
        state.i = ctx->vu0_i;
        alignas(16) uint32_t rWords[4]{};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(rWords), _mm_castps_si128(ctx->vu0_r));
        state.r = 0x3F800000u | (rWords[0] & 0x007FFFFFu);
        state.pc = ctx->vu0_pc;
        state.mac = ctx->vu0_mac_flags;
        state.clip = ctx->vu0_clip_flags;
        state.status = ctx->vu0_status;
        state.itop = ctx->vu0_itop;
        state.dBitEnabled = (ctx->vu0_fbrst & (1u << 2)) != 0u;
        state.tBitEnabled = (ctx->vu0_fbrst & (1u << 3)) != 0u;

        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;
    }

    void copyVu0StateToContext(const VU1State &state, R5900Context *ctx)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            ctx->vu0_vf[i] = _mm_loadu_ps(state.vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            ctx->vi[i] = static_cast<uint16_t>(state.vi[i]);
        }

        ctx->vu0_acc = _mm_loadu_ps(state.acc);
        ctx->vu0_q = state.q;
        ctx->vu0_p = state.p;
        ctx->vu0_i = state.i;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int32_t>(state.r)));
        ctx->vu0_mac_flags = state.mac;
        ctx->vu0_clip_flags = state.clip;
        ctx->vu0_clip_flags2 = state.clip;
        ctx->vu0_status = static_cast<uint16_t>(state.status);
        ctx->vu0_itop = state.itop;
        ctx->vu0_pc = state.pc;
        ctx->vu0_tpc = state.pc;
        ctx->vu0_vpu_stat = (ctx->vu0_vpu_stat & 0xFF00u) | (state.stoppedByD ? (1u << 1) : 0u) | (state.stoppedByT ? (1u << 2) : 0u);
        ctx->vu0_vpu_stat2 = 0;

        ctx->vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
        ctx->vi[0] = 0;
    }

    void raiseCop0Exception(R5900Context *ctx, uint32_t exceptionCode, bool tlbRefill = false)
    {
        if (ctx->in_delay_slot)
        {
            ctx->cop0_epc = ctx->branch_pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~COP0_CAUSE_EXCCODE_MASK) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK) |
                              COP0_CAUSE_BD;
        }
        else
        {
            ctx->cop0_epc = ctx->pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~(COP0_CAUSE_EXCCODE_MASK | COP0_CAUSE_BD)) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK);
        }

        ctx->cop0_status |= COP0_STATUS_EXL;
        ctx->pc = selectExceptionVector(ctx, tlbRefill);
        ctx->in_delay_slot = false;
    }

    std::filesystem::path normalizeAbsolutePath(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return {};
        }

#if defined(PLATFORM_VITA)
        const std::string generic = path.generic_string();
        const std::size_t colon = generic.find(':');
        if (colon != std::string::npos && colon != 0u)
        {
            const std::size_t slash = generic.find_first_of("/\\");
            if (slash == std::string::npos || colon < slash)
            {
                return path.lexically_normal();
            }
        }
#endif

        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        if (ec)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    PS2Runtime::IoPaths &runtimeIoPaths()
    {
        static PS2Runtime::IoPaths paths = []()
        {
            PS2Runtime::IoPaths defaults;
            std::error_code ec;
            const std::filesystem::path cwd = std::filesystem::current_path(ec);
            defaults.elfDirectory = ec ? std::filesystem::path(".") : cwd.lexically_normal();
            defaults.hostRoot = defaults.elfDirectory;
            defaults.cdRoot = defaults.elfDirectory;
            defaults.mcRoot = defaults.elfDirectory / "mc0";
            return defaults;
        }();

        return paths;
    }

    std::string readGuestPrintableString(const uint8_t *rdram, uint32_t addr, size_t maxLen)
    {
        std::string out;
        if (!rdram || maxLen == 0)
        {
            return out;
        }

        out.reserve(std::min<size_t>(maxLen, 64));
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char ch = static_cast<char>(rdram[(addr + static_cast<uint32_t>(i)) & PS2_RAM_MASK]);
            if (ch == '\0')
            {
                break;
            }
            if (ch >= 0x20 && ch < 0x7F)
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('.');
            }
        }
        return out;
    }
}

namespace
{
    int s_frameTextureFilter = TEXTURE_FILTER_BILINEAR;
    // shown through Vulkan: the newest picture shared with the window's Vulkan device (valid until the next take)
    const GSGpuFrame *s_sharedFrame = nullptr;
    // The graphics card's picture: (covered share of the video mode's width) / (share of its height), for the aspect.
    float s_presentAspectScale = 1.0f;

    // the log's note on the refresh choice
    const char *refreshChoiceName(int refreshRate)
    {
        return refreshRate == ps2_host_settings::kRefreshSmoothest ? " (the smoothest)"
               : refreshRate == ps2_host_settings::kRefreshAutomatic ? " (automatic)"
               : refreshRate == 0                                    ? " (the screen's own)"
                                                                     : "";
    }
}

// The window as the settings ask (ps2_host_settings): which screen, maximised, borderless, or fullscreen at a chosen
// resolution and refresh rate. raylib's own fullscreen asks for any refresh rate (GLFW then takes the screen's highest),
// so the mode is set again through GLFW, built into raylib on desktops: the rate asked for, or the screen's current one.
static void applyWindowSettings()
{
#if !defined(PLATFORM_VITA)
    const ps2_host_settings::Settings &settings = ps2_host_settings::get();
    const int monitorCount = GetMonitorCount();
    const int monitor = std::clamp(settings.screen - 1, 0, std::max(0, monitorCount - 1));
    if (settings.screen > monitorCount)
        std::cerr << "[window] screen " << settings.screen << " not found (" << monitorCount << " screens); using screen "
                  << (monitor + 1) << std::endl;
    if (monitorCount > 1 && monitor != GetCurrentMonitor())
        SetWindowMonitor(monitor);
    switch (settings.mode)
    {
    case ps2_host_settings::WindowMode::Window:
        if (settings.maximised)
            MaximizeWindow();
        break;
    case ps2_host_settings::WindowMode::Borderless:
        ToggleBorderlessWindowed();
        ClearWindowState(FLAG_WINDOW_TOPMOST); // raylib makes it always-on-top; Alt-Tab and other windows should still work
        break;
    case ps2_host_settings::WindowMode::Fullscreen:
    {
        const int width = settings.fullscreenWidth > 0 ? settings.fullscreenWidth : GetMonitorWidth(monitor);
        const int height = settings.fullscreenHeight > 0 ? settings.fullscreenHeight : GetMonitorHeight(monitor);
        SetWindowSize(width, height);
#if defined(PS2X_HAS_GLFW)
        // One mode change, straight to the size and rate wanted: raylib's ToggleFullscreen first took the screen's
        // highest rate, then the rate was set again — two changes, the desktop showing for a moment in between.
        bool switched = false;
        int count = 0;
        GLFWmonitor **monitors = glfwGetMonitors(&count);
        GLFWwindow *window = glfwGetCurrentContext();
        if (window && monitors && monitor < count)
        {
            // the rates this screen has at this size (GLFW takes the closest one to what is asked)
            int modeCount = 0;
            const GLFWvidmode *modes = glfwGetVideoModes(monitors[monitor], &modeCount);
            std::vector<int> rates;
            for (int i = 0; i < modeCount; ++i)
                if (modes[i].width == width && modes[i].height == height &&
                    std::find(rates.begin(), rates.end(), modes[i].refreshRate) == rates.end())
                    rates.push_back(modes[i].refreshRate);
            std::sort(rates.begin(), rates.end());
            const GLFWvidmode *current = glfwGetVideoMode(monitors[monitor]);
            // the game makes ~60 pictures a second: automatic / smoothest / the screen's own / a fixed rate
            // (ps2_host_settings::chooseRefreshRate)
            const int chosen = ps2_host_settings::chooseRefreshRate(settings.refreshRate, current ? current->refreshRate : 0, rates);
            const int rate = chosen > 0 ? chosen : GLFW_DONT_CARE;
            std::cerr << "[window] fullscreen " << width << "x" << height << " at " << rate << " Hz"
                      << refreshChoiceName(settings.refreshRate)
                      << "; the screen has";
            for (int r : rates)
                std::cerr << " " << r;
            std::cerr << (rates.empty() ? " no mode of that size" : " Hz") << std::endl;
            // The screen black before the mode change: the change blanks the screen for a moment, and with the window
            // still hidden the desktop came back on screen until the game's first picture. A frameless black window
            // over the whole screen (at its current mode) first, then the change under it.
            if (current)
            {
                const Vector2 at = GetMonitorPosition(monitor);
                SetWindowState(FLAG_WINDOW_UNDECORATED);
                SetWindowPosition(static_cast<int>(at.x), static_cast<int>(at.y));
                SetWindowSize(current->width, current->height);
                ClearWindowState(FLAG_WINDOW_HIDDEN);
                for (int i = 0; i < 2; ++i)
                {
                    BeginDrawing();
                    ClearBackground(BLACK);
                    EndDrawing();
                }
            }
            // Alt-Tab: raylib makes every window with GLFW_AUTO_ICONIFY off, so a fullscreen window (always on top)
            // stayed over the window Alt-Tab picked. Now losing focus to the player's own switch minimises it and gives
            // the screen back its own mode (clicking it or Alt-Tab back restores both); another program taking the
            // front by itself (Copilot's Quick View at a mode change) leaves it fullscreen (ps2_host_window).
            ps2_host_window::minimiseOnlyWhenThePlayerSwitchesAway(window);
            glfwSetWindowMonitor(window, monitors[monitor], 0, 0, width, height, rate);
            switched = true;
        }
        if (!switched)
            ToggleFullscreen();
#else
        ToggleFullscreen();
#endif
        HideCursor(); // the game has no pointer; the debug panel (F1) shows it again while open
        break;
    }
    }
    std::cerr << "[window] " << ps2_host_settings::windowModeName(settings.mode) << " on screen " << (monitor + 1) << " of "
              << monitorCount << "; the screen now " << GetMonitorWidth(monitor) << "x" << GetMonitorHeight(monitor) << " at "
              << GetMonitorRefreshRate(monitor) << " Hz; vsync " << (settings.vsync ? "on" : "off") << std::endl;
#endif
}

#if defined(PS2X_VULKAN_PRESENT)
namespace
{
    // The game's window shown through Vulkan (vulkanPresentWanted): a GLFW window without OpenGL (raylib's window is
    // made too and stays hidden: raylib still plays the sound)
    GLFWwindow *s_vulkanWindow = nullptr;
    std::unique_ptr<VulkanPresenter> s_presenter;

    // The refresh rate for a fullscreen mode: the rate asked for; "smoothest" = the highest multiple of 60 Hz the screen
    // has at that size (else its highest); 0 = the screen's current rate.
    int fullscreenRate(GLFWmonitor *monitor, int width, int height, const ps2_host_settings::Settings &settings,
                       std::vector<int> &rates)
    {
        int modeCount = 0;
        const GLFWvidmode *modes = glfwGetVideoModes(monitor, &modeCount);
        for (int i = 0; i < modeCount; ++i)
            if (modes[i].width == width && modes[i].height == height &&
                std::find(rates.begin(), rates.end(), modes[i].refreshRate) == rates.end())
                rates.push_back(modes[i].refreshRate);
        std::sort(rates.begin(), rates.end());
        const GLFWvidmode *current = glfwGetVideoMode(monitor);
        const int chosen = ps2_host_settings::chooseRefreshRate(settings.refreshRate, current ? current->refreshRate : 0, rates);
        return chosen > 0 ? chosen : GLFW_DONT_CARE;
    }

    // Made hidden, placed as the settings ask (window / maximised / borderless / fullscreen at its resolution and
    // rate, on its screen), painted black, then shown - for fullscreen the black window covers the screen first and
    // the mode changes under it (no desktop between). In fullscreen, losing focus to the player's own switch (Alt-Tab)
    // minimises it; another program taking the front by itself doesn't (ps2_host_window).
    GLFWwindow *createVulkanWindow(const char *title)
    {
        const ps2_host_settings::Settings &settings = ps2_host_settings::get();
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_TRUE);
        int count = 0;
        GLFWmonitor **monitors = glfwGetMonitors(&count);
        const int index = std::clamp(settings.screen - 1, 0, std::max(0, count - 1));
        GLFWmonitor *monitor = count > 0 ? monitors[index] : glfwGetPrimaryMonitor();
        const GLFWvidmode *current = monitor ? glfwGetVideoMode(monitor) : nullptr;
        int monitorX = 0, monitorY = 0, workX = 0, workY = 0, workW = 1280, workH = 720;
        if (monitor)
        {
            glfwGetMonitorPos(monitor, &monitorX, &monitorY);
            glfwGetMonitorWorkarea(monitor, &workX, &workY, &workW, &workH);
        }
        const int screenW = current ? current->width : 1280, screenH = current ? current->height : 720;
        const int windowW = std::min(1280, workW), windowH = std::min(720, workH);
        GLFWwindow *window = glfwCreateWindow(windowW, windowH, title, nullptr, nullptr);
        if (!window)
            return nullptr;
        ps2_host_window::minimiseOnlyWhenThePlayerSwitchesAway(window);

        switch (settings.mode)
        {
        case ps2_host_settings::WindowMode::Window:
            glfwSetWindowPos(window, workX + (workW - windowW) / 2, workY + (workH - windowH) / 2);
            if (settings.maximised)
                glfwMaximizeWindow(window);
            glfwShowWindow(window);
            glfwFocusWindow(window); // to the front (started from another program, Windows may leave it behind)
            break;
        case ps2_host_settings::WindowMode::Borderless:
            glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
            glfwSetWindowPos(window, monitorX, monitorY);
            glfwSetWindowSize(window, screenW, screenH);
            glfwShowWindow(window);
            glfwFocusWindow(window);
            break;
        case ps2_host_settings::WindowMode::Fullscreen:
        {
            const int width = settings.fullscreenWidth > 0 ? settings.fullscreenWidth : screenW;
            const int height = settings.fullscreenHeight > 0 ? settings.fullscreenHeight : screenH;
            std::vector<int> rates;
            const int rate = monitor ? fullscreenRate(monitor, width, height, settings, rates) : GLFW_DONT_CARE;
            // the black cover first, the mode change under it
            glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
            glfwSetWindowPos(window, monitorX, monitorY);
            glfwSetWindowSize(window, screenW, screenH);
            glfwShowWindow(window);
            // in front before the mode change: Windows only lets the window in front take the screen over (left behind,
            // the game ran in the background and the screen kept its own rate)
            glfwFocusWindow(window);
            paintWindowBlack(window);
            glfwPollEvents();
            std::cerr << "[window] fullscreen " << width << "x" << height << " at " << rate << " Hz"
                      << refreshChoiceName(settings.refreshRate) << "; the screen has";
            for (int r : rates)
                std::cerr << " " << r;
            std::cerr << (rates.empty() ? " no mode of that size" : " Hz") << std::endl;
            if (monitor)
                glfwSetWindowMonitor(window, monitor, 0, 0, width, height, rate);
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN); // the game has no pointer
            break;
        }
        }
        paintWindowBlack(window);
        glfwPollEvents();
        const GLFWvidmode *now = monitor ? glfwGetVideoMode(monitor) : nullptr;
        std::cerr << "[window] " << ps2_host_settings::windowModeName(settings.mode) << " (Vulkan) on screen " << (index + 1)
                  << " of " << count << "; the screen now " << (now ? now->width : 0) << "x" << (now ? now->height : 0)
                  << " at " << (now ? now->refreshRate : 0) << " Hz; vsync " << (settings.vsync ? "on" : "off") << std::endl;
        return window;
    }
}
#endif

// VENDETTA_WINDOW_PACING=timer: the window refreshes on its own 60-a-second timer, as before (else once per new picture)
static bool windowTimerPacing()
{
    static const bool timer = []
    {
        const char *value = std::getenv("VENDETTA_WINDOW_PACING");
        return value && std::string(value) == "timer";
    }();
    return timer;
}

// The graphics card's newest picture (VENDETTA_GS=gpu): the texture takes its size, which grows with super-sampling.
static bool UploadGpuFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    static uint64_t s_lastSeq = 0u;
    static uint32_t s_width = 0u;
    static uint32_t s_height = 0u;
    const uint64_t previousSeq = s_lastSeq;
    const GSGpuFrame *frame = rt->gs().takeGpuFrame(s_lastSeq);
    if (frame && previousSeq != 0u && frame->seq > previousSeq + 1u)
        ps2_host_profile::count(ps2_host_profile::WindowSkips, frame->seq - previousSeq - 1u);
    if (frame && frame->shared)
    {
        // shown by the Vulkan presenter: nothing to upload here
        s_sharedFrame = frame;
        s_width = frame->width;
        s_height = frame->height;
        if (frame->modeWidth != 0u && frame->modeHeight != 0u && frame->internalWidth != 0u && frame->internalHeight != 0u)
            s_presentAspectScale = (static_cast<float>(frame->internalWidth) / static_cast<float>(frame->modeWidth)) /
                                   (static_cast<float>(frame->internalHeight) / static_cast<float>(frame->modeHeight));
    }
    else if (frame)
    {
        if (frame->width == 0u || frame->height == 0u)
        {
            // both display circuits off (after a GS reset until the game sets its display up again): black
            std::vector<uint8_t> black(static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height) * 4u, 0u);
            for (size_t i = 3u; i < black.size(); i += 4u)
                black[i] = 0xFFu;
            UpdateTexture(tex, black.data());
            s_width = static_cast<uint32_t>(tex.width);
            s_height = static_cast<uint32_t>(tex.height);
            s_presentAspectScale = 1.0f;
        }
        else
        {
            if (static_cast<uint32_t>(tex.width) != frame->width || static_cast<uint32_t>(tex.height) != frame->height)
            {
                UnloadTexture(tex);
                Image blank = GenImageColor(static_cast<int>(frame->width), static_cast<int>(frame->height), BLACK);
                tex = LoadTextureFromImage(blank);
                UnloadImage(blank);
                SetTextureFilter(tex, s_frameTextureFilter);
            }
            UpdateTexture(tex, frame->rgba.data());
            s_width = frame->width;
            s_height = frame->height;
            if (frame->modeWidth != 0u && frame->modeHeight != 0u && frame->internalWidth != 0u && frame->internalHeight != 0u)
                s_presentAspectScale = (static_cast<float>(frame->internalWidth) / static_cast<float>(frame->modeWidth)) /
                                       (static_cast<float>(frame->internalHeight) / static_cast<float>(frame->modeHeight));
        }
    }
    outWidth = (s_width != 0u) ? s_width : static_cast<uint32_t>(tex.width);
    outHeight = (s_height != 0u) ? s_height : static_cast<uint32_t>(tex.height);
    return frame != nullptr;
}

// true = a new picture was uploaded
static bool UploadFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    // the scaling filter can change at any time (the debug panel); FSR is done on the graphics card, the window then
    // only places its picture (drawn at exactly the window's size)
    const int wantedFilter =
        rt->gs().presentFilter() == GSPresentFilter::Nearest ? TEXTURE_FILTER_POINT : TEXTURE_FILTER_BILINEAR;
    if (wantedFilter != s_frameTextureFilter)
    {
        s_frameTextureFilter = wantedFilter;
        SetTextureFilter(tex, s_frameTextureFilter);
    }
#if defined(PS2X_NATIVE_HOST)
    if (GSD3D11Backend::IsActive())
    {
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        return GSD3D11Backend::TakeFreshFrame(); // DrawTexturePro (host/raylib_shim.cpp) shows it
    }
#endif
    if (rt->gs().showsGpuPicture())
        return UploadGpuFrame(tex, rt, outWidth, outHeight);
    // back from the graphics card's picture (VENDETTA_GS=both, switched in the debug panel): the CPU picture's texture
    if (tex.width != FB_WIDTH || tex.height != FB_HEIGHT)
    {
        UnloadTexture(tex);
        Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLACK);
        tex = LoadTextureFromImage(blank);
        UnloadImage(blank);
        SetTextureFilter(tex, s_frameTextureFilter);
        s_presentAspectScale = 1.0f;
    }
    // The newest picture the game thread presented (GS::presentAtSettlePoint), converted here; nothing new = keep
    // showing the last one. The main thread never touches the renderer.
    static uint64_t s_lastPresentedSeq = std::numeric_limits<uint64_t>::max();
    static uint32_t s_lastDisplayFbp = std::numeric_limits<uint32_t>::max();
    static uint32_t s_lastSourceFbp = std::numeric_limits<uint32_t>::max();
    static bool s_lastPreferred = false;
    static uint32_t s_lastWidth = 0u;
    static uint32_t s_lastHeight = 0u;
    static bool s_hasUploadedFrame = false;
    static std::vector<uint8_t> s_scratch;
    static std::vector<uint8_t> s_uploadBuffer(DEFAULT_FB_SIZE, 0u);

    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferredDisplaySource = false;
    const bool fresh = rt->gs().takePresentedFrame(s_lastPresentedSeq,
                                                   s_scratch,
                                                   width,
                                                   height,
                                                   &displayFbp,
                                                   &sourceFbp,
                                                   &usedPreferredDisplaySource);
    if (!fresh && s_hasUploadedFrame)
    {
        outWidth = (s_lastWidth != 0u) ? s_lastWidth : FB_WIDTH;
        outHeight = (s_lastHeight != 0u) ? s_lastHeight : DEFAULT_DISPLAY_HEIGHT;
        return false;
    }
    if (!fresh || s_scratch.empty() || width == 0u || height == 0u)
    {
        // nothing to show (no picture yet, or both display circuits off — after a GS reset until the game sets its
        // display up again): black, as on the console
        Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLACK);
        UpdateTexture(tex, blank.data);
        UnloadImage(blank);
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        s_lastWidth = outWidth;
        s_lastHeight = outHeight;
        s_hasUploadedFrame = true;
        return true;
    }

    PS2_IF_AGRESSIVE_LOGS({
        static uint32_t s_uploadDebugCount = 0u;
        if (s_uploadDebugCount < 128u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred ||
            width != s_lastWidth ||
            height != s_lastHeight)
        {
            std::cout << "[frame:upload] idx=" << s_uploadDebugCount
                      << " seq=" << s_lastPresentedSeq
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        ++s_uploadDebugCount;
    });
    s_lastDisplayFbp = displayFbp;
    s_lastSourceFbp = sourceFbp;
    s_lastPreferred = usedPreferredDisplaySource;
    s_lastWidth = width;
    s_lastHeight = height;

    std::fill(s_uploadBuffer.begin(), s_uploadBuffer.end(), 0u);
    if (!s_scratch.empty() && width != 0u && height != 0u)
    {
        const uint32_t copyWidth = std::min<uint32_t>(width, FB_WIDTH);
        const uint32_t copyHeight = std::min<uint32_t>(height, FB_HEIGHT);
        const size_t srcRowBytes = static_cast<size_t>(width) * 4u;
        const size_t dstRowBytes = static_cast<size_t>(FB_WIDTH) * 4u;
        const size_t copyRowBytes = static_cast<size_t>(copyWidth) * 4u;
        for (uint32_t y = 0; y < copyHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * srcRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * dstRowBytes;
            if (srcOffset + copyRowBytes > s_scratch.size() ||
                dstOffset + copyRowBytes > s_uploadBuffer.size())
            {
                break;
            }
            std::memcpy(s_uploadBuffer.data() + dstOffset, s_scratch.data() + srcOffset, copyRowBytes);
        }
    }

    UpdateTexture(tex, s_uploadBuffer.data());
    outWidth = width;
    outHeight = height;
    s_hasUploadedFrame = true;
    return true;
}

PS2Runtime::PS2Runtime()
{
    m_iopHost = std::make_unique<PS2IopHostAdapter>(*this);
    m_iopSubsystem = std::make_unique<ps2x::iop::IopSubsystem>(*m_iopHost);

    m_eeScheduler = std::make_unique<EeScheduler>(*this);

    // Assign rather than memset: R5900Context's constructor zeroes itself and
    // then applies the COP0 reset values, which a memset here would discard.
    m_cpuContext = R5900Context{};

    // R0 is always zero in MIPS
    m_cpuContext.r[0] = _mm_set1_epi32(0);
    m_cpuContext.vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    m_cpuContext.vu0_q = 1.0f;
    m_cpuContext.vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));

    // Stack pointer (SP) and global pointer (GP) will be set by the loaded ELF

    m_loadedModules.clear();
    m_guestHeapBlocks.clear();
    m_guestHeapBase = kGuestHeapDefaultBase;
    m_guestHeapEnd = kGuestHeapDefaultBase;
    m_guestHeapLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_guestHeapSuggestedBase = kGuestHeapDefaultBase;
    m_guestHeapConfigured = false;
    m_asyncCallbackStackFloor = kKernelCallbackStackFloor;
    m_asyncCallbackStackTop = kKernelCallbackStackTop;
}

void PS2Runtime::setDebugUiCallbacks(DebugUiCallback initCallback,
                                     DebugUiCallback drawCallback,
                                     DebugUiCallback shutdownCallback,
                                     void *userData)
{
    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    m_debugUiInitCallback = initCallback;
    m_debugUiDrawCallback = drawCallback;
    m_debugUiShutdownCallback = shutdownCallback;
    m_debugUiUserData = userData;
}

PS2Runtime::~PS2Runtime()
{
    try
    {
        requestStop();
        m_iopSubsystem.reset();
        m_iopHost.reset();
#if defined(PLATFORM_VITA)
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
#else
        if (IsAudioDeviceReady())
        {
            CloseAudioDevice();
            m_audioBackend.setAudioReady(false);
        }
#endif
        if (m_debugUiInitialized && m_debugUiShutdownCallback)
        {
            m_debugUiShutdownCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = false;
        }

        if (IsWindowReady())
        {
            CloseWindow();
        }

        m_loadedModules.clear();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: unknown" << std::endl;
    }
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModule(std::string_view path, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModule(path, arguments, argumentSize);
}

ps2x::iop::ModuleLoadResult PS2Runtime::loadIopModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->loadModuleBuffer(guestAddress, arguments, argumentSize);
}

bool PS2Runtime::stopIopModule(int32_t moduleId, int32_t *result)
{
    auto scope = m_iopHost->enterCall(nullptr, m_memory.getRDRAM());
    return m_iopSubsystem->stopModule(moduleId, result);
}

ps2x::iop::RpcAbi PS2Runtime::selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const
{
    return m_iopSubsystem->selectRpcAbi(request);
}

bool PS2Runtime::canBindIopRpc(uint32_t sid) const noexcept
{
    return m_iopSubsystem->canBindRpc(sid);
}

ps2x::iop::RpcResult PS2Runtime::handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request)
{
    auto scope = m_iopHost->enterCall(ctx, rdram);
    request.callToken = scope.token();
    return m_iopSubsystem->handleRpc(request);
}

bool PS2Runtime::sendIopSifCommand(uint8_t *rdram, R5900Context *ctx, uint32_t commandId, const void *packet, size_t packetSize)
{
    auto scope = m_iopHost->enterCall(ctx, rdram);
    return m_iopSubsystem->handleSifCommand(commandId, packet, packetSize);
}

void PS2Runtime::notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
{
    auto scope = m_iopHost->enterCall(nullptr, rdram);
    m_iopSubsystem->onSifTransfer(transfer);
}

void PS2Runtime::advanceIopEeCycles(uint64_t eeCycles) noexcept
{
    ps2_host_profile::Scope profile(ps2_host_profile::Iop);
    m_iopSubsystem->runEeCycles(eeCycles);
}

void PS2Runtime::resetIop()
{
    m_iopSubsystem->reset();
}

ps2x::iop::DebugSnapshot PS2Runtime::iopDebugSnapshot() const
{
    return m_iopSubsystem->debugSnapshot();
}

uint32_t PS2Runtime::allocateIopMemory(uint32_t size, uint32_t alignment)
{
    return m_iopSubsystem ? m_iopSubsystem->allocateMemory(size, alignment) : 0u;
}

bool PS2Runtime::freeIopMemory(uint32_t address)
{
    return m_iopSubsystem && m_iopSubsystem->freeMemory(address);
}

bool PS2Runtime::readIopMemory(uint32_t address, void *destination, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->readMemory(address, destination, size);
}

bool PS2Runtime::writeIopMemory(uint32_t address, const void *source, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->writeMemory(address, source, size);
}

bool PS2Runtime::zeroIopMemory(uint32_t address, size_t size)
{
    return m_iopSubsystem && m_iopSubsystem->zeroMemory(address, size);
}

bool PS2Runtime::isIopMemoryRange(uint32_t address, size_t size) const
{
    return m_iopSubsystem && m_iopSubsystem->isMemoryRange(address, size);
}

bool PS2Runtime::syncCoreSubsystems()
{
    uint8_t *const rdram = m_memory.getRDRAM();
    uint8_t *const gsVram = m_memory.getGSVRAM();
    if (!rdram || !gsVram)
    {
        return false;
    }

    if (m_boundRdram == rdram && m_boundGSVram == gsVram)
    {
        return true;
    }

    m_gs.init(gsVram, static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &m_memory.gs());
    m_gifArbiter.setProcessPacketFn([this](GifPathId path, const uint8_t *data, uint32_t size)
                                    { m_gs.processGIFPacket(data, size, static_cast<uint8_t>(path)); });
    m_memory.setGifArbiter(&m_gifArbiter);
    m_memory.setChainWorker(&m_chain);
    // VU1 runs where the VIF1 stream runs (the chain worker, or the game thread when inline): FBRST's DE1 / TE1 come
    // from the kick, and VPU_STAT's stop bits go back through the chain.
    m_memory.setVu1MscalCallback([this](uint32_t startPC, uint32_t top, uint32_t itop)
                                 {
                                     const uint32_t fbrst = m_chain.kickFbrst();
                                     m_vu1.state().dBitEnabled = (fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled = (fbrst & (1u << 11)) != 0u;
                                     if (hctpNativeVu1Mscal(m_memory.getVU1Code(), m_memory.getVU1Data(), startPC))
                                     {
                                         m_chain.noteVu1Run(0u);
                                         return;
                                     }
                                     if (hctpNativeVu1MscalGeneric(m_memory.getVU1Code(), m_memory.getVU1Data(), startPC, m_gs))
                                     {
                                         m_chain.noteVu1Run(0u);
                                         return;
                                     }
                                     hctpNativeVu1Census(m_memory.getVU1Code(), startPC, true);
                                     m_vu1.execute(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                   m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                   m_gs, &m_memory, startPC, top, itop, 65536);
                                     hctpNativeVu1AfterMscal(m_memory.getVU1Data());
                                     hctpNativeVu1AfterMscalGeneric(m_memory.getVU1Data());
                                     m_chain.noteVu1Run((m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                                        (m_vu1.state().stoppedByT ? 0x0400u : 0u)); });
    m_memory.setVu1MscntCallback([this](uint32_t top, uint32_t itop)
                                 {
                                     const uint32_t fbrst = m_chain.kickFbrst();
                                     m_vu1.state().dBitEnabled = (fbrst & (1u << 10)) != 0u;
                                     m_vu1.state().tBitEnabled = (fbrst & (1u << 11)) != 0u;
                                     // native p14 shadow batch (HCTP_NATIVE_P14=1): registers stay in the VU1 state
                                     if (!Vu1Compiled::xgkickActive(m_vu1) &&
                                         hctpNativeVu1MscntP14(m_memory.getVU1Code(), m_memory.getVU1Data(),
                                                               m_vu1.state(), top, m_gs))
                                     {
                                         m_chain.noteVu1Run(0u);
                                         return;
                                     }
                                     // native p08 chunk (HCTP_NATIVE_P08=1): drawn from the spec, the VU1 stays idle
                                     if (!Vu1Compiled::xgkickActive(m_vu1) &&
                                         hctpNativeVu1TryDraw(m_memory.getVU1Code(), m_memory.getVU1Data(),
                                                              m_vu1.state(), top, m_gs))
                                     {
                                         m_chain.noteVu1Run(0u);
                                         return;
                                     }
                                     hctpNativeVu1Census(m_memory.getVU1Code(), m_vu1.state().pc, false);
                                     hctpNativeVu1BeforeResume(m_memory.getVU1Code(), m_memory.getVU1Data(),
                                                               m_vu1.state(), top);
                                     m_vu1.resume(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                  m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                  m_gs, &m_memory, top, itop, 65536);
                                     hctpNativeVu1AfterResume(m_memory.getVU1Data());
                                     hctpNativeVu1AfterMscalGeneric(m_memory.getVU1Data()); // p14 verify
                                     m_chain.noteVu1Run((m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                                        (m_vu1.state().stoppedByT ? 0x0400u : 0u)); });
    resetIop();
    m_vu0.reset();
    m_vu1.reset();
    hctpNativeVu1Init();

    m_boundRdram = rdram;
    m_boundGSVram = gsVram;
    return true;
}

bool PS2Runtime::initialize(const char *title)
{
    try
    {
        if (!m_memory.initialize())
        {
            std::cerr << "Failed to initialize PS2 memory" << std::endl;
            return false;
        }

        if (!syncCoreSubsystems())
        {
            std::cerr << "Failed to bind runtime core subsystems" << std::endl;
            return false;
        }
#if defined(PLATFORM_VITA)
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title); // raylib vita does not support audio
#elif defined(PS2X_NATIVE_HOST)
        {
            // HCTP native-renderer build: Win32 window + Direct3D 11 + XInput + XAudio2 (host/win32_host.cpp,
            // settings in hctp_native.ini next to the game, defaults without it). HCTP_RENDERER=cpu: the CPU GS renderer
            // (its picture shown through the same window), else the Direct3D 11 GS renderer.
            const ps2_host_settings::Settings &settings = ps2_host_settings::get();
            ps2host::loadConfig(".\\hctp_native.ini"); // GetPrivateProfile*: a bare name would mean the Windows folder
            if (!ps2host::init(title))
            {
                std::cerr << "Failed to initialize the native Direct3D 11 host" << std::endl;
                return false;
            }
            m_audioBackend.setAudioReady(ps2host::audioReady());
            if (ps2host::audioReady())
                SetMasterVolume(static_cast<float>(settings.volume) / 100.0f);
            const char *renderer = std::getenv("HCTP_RENDERER");
            if (!renderer || std::strcmp(renderer, "cpu") != 0)
            {
                m_gs.setRasterBackend(std::make_unique<GSD3D11Backend>());
                std::cerr << "[runtime] GS: Direct3D 11 renderer (native host)" << std::endl;
            }
            else
                std::cerr << "[runtime] GS: CPU renderer (native host, HCTP_RENDERER=cpu)" << std::endl;
        }
#else
        const ps2_host_settings::Settings &settings = ps2_host_settings::get();
        // Made hidden, sized / made fullscreen as the settings ask, painted black, then shown: a window shown at once
        // appeared as a blank 640 x 448 box in the middle of the screen before going fullscreen.
        bool raylibMade = false;
#if defined(PS2X_VULKAN_PRESENT)
        GSGpuRenderer *gpu = m_gs.gpuRendererForSettings();
        if (vulkanPresentWanted() && gpu)
        {
            // raylib's window made hidden and kept so (sound); the game's window is the Vulkan one, with its presenter
            // made now: if either can't be, raylib's window is the game's as before (and the CPU renderer always)
            SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_HIDDEN);
            InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title);
            raylibMade = true;
            glfwSwapInterval(0);
            s_vulkanWindow = createVulkanWindow(title);
            std::string whyNot = "no window";
            if (s_vulkanWindow)
            {
                s_presenter = std::make_unique<VulkanPresenter>();
                if (!s_presenter->init(s_vulkanWindow, gpu->deviceName(), settings.vsync, whyNot))
                {
                    s_presenter.reset();
                    glfwDestroyWindow(s_vulkanWindow);
                    s_vulkanWindow = nullptr;
                }
            }
            if (s_vulkanWindow)
                host_input::useWindow(s_vulkanWindow); // the keyboard and controllers read from it
            else
            {
                std::cerr << "[present] the Vulkan window couldn't start (" << whyNot << "): raylib's window" << std::endl;
                stopSharedPictures();
                glfwSwapInterval(settings.vsync ? 1 : 0);
            }
        }
        if (!s_vulkanWindow)
        {
#endif
        if (!raylibMade)
        {
            SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_HIDDEN | (settings.vsync ? FLAG_VSYNC_HINT : 0u));
            InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title);
        }
        applyWindowSettings();
        BeginDrawing();
        ClearBackground(BLACK);
        EndDrawing();
        if (IsWindowState(FLAG_WINDOW_HIDDEN))
            ClearWindowState(FLAG_WINDOW_HIDDEN);
        for (int i = 0; i < 2; ++i) // a shown window keeps white until it is presented to
        {
            BeginDrawing();
            ClearBackground(BLACK);
            EndDrawing();
        }
#if defined(PS2X_VULKAN_PRESENT)
        }
#endif
        InitAudioDevice();
        m_audioBackend.setAudioReady(IsAudioDeviceReady());
        if (IsAudioDeviceReady())
            SetMasterVolume(static_cast<float>(settings.volume) / 100.0f);
#endif
        SetTargetFPS(windowTimerPacing() ? 60 : 0);
        if (m_debugUiInitCallback)
        {
            m_debugUiInitCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = true;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize PS2 runtime: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to initialize PS2 runtime: unknown exception" << std::endl;
    }

    return false;
}

bool PS2Runtime::loadELF(const std::string &elfPath)
{
    configureIoPathsFromElf(elfPath);

    std::ifstream file(elfPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open ELF file: " << elfPath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(ElfHeader)))
    {
        std::cerr << "ELF file is too small: " << elfPath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    ElfHeader header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Failed to read ELF header from: " << elfPath << std::endl;
        return false;
    }

    if (header.magic != ELF_MAGIC)
    {
        std::cerr << "Invalid ELF magic number" << std::endl;
        return false;
    }

    if (header.elf_class != 1u || header.endianness != 1u)
    {
        std::cerr << "Unsupported ELF format (expected 32-bit little-endian)." << std::endl;
        return false;
    }

    if (header.machine != EM_MIPS || header.type != ET_EXEC)
    {
        std::cerr << "Not a MIPS executable ELF file" << std::endl;
        return false;
    }

    if (header.phnum != 0u && header.phentsize < sizeof(ProgramHeader))
    {
        std::cerr << "Unsupported ELF program-header entry size: " << header.phentsize << std::endl;
        return false;
    }

    const uint64_t programHeaderTableEnd =
        static_cast<uint64_t>(header.phoff) +
        static_cast<uint64_t>(header.phnum) * static_cast<uint64_t>(header.phentsize);
    if (programHeaderTableEnd > static_cast<uint64_t>(fileSize))
    {
        std::cerr << "ELF program-header table is out of range." << std::endl;
        return false;
    }

    m_cpuContext.pc = header.entry;
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);

    uint32_t maxLoadedRdramEnd = kGuestHeapDefaultBase;
    uint32_t moduleBase = std::numeric_limits<uint32_t>::max();
    uint32_t moduleEnd = 0u;
    bool loadedAnySegment = false;

    for (uint16_t i = 0; i < header.phnum; i++)
    {
        const uint64_t phOffset =
            static_cast<uint64_t>(header.phoff) +
            static_cast<uint64_t>(i) * static_cast<uint64_t>(header.phentsize);
        if (phOffset + sizeof(ProgramHeader) > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF program header " << i << " is out of range." << std::endl;
            return false;
        }

        ProgramHeader ph{};
        file.seekg(static_cast<std::streamoff>(phOffset), std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(&ph), sizeof(ph)))
        {
            std::cerr << "Failed to read ELF program header " << i << std::endl;
            return false;
        }

        if (ph.type != PT_LOAD || ph.memsz == 0u)
        {
            continue;
        }

        if (ph.filesz > ph.memsz)
        {
            std::cerr << "ELF segment " << i << " has filesz > memsz." << std::endl;
            return false;
        }

        const uint64_t segmentFileEnd = static_cast<uint64_t>(ph.offset) + static_cast<uint64_t>(ph.filesz);
        if (segmentFileEnd > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF segment " << i << " exceeds file bounds." << std::endl;
            return false;
        }

        const bool scratch =
            ph.vaddr >= PS2_SCRATCHPAD_BASE &&
            ph.vaddr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);

        uint32_t physAddr = 0u;
        try
        {
            physAddr = m_memory.translateAddress(ph.vaddr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to translate ELF segment " << i
                      << " virtual address 0x" << std::hex << ph.vaddr
                      << std::dec << ": " << e.what() << std::endl;
            return false;
        }
        const uint64_t regionSize = scratch ? static_cast<uint64_t>(PS2_SCRATCHPAD_SIZE)
                                            : static_cast<uint64_t>(PS2_RAM_SIZE);
        const uint64_t segmentMemEnd = static_cast<uint64_t>(physAddr) + static_cast<uint64_t>(ph.memsz);
        if (segmentMemEnd > regionSize)
        {
            std::cerr << "ELF segment " << i << " exceeds "
                      << (scratch ? "scratchpad" : "RDRAM")
                      << " bounds (vaddr=0x" << std::hex << ph.vaddr
                      << " memsz=0x" << ph.memsz << std::dec << ")." << std::endl;
            return false;
        }

        uint8_t *destBase = scratch ? m_memory.getScratchpad() : m_memory.getRDRAM();
        if (!destBase)
        {
            std::cerr << "ELF segment " << i << " has no destination memory backing." << std::endl;
            return false;
        }

        uint8_t *dest = destBase + physAddr;
        if (ph.filesz > 0u)
        {
            file.seekg(static_cast<std::streamoff>(ph.offset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(dest), ph.filesz))
            {
                std::cerr << "Failed to read ELF segment " << i << " payload." << std::endl;
                return false;
            }
        }

        if (ph.memsz > ph.filesz)
        {
            std::memset(dest + ph.filesz, 0, ph.memsz - ph.filesz);
        }

        RUNTIME_LOG("Loading segment: 0x" << std::hex << ph.vaddr
                                          << " - 0x" << (static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz))
                                          << " (filesz: 0x" << ph.filesz
                                          << ", memsz: 0x" << ph.memsz << ")"
                                          << std::dec << std::endl);

        if (!scratch)
        {
            maxLoadedRdramEnd = std::max(maxLoadedRdramEnd, static_cast<uint32_t>(segmentMemEnd));
        }

        if (ph.flags & 0x1u) // PF_X
        {
            const uint64_t execEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.filesz);
            if (execEnd <= std::numeric_limits<uint32_t>::max())
            {
                m_memory.registerCodeRegion(ph.vaddr, static_cast<uint32_t>(execEnd));
            }
        }

        loadedAnySegment = true;
        moduleBase = std::min(moduleBase, ph.vaddr);
        const uint64_t segmentVirtualEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
        const uint32_t clampedVirtualEnd =
            (segmentVirtualEnd > std::numeric_limits<uint32_t>::max())
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(segmentVirtualEnd);
        moduleEnd = std::max(moduleEnd, clampedVirtualEnd);
    }

    if (!loadedAnySegment)
    {
        std::cerr << "ELF contains no loadable PT_LOAD segments." << std::endl;
        return false;
    }

    if (maxLoadedRdramEnd > PS2_RAM_SIZE)
    {
        maxLoadedRdramEnd = PS2_RAM_SIZE;
    }

    const uint32_t paddedEnd = (maxLoadedRdramEnd > (PS2_RAM_SIZE - kGuestHeapSafetyPad))
                                   ? PS2_RAM_SIZE
                                   : (maxLoadedRdramEnd + kGuestHeapSafetyPad);
    const uint32_t suggestedHeapBase = alignGuestHeapValue(paddedEnd, kGuestHeapDefaultAlignment);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        if (!m_guestHeapConfigured)
        {
            const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
            m_guestHeapSuggestedBase = std::min(suggestedHeapBase, hardLimit);
            m_guestHeapBase = m_guestHeapSuggestedBase;
            m_guestHeapEnd = m_guestHeapSuggestedBase;
            m_guestHeapLimit = hardLimit;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        m_asyncCallbackStackFloor = kKernelCallbackStackFloor;
        m_asyncCallbackStackTop = kKernelCallbackStackTop;
    }

    LoadedModule module;
    module.name = elfPath.substr(elfPath.find_last_of("/\\") + 1);
    module.baseAddress = (moduleBase == std::numeric_limits<uint32_t>::max()) ? 0x00100000u : moduleBase;
    module.size = (moduleEnd > module.baseAddress) ? static_cast<size_t>(moduleEnd - module.baseAddress) : 0u;
    module.active = true;

    m_loadedModules.push_back(module);

    uint32_t elfCrc32 = 0u;
    const bool elfCrc32Valid = computeFileCrc32(elfPath, elfCrc32);
    if (!elfCrc32Valid)
    {
        std::cerr << "[ps2xIOP] failed to compute ELF CRC32 for '" << elfPath << "'" << std::endl;
    }
    ps2x::iop::GameIdentity identity;
    identity.elfName = module.name;
    identity.entryPoint = m_cpuContext.pc;
    identity.crc32 = elfCrc32;
    std::string romError;
    if (!m_romDevice.configure(identity, &romError))
    {
        std::cerr << "[ROM0] failed to configure profile: " << romError << std::endl;
        return false;
    }

    m_iopSubsystem->reset();

    ps2_game_overrides::applyMatching(*this,
                                      elfPath,
                                      m_cpuContext.pc,
                                      elfCrc32,
                                      elfCrc32Valid);

    RUNTIME_LOG("ELF file loaded successfully. Entry point: 0x" << std::hex << m_cpuContext.pc << std::dec);
    return true;
}

void ps2_game_overrides::setWindowTitle(PS2Runtime &runtime, std::string_view title)
{
    (void)runtime;
    if (IsWindowReady())
        SetWindowTitle(std::string(title).c_str());
#if defined(PS2X_VULKAN_PRESENT)
    if (s_vulkanWindow)
        glfwSetWindowTitle(s_vulkanWindow, std::string(title).c_str());
#endif
}

const PS2Runtime::IoPaths &PS2Runtime::getIoPaths()
{
    return runtimeIoPaths();
}

void PS2Runtime::setIoPaths(const IoPaths &paths)
{
    IoPaths normalized = paths;
    normalized.elfPath = normalizeAbsolutePath(normalized.elfPath);
    normalized.elfDirectory = normalizeAbsolutePath(normalized.elfDirectory);
    normalized.hostRoot = normalizeAbsolutePath(normalized.hostRoot);
    normalized.cdRoot = normalizeAbsolutePath(normalized.cdRoot);
    normalized.mcRoot = normalizeAbsolutePath(normalized.mcRoot);
    normalized.cdImage = normalizeAbsolutePath(normalized.cdImage);

    if (normalized.elfDirectory.empty() && !normalized.elfPath.empty())
    {
        normalized.elfDirectory = normalized.elfPath.parent_path();
    }

    if (normalized.hostRoot.empty())
    {
        normalized.hostRoot = normalized.elfDirectory;
    }
    if (normalized.cdRoot.empty())
    {
        normalized.cdRoot = normalized.elfDirectory;
    }
    if (normalized.mcRoot.empty())
    {
        normalized.mcRoot = normalized.elfDirectory / "mc0";
    }

    runtimeIoPaths() = normalized;
}

void PS2Runtime::configureIoPathsFromElf(const std::string &elfPath)
{
    IoPaths paths = runtimeIoPaths();
    paths.elfPath = normalizeAbsolutePath(std::filesystem::path(elfPath));
    if (!paths.elfPath.empty())
    {
        paths.elfDirectory = paths.elfPath.parent_path();
    }

    if (!paths.elfDirectory.empty())
    {
        paths.hostRoot = paths.elfDirectory;
        paths.cdRoot = paths.elfDirectory;
        paths.mcRoot = paths.elfDirectory / "mc0";
    }

    setIoPaths(paths);
}

namespace
{
    bool generatedFunctionTableSlot(uint32_t address, uint32_t &slot)
    {
        if ((address & 3u) != 0u || g_ps2RecompiledFunctionTableSlotCount == 0u)
        {
            return false;
        }

        if (address < g_ps2RecompiledFunctionTableBase || address >= g_ps2RecompiledFunctionTableEnd)
        {
            return false;
        }

        const uint32_t offset = address - g_ps2RecompiledFunctionTableBase;
        slot = offset >> 2;
        return slot < g_ps2RecompiledFunctionTableSlotCount;
    }
}

bool PS2Runtime::replaceFunction(uint32_t address, RecompiledFunction func)
{
    uint32_t slot = 0u;
    if (!generatedFunctionTableSlot(address, slot))
    {
        std::cerr << "[function-table] cannot replace guest PC 0x" << std::hex << address
                  << ": outside generated dense table [0x" << g_ps2RecompiledFunctionTableBase
                  << ", 0x" << g_ps2RecompiledFunctionTableEnd << ")"
                  << std::dec << std::endl;
        return false;
    }

    g_ps2RecompiledFunctionTable[slot] = func;
    return true;
}

bool PS2Runtime::registerFunction(uint32_t address, RecompiledFunction func)
{
    return replaceFunction(address, func);
}

bool PS2Runtime::hasFunction(uint32_t address) const
{
    uint32_t slot = 0u;
    return generatedFunctionTableSlot(address, slot) && g_ps2RecompiledFunctionTable[slot] != nullptr;
}

const char *describeGuestBranchKind(PS2Runtime::GuestBranchKind kind)
{
    switch (kind)
    {
    case PS2Runtime::GuestBranchKind::DirectJump:
        return "DirectJump";
    case PS2Runtime::GuestBranchKind::DirectCall:
        return "DirectCall";
    case PS2Runtime::GuestBranchKind::IndirectJump:
        return "IndirectJump";
    case PS2Runtime::GuestBranchKind::IndirectCall:
        return "IndirectCall";
    case PS2Runtime::GuestBranchKind::Return:
        return "Return";
    default:
        return "Unknown";
    }
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(uint32_t address)
{
    pushDispatchPc(address);

    uint32_t slot = 0u;
    if (generatedFunctionTableSlot(address, slot))
    {
        RecompiledFunction fn = g_ps2RecompiledFunctionTable[slot];
        if (fn != nullptr)
        {
            return fn;
        }
    }

    std::cerr << "Error: No exact recompiled function for guest PC 0x" << std::hex << address
              << " tableBase=0x" << g_ps2RecompiledFunctionTableBase
              << " tableEnd=0x" << g_ps2RecompiledFunctionTableEnd
              << " codeRegion=" << (m_memory.isCodeAddress(address) ? "yes" : "no")
              << " trace=" << formatDispatchHistory()
              << std::dec << std::endl;

    static RecompiledFunction missingFunction = [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t badPc = ctx->pc;
        runtime->reportMissingFunction(rdram,
                                       ctx,
                                       badPc,
                                       0u,
                                       PS2Runtime::GuestBranchKind::IndirectJump,
                                       "dispatch");
    };

    return missingFunction;
}

void PS2Runtime::setMissingFunctionPolicy(MissingFunctionPolicy policy)
{
    m_missingFunctionPolicy.store(static_cast<uint32_t>(policy), std::memory_order_release);
}

PS2Runtime::MissingFunctionPolicy PS2Runtime::missingFunctionPolicy() const
{
    return static_cast<MissingFunctionPolicy>(m_missingFunctionPolicy.load(std::memory_order_acquire));
}

void PS2Runtime::resetMissingFunctionReportOnce()
{
    m_missingFunctionReported.store(false, std::memory_order_release);
}

void PS2Runtime::reportMissingFunction(uint8_t *rdram,
                                       R5900Context *ctx,
                                       uint32_t targetPc,
                                       uint32_t sourcePc,
                                       GuestBranchKind kind,
                                       const char *debugName)
{
    const MissingFunctionPolicy policy = missingFunctionPolicy();
    const bool firstReport = !m_missingFunctionReported.exchange(true, std::memory_order_acq_rel);

    const uint32_t pc = ctx->pc;
    const uint32_t ra = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));
    const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
    const uint32_t gp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0));
    const uint32_t a0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[4], 0));
    const uint32_t a1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[5], 0));
    const uint32_t a2 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[6], 0));
    const uint32_t a3 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[7], 0));
    const uint32_t s0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[16], 0));
    const uint32_t s1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[17], 0));
    const uint32_t v0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[2], 0));
    const uint32_t v1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[3], 0));

    auto readGuestU32At = [rdram](uint32_t addr, uint32_t &out) -> bool
    {
        // TODO this !rdram exist only because of test fix those test later
        if (!rdram || addr > PS2_RAM_SIZE - sizeof(uint32_t))
        {
            out = 0u;
            return false;
        }

        std::memcpy(&out, rdram + addr, sizeof(uint32_t));
        return true;
    };

    auto readGuestU32Offset = [&readGuestU32At](uint32_t base, uint32_t offset, uint32_t &out) -> bool
    {
        if (base > PS2_RAM_SIZE - sizeof(uint32_t) || offset > PS2_RAM_SIZE - sizeof(uint32_t) - base)
        {
            out = 0u;
            return false;
        }

        return readGuestU32At(base + offset, out);
    };

    uint32_t a0Word0 = 0u;
    uint32_t a0Word4 = 0u;
    uint32_t a0Word8 = 0u;
    uint32_t a0WordC = 0u;
    const bool a0Readable =
        readGuestU32Offset(a0, 0x00u, a0Word0) &&
        readGuestU32Offset(a0, 0x04u, a0Word4) &&
        readGuestU32Offset(a0, 0x08u, a0Word8) &&
        readGuestU32Offset(a0, 0x0cu, a0WordC);

    uint32_t s0Word0 = 0u;
    uint32_t s0Word4 = 0u;
    uint32_t s0Word8 = 0u;
    uint32_t s0WordC = 0u;
    const bool s0Readable =
        readGuestU32Offset(s0, 0x00u, s0Word0) &&
        readGuestU32Offset(s0, 0x04u, s0Word4) &&
        readGuestU32Offset(s0, 0x08u, s0Word8) &&
        readGuestU32Offset(s0, 0x0cu, s0WordC);

    uint32_t recordWord0 = 0u;
    uint32_t recordWord4 = 0u;
    uint32_t recordWord8 = 0u;
    uint32_t recordWordC = 0u;
    const bool recordReadable =
        s0Readable && s0Word4 != 0u &&
        readGuestU32Offset(s0Word4, 0x00u, recordWord0) &&
        readGuestU32Offset(s0Word4, 0x04u, recordWord4) &&
        readGuestU32Offset(s0Word4, 0x08u, recordWord8) &&
        readGuestU32Offset(s0Word4, 0x0cu, recordWordC);

    uint32_t vtableSlot0 = 0u;
    uint32_t vtableSlot4 = 0u;
    uint32_t vtableSlot8 = 0u;
    uint32_t vtableSlotC = 0u;
    const bool vtableReadable =
        a0Readable && a0Word0 != 0u &&
        readGuestU32Offset(a0Word0, 0x00u, vtableSlot0) &&
        readGuestU32Offset(a0Word0, 0x04u, vtableSlot4) &&
        readGuestU32Offset(a0Word0, 0x08u, vtableSlot8) &&
        readGuestU32Offset(a0Word0, 0x0cu, vtableSlotC);

    if (firstReport)
    {
        std::ostringstream oss;
        oss << "[guest-branch:missing-target] kind=" << describeGuestBranchKind(kind)
            << " op=" << (debugName ? debugName : "<unknown>")
            << " source=0x" << std::hex << sourcePc
            << " target=0x" << targetPc
            << " pc=0x" << pc
            << " ra=0x" << ra
            << " sp=0x" << sp
            << " gp=0x" << gp
            << " a0=0x" << a0
            << " a1=0x" << a1
            << " a2=0x" << a2
            << " a3=0x" << a3
            << " s0=0x" << s0
            << " s1=0x" << s1
            << " v0=0x" << v0
            << " v1=0x" << v1
            << " a0Readable=" << (a0Readable ? "yes" : "no")
            << " a0[0]=0x" << a0Word0
            << " a0[4]=0x" << a0Word4
            << " a0[8]=0x" << a0Word8
            << " a0[c]=0x" << a0WordC
            << " s0Readable=" << (s0Readable ? "yes" : "no")
            << " s0[0]=0x" << s0Word0
            << " s0[4]=0x" << s0Word4
            << " s0[8]=0x" << s0Word8
            << " s0[c]=0x" << s0WordC
            << " recordReadable=" << (recordReadable ? "yes" : "no")
            << " record[0]=0x" << recordWord0
            << " record[4]=0x" << recordWord4
            << " record[8]=0x" << recordWord8
            << " record[c]=0x" << recordWordC
            << " vtableReadable=" << (vtableReadable ? "yes" : "no")
            << " vtbl[0]=0x" << vtableSlot0
            << " vtbl[4]=0x" << vtableSlot4
            << " vtbl[8]=0x" << vtableSlot8
            << " vtbl[c]=0x" << vtableSlotC
            << " codeRegion=" << (m_memory.isCodeAddress(targetPc) ? "yes" : "no")
            << " policy=" << static_cast<uint32_t>(policy)
            << " trace=" << formatDispatchHistory()
            << std::dec;
        // which guest thread it was and what every thread was doing (callback nesting = invocationDepth)
        if (m_eeScheduler)
        {
            const EeKernelSnapshot snap = m_eeScheduler->snapshot();
            oss << " runningThread=" << snap.runningThreadId << " threads:";
            for (const EeThreadSnapshot &t : snap.threads)
                oss << " [" << t.id << " entry=0x" << std::hex << t.entry << " pc=0x" << t.pc << " ra=0x" << t.ra
                    << std::dec << " st=" << static_cast<int>(t.status) << " wait=" << static_cast<int>(t.waitReason)
                    << "/" << t.waitId << " depth=" << t.invocationDepth << "]";
        }

        static std::mutex s_missingFunctionLogMutex;
        {
            std::lock_guard<std::mutex> lock(s_missingFunctionLogMutex);
            std::cerr << oss.str() << std::endl;
        }
    }

    if (firstReport && policy == MissingFunctionPolicy::BreakOnce)
    {
#if defined(_MSC_VER)
        __debugbreak();
#endif // TODO others breakpoints
    }

    if (ctx)
    {
        ctx->pc = targetPc;
    }

    if (policy == MissingFunctionPolicy::Stop)
    {
        requestStop();
    }
}

bool PS2Runtime::dispatchGuestBranch(uint8_t *rdram,
                                     R5900Context *ctx,
                                     uint32_t targetPc,
                                     uint32_t sourcePc,
                                     uint32_t fallthroughPc,
                                     GuestBranchKind kind,
                                     const char *debugName)
{
    ctx->pc = targetPc;
    const bool isCall = (kind == GuestBranchKind::DirectCall || kind == GuestBranchKind::IndirectCall);

    // Every inter-function transfer is also a deterministic EE safe point.
    // Backward edges inside generated functions use eeCheckpointDue(), while
    // this charge bounds straight-line call chains that have no local loop.
    if (m_eeScheduler && m_eeScheduler->checkpointDue(EeScheduler::kGuestDispatchCycles))
    {
        return false;
    }

    if (!isCall)
    {
        if (!hasFunction(targetPc))
        {
            reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);
        }

        ctx->pc = targetPc;
        return false;
    }

    if (!hasFunction(targetPc))
    {
        reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);

        const MissingFunctionPolicy policy = missingFunctionPolicy();

        if (policy == MissingFunctionPolicy::SkipCallDebug && isCall)
        {
            ctx->pc = fallthroughPc;
            return true;
        }

        if (policy == MissingFunctionPolicy::ContinueToTarget)
        {
            ctx->pc = targetPc;
            // if you need the app to keep open to open debug pannel change this to false
            return false;
        }

        return false;
    }

    RecompiledFunction targetFn = lookupFunction(targetPc);
    const uint32_t entryPc = ctx->pc;
    const uint32_t entrySp = getRegU32(ctx, 29);
    const uint32_t entryRa = getRegU32(ctx, 31);
    if (!g_recordDispatches)
    {
        targetFn(rdram, ctx, this);
    }
    else
    {
        const int thread = m_eeScheduler ? m_eeScheduler->currentThreadId() : 0;
        ps2RecordDispatch('C', targetPc, entrySp, entryRa, thread);
        targetFn(rdram, ctx, this);
        ps2RecordDispatch('R', ctx->pc, getRegU32(ctx, 29), getRegU32(ctx, 31), thread);
    }

    if (isStopRequested() || ctx->pc == 0u)
    {
        return false;
    }

    // A native stub returns without touching pc (it is still the call's target) and leaves sp / ra as they were.
    // A pc equal to the target can also mean a DEFERRED call deeper down: the callee (or something it called)
    // reached a call of this same function, the call-site checkpoint fired, and pc names that not-yet-made call
    // — with the inner frame's sp and the inner return address in ra. Taking that for "returned" resumed the caller
    // on the inner frame (Def Jam Vendetta's recursive sound routine 0x1CEB18: the caller 0x1CEAF8 loaded ra from
    // another frame, returned to 0 and the sound thread ended — the match freezes).
    if (ctx->pc == entryPc && getRegU32(ctx, 29) == entrySp && getRegU32(ctx, 31) == entryRa)
    {
        ctx->pc = fallthroughPc;
    }

    return ctx->pc == fallthroughPc;
}

void PS2Runtime::SignalException(R5900Context *ctx, PS2Exception exception)
{
    if (exception == EXCEPTION_INTEGER_OVERFLOW)
    {
        HandleIntegerOverflow(ctx);
        return;
    }

    raiseCop0Exception(ctx, static_cast<uint32_t>(exception),
                       exception == EXCEPTION_TLB_REFILL);
}

void PS2Runtime::executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    m_vu0.reset();
    copyVu0ContextToState(ctx, m_vu0.state());
    // TEMP diag (HCTP_VU0LOG=1): first 24 VCALLMS - code words at the start PC, vf1/vf2 in, vf28-30 out (old fork [vu0io])
    static const bool s_vu0Log = std::getenv("HCTP_VU0LOG") != nullptr;
    static uint32_t s_vu0Calls = 0u;
    static __m128 s_vu0Before[32];
    const bool logThis = s_vu0Log && s_vu0Calls++ < 24u;
    auto logVf = [ctx](const char *tag, int r) {
        alignas(16) float f[4];
        _mm_store_ps(f, ctx->vu0_vf[r]);
        std::fprintf(stderr, " %s%d=(%g,%g,%g,%g)", tag, r, f[0], f[1], f[2], f[3]);
    };
    if (logThis)
    {
        uint32_t w[4];
        std::memcpy(w, vu0Code + startPC, 16);
        std::fprintf(stderr, "[vu0io] in pc=0x%x code=%08x %08x %08x %08x", startPC, w[0], w[1], w[2], w[3]);
        logVf("vf", 1); logVf("vf", 2); logVf("vf", 20);
        std::fprintf(stderr, "\n");
        for (int r = 0; r < 32; ++r)
            s_vu0Before[r] = ctx->vu0_vf[r];
    }
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop, 4096);
    copyVu0StateToContext(m_vu0.state(), ctx);
    if (logThis)
    {
        std::fprintf(stderr, "[vu0io] ran cycles=%llu endpc=0x%x changed:", static_cast<unsigned long long>(m_vu0.state().cycles), m_vu0.state().pc);
        for (int r = 1; r < 32; ++r)
            if (std::memcmp(&s_vu0Before[r], &ctx->vu0_vf[r], 16) != 0)
                std::fprintf(stderr, " vf%d", r);
        std::fprintf(stderr, " vi1-4=%d,%d,%d,%d\n", ctx->vi[1], ctx->vi[2], ctx->vi[3], ctx->vi[4]);
        std::fprintf(stderr, "[vu0io] out");
        logVf("vf", 28); logVf("vf", 29); logVf("vf", 30); logVf("vf", 31);
        std::fprintf(stderr, "\n");
    }
}

void PS2Runtime::vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    // VCALLMS and VCALLMSR both route here.
    executeVU0Microprogram(rdram, ctx, address);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx)
{
    handleSyscall(rdram, ctx, 0);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId)
{
    if (ctx->in_delay_slot)
    {
        throw std::runtime_error("Attempted to execute a syscall inside a branch delay slot! "
                                 "This breaks the atomic basic block model and is structurally unsupported by the emulator.");
    }

    const uint32_t syscallId = (encodedSyscallId != 0u)
                                   ? encodedSyscallId
                                   : getRegU32(ctx, 3); // $v1 / $3 is the EE kernel syscall number

    if (ps2_syscalls::dispatchNumericSyscall(syscallId, rdram, ctx, this))
    {
        return;
    }

    // God help you
    ps2_syscalls::TODO(rdram, ctx, this, encodedSyscallId);
}

void PS2Runtime::handleBreak(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_BREAKPOINT);
}

void PS2Runtime::drainCompletedDmacHandlers(uint8_t *rdram)
{
    for (uint32_t cause : m_memory.consumeCompletedDmacCauses())
    {
        ps2_syscalls::dispatchDmacHandlersForCause(rdram, this, cause);
    }
}

void PS2Runtime::handleTrap(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_TRAP);
}

void PS2Runtime::handleTLBR(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t vpn = 0;
    uint32_t pfn = 0;
    uint32_t mask = 0;
    bool valid = false;

    const uint32_t index = ctx->cop0_index & 0x3Fu;
    if (!m_memory.tlbRead(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Preserve low ASID bits in EntryHi.
    ctx->cop0_entryhi = (ctx->cop0_entryhi & 0x00000FFFu) | (vpn & 0xFFFFF000u);
    ctx->cop0_entrylo0 = (ctx->cop0_entrylo0 & ~0x03FFFFC2u) |
                         ((pfn & 0x000FFFFFu) << 6) |
                         (valid ? 0x2u : 0u);
    ctx->cop0_pagemask = mask & 0x01FFE000u;
}

void PS2Runtime::handleTLBWI(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t index = ctx->cop0_index & 0x3Fu;
    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
    }
}

void PS2Runtime::handleTLBWR(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t entryCount = static_cast<uint32_t>(m_memory.tlbEntryCount());
    if (entryCount == 0)
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    const uint32_t wired = std::min(ctx->cop0_wired, entryCount - 1);
    uint32_t random = ctx->cop0_random % entryCount;
    if (random < wired)
    {
        random = wired;
    }

    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(random, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Keep COP0 bookkeeping in sync with the selected slot.
    ctx->cop0_index = (ctx->cop0_index & ~0x3Fu) | (random & 0x3Fu);
    ctx->cop0_random = (random <= wired) ? (entryCount - 1) : (random - 1);
}

void PS2Runtime::handleTLBP(uint8_t *rdram, R5900Context *ctx)
{
    const int32_t index = m_memory.tlbProbe(ctx->cop0_entryhi & 0xFFFFF000u);
    if (index >= 0)
    {
        ctx->cop0_index = (ctx->cop0_index & ~0x8000003Fu) |
                          (static_cast<uint32_t>(index) & 0x3Fu);
    }
    else
    {
        // MIPS sets probe failure bit (P) in Index[31].
        ctx->cop0_index |= 0x80000000u;
    }
}

void PS2Runtime::clearLLBit(R5900Context *ctx)
{
    // LL/SC reservation is tracked separately from COP0 Status.
    ctx->llbit = 0;
    ctx->lladdr = 0;
}

uint32_t PS2Runtime::alignGuestHeapValue(uint32_t value, uint32_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    const uint32_t mask = alignment - 1u;
    if (value > (std::numeric_limits<uint32_t>::max() - mask))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return (value + mask) & ~mask;
}

bool PS2Runtime::isGuestHeapAlignmentValid(uint32_t alignment)
{
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

uint32_t PS2Runtime::normalizeGuestHeapAlignment(uint32_t alignment)
{
    if (!isGuestHeapAlignmentValid(alignment))
    {
        return kGuestHeapDefaultAlignment;
    }
    return std::max(alignment, kGuestHeapDefaultAlignment);
}

uint32_t PS2Runtime::clampGuestHeapBase(uint32_t guestBase) const
{
    uint32_t normalized = guestBase;
    if (normalized >= PS2_RAM_SIZE)
    {
        normalized &= PS2_RAM_MASK;
    }
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    return std::min(normalized, hardLimit);
}

uint32_t PS2Runtime::clampGuestHeapLimit(uint32_t guestLimit) const
{
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    if (guestLimit == 0u || guestLimit > hardLimit)
    {
        return hardLimit;
    }
    return guestLimit;
}

void PS2Runtime::resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit)
{
    uint32_t base = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    uint32_t limit = clampGuestHeapLimit(guestLimit);
    if (base == 0u)
    {
        const uint32_t fallbackBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
        base = alignGuestHeapValue(clampGuestHeapBase(fallbackBase), kGuestHeapDefaultAlignment);
    }

    if (limit <= base)
    {
        base = alignGuestHeapValue(clampGuestHeapBase(m_guestHeapSuggestedBase), kGuestHeapDefaultAlignment);
        limit = clampGuestHeapLimit(0u);
    }

    if (limit <= base)
    {
        base = 0u;
        limit = 0u;
    }

    m_guestHeapBlocks.clear();
    if (limit > base)
    {
        m_guestHeapBlocks.push_back({base, limit - base, true});
    }
    // HCTP (old-fork fix, Oct 10/11): the game's ymem heap grabs all user RAM with one big malloc, so later libc
    // mallocs (strtod Bigint -> endless _lshift loop) got NULL. Unused kernel RAM 0xA0000-0xFF000 is a fallback pool,
    // kept LAST so first-fit only uses it once the main heap is full (not address-adjacent: never coalesced with it).
    if (base > kGuestLowPoolBase + kGuestLowPoolSize)
    {
        m_guestHeapBlocks.push_back({kGuestLowPoolBase, kGuestLowPoolSize, true});
    }

    m_guestHeapBase = base;
    m_guestHeapEnd = base;
    m_guestHeapLimit = limit;
    m_guestHeapConfigured = true;
}

void PS2Runtime::ensureGuestHeapInitializedLocked()
{
    if (m_guestHeapConfigured)
    {
        return;
    }

    const uint32_t suggested = (m_guestHeapSuggestedBase == 0u) ? kGuestHeapDefaultBase : m_guestHeapSuggestedBase;
    resetGuestHeapLocked(suggested, clampGuestHeapLimit(0u));
}

int32_t PS2Runtime::findGuestHeapBlockIndexLocked(uint32_t guestAddr) const
{
    const uint32_t normalizedAddr = guestAddr & PS2_RAM_MASK;
    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock &block = m_guestHeapBlocks[i];
        if (!block.free && block.addr == normalizedAddr)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

uint32_t PS2Runtime::allocateGuestBlockLocked(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    if (size > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock block = m_guestHeapBlocks[i];
        if (!block.free)
        {
            continue;
        }

        const uint64_t blockStart = block.addr;
        const uint64_t blockEnd = blockStart + static_cast<uint64_t>(block.size);
        const uint32_t alignedAddr = alignGuestHeapValue(block.addr, normalizedAlignment);
        if (alignedAddr < block.addr)
        {
            continue;
        }

        const uint64_t alignedStart = alignedAddr;
        if (alignedStart > blockEnd)
        {
            continue;
        }

        const uint64_t allocEnd = alignedStart + static_cast<uint64_t>(allocSize);
        if (allocEnd > blockEnd)
        {
            continue;
        }

        const uint32_t prefixSize = static_cast<uint32_t>(alignedStart - blockStart);
        const uint32_t suffixSize = static_cast<uint32_t>(blockEnd - allocEnd);

        std::vector<GuestHeapBlock> replacement;
        replacement.reserve(3);
        if (prefixSize > 0u)
        {
            replacement.push_back({block.addr, prefixSize, true});
        }
        replacement.push_back({alignedAddr, allocSize, false});
        if (suffixSize > 0u)
        {
            replacement.push_back({static_cast<uint32_t>(allocEnd), suffixSize, true});
        }

        m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
        m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i),
                                 replacement.begin(),
                                 replacement.end());

        m_guestHeapEnd = std::max(m_guestHeapEnd, static_cast<uint32_t>(allocEnd));
        return alignedAddr;
    }

    return 0u;
}

void PS2Runtime::coalesceGuestHeapLocked()
{
    if (m_guestHeapBlocks.empty())
    {
        return;
    }

    size_t i = 1;
    while (i < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &prev = m_guestHeapBlocks[i - 1];
        GuestHeapBlock &curr = m_guestHeapBlocks[i];
        const uint64_t prevEnd = static_cast<uint64_t>(prev.addr) + static_cast<uint64_t>(prev.size);
        if (prev.free && curr.free && prevEnd == curr.addr)
        {
            prev.size += curr.size;
            m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void PS2Runtime::freeGuestBlockLocked(uint32_t guestAddr)
{
    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return;
    }

    m_guestHeapBlocks[static_cast<size_t>(index)].free = true;
    coalesceGuestHeapLocked();
}

void PS2Runtime::configureGuestHeap(uint32_t guestBase, uint32_t guestLimit)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    uint32_t normalizedBase = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    if (normalizedBase == 0u)
    {
        normalizedBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
    }
    m_guestHeapSuggestedBase = normalizedBase;
    resetGuestHeapLocked(normalizedBase, guestLimit);
}

uint32_t PS2Runtime::guestMalloc(uint32_t size, uint32_t alignment)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    return allocateGuestBlockLocked(size, alignment);
}

uint32_t PS2Runtime::guestCalloc(uint32_t count, uint32_t size, uint32_t alignment)
{
    if (count == 0u || size == 0u)
    {
        return 0u;
    }
    if (count > (std::numeric_limits<uint32_t>::max() / size))
    {
        return 0u;
    }

    const uint32_t totalSize = count * size;
    const uint32_t guestAddr = guestMalloc(totalSize, alignment);
    if (guestAddr != 0u)
    {
        uint8_t *rdram = m_memory.getRDRAM();
        if (rdram)
        {
            uint32_t physAddr = guestAddr & PS2_RAM_MASK;
            if (physAddr + totalSize <= PS2_RAM_SIZE)
                std::memset(rdram + physAddr, 0, totalSize);
        }
    }

    return guestAddr;
}

uint32_t PS2Runtime::guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment)
{
    if (guestAddr == 0u)
    {
        return guestMalloc(newSize, alignment);
    }
    if (newSize == 0u)
    {
        guestFree(guestAddr);
        return 0u;
    }

    if (newSize > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t requestedSize = alignGuestHeapValue(newSize, kGuestHeapDefaultAlignment);

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();

    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return 0u;
    }

    const size_t blockIndex = static_cast<size_t>(index);
    const uint32_t oldAddr = m_guestHeapBlocks[blockIndex].addr;
    const uint32_t oldSize = m_guestHeapBlocks[blockIndex].size;

    if (requestedSize <= oldSize)
    {
        if (requestedSize < oldSize)
        {
            const uint32_t tailAddr = oldAddr + requestedSize;
            const uint32_t tailSize = oldSize - requestedSize;
            m_guestHeapBlocks[blockIndex].size = requestedSize;
            m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u),
                                     GuestHeapBlock{tailAddr, tailSize, true});
            coalesceGuestHeapLocked();
        }
        return oldAddr;
    }

    if (blockIndex + 1u < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &next = m_guestHeapBlocks[blockIndex + 1u];
        const uint64_t blockEnd = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].addr) +
                                  static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size);
        if (next.free && blockEnd == next.addr)
        {
            const uint64_t combined = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size) +
                                      static_cast<uint64_t>(next.size);
            if (combined >= requestedSize)
            {
                const uint32_t extraNeeded = requestedSize - m_guestHeapBlocks[blockIndex].size;
                m_guestHeapBlocks[blockIndex].size = requestedSize;
                if (next.size == extraNeeded)
                {
                    m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u));
                }
                else
                {
                    next.addr += extraNeeded;
                    next.size -= extraNeeded;
                }
                m_guestHeapEnd = std::max(m_guestHeapEnd, oldAddr + requestedSize);
                return oldAddr;
            }
        }
    }

    const uint32_t newAddr = allocateGuestBlockLocked(newSize, normalizedAlignment);
    if (newAddr == 0u)
    {
        return 0u;
    }

    uint8_t *rdram = m_memory.getRDRAM();
    if (rdram)
    {
        const uint32_t copyBytes = std::min(oldSize, newSize);
        uint32_t dstPhys = newAddr & PS2_RAM_MASK;
        uint32_t srcPhys = oldAddr & PS2_RAM_MASK;
        if (dstPhys + copyBytes <= PS2_RAM_SIZE && srcPhys + copyBytes <= PS2_RAM_SIZE)
            std::memmove(rdram + dstPhys, rdram + srcPhys, copyBytes);
    }

    freeGuestBlockLocked(oldAddr);
    return newAddr;
}

// A ring of the last dispatches with the stack pointer and return address (what = 'D' dispatcher, 'C' call,
// 'R' back from a call) — printed when a guest thread ends unexpectedly.
namespace
{
    struct DispatchRecord
    {
        uint32_t pc, sp, ra;
        char what;
        int thread;
    };
    std::array<DispatchRecord, 64> g_dispatchRecords{};
    uint32_t g_dispatchRecordNext = 0u;
}

void ps2RecordDispatch(char what, uint32_t pc, uint32_t sp, uint32_t ra, int thread)
{
    g_dispatchRecords[g_dispatchRecordNext++ % g_dispatchRecords.size()] = DispatchRecord{pc, sp, ra, what, thread};
}

bool ps2DispatchRecording() noexcept
{
    return g_recordDispatches;
}

std::string ps2FormatDispatchRecords()
{
    if (!g_recordDispatches)
        return " (off: VENDETTA_DISPATCHLOG=1 records the last calls, returns and dispatches)";
    std::ostringstream out;
    for (uint32_t i = 0; i < g_dispatchRecords.size(); ++i)
    {
        const DispatchRecord &r = g_dispatchRecords[(g_dispatchRecordNext + i) % g_dispatchRecords.size()];
        if (r.what == 0)
            continue;
        out << ' ' << r.what << r.thread << ":pc=" << std::hex << r.pc << ",sp=" << r.sp << ",ra=" << r.ra << std::dec;
    }
    return out.str();
}

std::string PS2Runtime::dispatchTrace() const
{
    return formatDispatchHistory();
}

uint32_t PS2Runtime::kernelMalloc(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
        return 0u;
    const uint32_t align = std::max<uint32_t>(16u, alignment);
    const uint32_t need = (size + 15u) & ~15u;
    {
        std::lock_guard<std::mutex> lock(m_kernelArenaMutex);
        uint32_t cursor = kKernelArenaBase;
        for (const auto &[addr, blockSize] : m_kernelArenaBlocks)
        {
            const uint32_t start = (cursor + align - 1u) & ~(align - 1u);
            if (start + need <= addr)
                break;
            cursor = addr + blockSize;
        }
        const uint32_t start = (cursor + align - 1u) & ~(align - 1u);
        bool fits = start + need <= kKernelArenaLimit;
        for (const auto &[addr, blockSize] : m_kernelArenaBlocks)
            if (addr < start + need && start < addr + blockSize)
                fits = false;
        if (fits)
        {
            m_kernelArenaBlocks.emplace(start, need);
            return start;
        }
        static uint32_t overflows = 0u;
        if (++overflows <= 20u)
            std::cerr << "[kernel] kernelMalloc area full (" << m_kernelArenaBlocks.size() << " blocks) - " << size
                      << " bytes from the guest heap instead (it may share RAM with the game's own heap)" << std::endl;
    }
    return guestMalloc(size, alignment);
}

void PS2Runtime::guestFree(uint32_t guestAddr)
{
    if (guestAddr == 0u)
    {
        return;
    }
    if (guestAddr >= kKernelArenaBase && guestAddr < kKernelArenaLimit)
    {
        std::lock_guard<std::mutex> lock(m_kernelArenaMutex);
        m_kernelArenaBlocks.erase(guestAddr);
        return;
    }

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    freeGuestBlockLocked(guestAddr);
}

uint32_t PS2Runtime::guestHeapBase() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapBase : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapEnd() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapEnd : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapLimit() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapLimit : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::reserveAsyncCallbackStack(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
    uint32_t top = m_asyncCallbackStackTop;
    if (top > PS2_RAM_SIZE)
    {
        top = PS2_RAM_SIZE;
    }
    top &= ~(kGuestHeapDefaultAlignment - 1u);

    if (top <= allocSize)
    {
        return 0u;
    }

    uint32_t base = top - allocSize;
    base &= ~(normalizedAlignment - 1u);
    if (base < m_asyncCallbackStackFloor || base >= top)
    {
        return 0u;
    }

    m_asyncCallbackStackTop = base;
    return top - 0x10u;
}

uint8_t PS2Runtime::Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read8(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint16_t PS2Runtime::Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read16(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

namespace
{
    bool isGsCsrAddress(uint32_t vaddr) { return (vaddr & 0x1FFFFFF8u) == 0x12001000u; }
}

// A CSR read at a site registered as using only CSR.FIELD (the chain's exemption): no fence. Under
// VENDETTA_CHAIN_VERIFY the value is compared with a fenced read (FIELD must match; SIGNAL / FINISH may lag).
template <typename T, typename Read>
T PS2Runtime::loadCsrExempt(Read read)
{
    ps2_host_profile::count(ps2_host_profile::ChainCsrExempt);
    PS2Memory::skipNextCsrFence();
    const T value = read();
    if (m_chain.verifyExemptReads())
    {
        const T fenced = read();
        if (((value ^ fenced) & static_cast<T>(0x2000u)) != 0u)
            ps2_host_profile::count(ps2_host_profile::ChainVerifyMismatch);
        else if (((value ^ fenced) & static_cast<T>(0x3u)) != 0u)
            ps2_host_profile::count(ps2_host_profile::ChainCsrExemptDiff);
    }
    return value;
}

uint32_t PS2Runtime::Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        if (isGsCsrAddress(vaddr) && m_chain.async() && m_chain.csrReadExempt(ctx->pc))
            return loadCsrExempt<uint32_t>([&]() { return m_memory.read32(vaddr); });
        return m_memory.read32(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint64_t PS2Runtime::Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        if (isGsCsrAddress(vaddr) && m_chain.async() && m_chain.csrReadExempt(ctx->pc))
            return loadCsrExempt<uint64_t>([&]() { return m_memory.read64(vaddr); });
        return m_memory.read64(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

__m128i PS2Runtime::Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read128(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return _mm_setzero_si128();
    }
}

void PS2Runtime::Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 1u, value, 0u, "WRITE8", ctx);
    try
    {
        m_memory.write8(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 2u, value, 0u, "WRITE16", ctx);
    try
    {
        m_memory.write16(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 4u, value, 0u, "WRITE32", ctx);
    try
    {
        m_memory.write32(vaddr, value);
        drainCompletedDmacHandlers(rdram);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 8u, value, 0u, "WRITE64", ctx);
    try
    {
        m_memory.write64(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
{
    alignas(16) uint64_t _parts[2];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(_parts), value);
    ps2TraceGuestWrite(rdram, vaddr, 16u, _parts[0], _parts[1], "WRITE128", ctx);
    try
    {
        m_memory.write128(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::kickGifDmaChainFromMMIO(uint8_t *rdram,
                                         R5900Context *ctx,
                                         uint32_t dPcrValue,
                                         uint32_t dStatValue,
                                         uint32_t tadr,
                                         uint32_t chcr)
{
    constexpr uint32_t D_PCR = 0x1000E020u;
    constexpr uint32_t D_STAT = 0x1000E010u;
    constexpr uint32_t GIF_TADR = 0x1000A030u;
    constexpr uint32_t GIF_CHCR = 0x1000A000u;

    ps2TraceGuestWrite(rdram, D_PCR, 4u, dPcrValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_PCR, dPcrValue);
    ps2TraceGuestWrite(rdram, D_STAT, 4u, dStatValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_STAT, dStatValue);
    ps2TraceGuestWrite(rdram, GIF_TADR, 4u, tadr, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(GIF_TADR, tadr);
    ps2TraceGuestWrite(rdram, GIF_CHCR, 4u, chcr, 0u, "WRITE32", ctx);
    if (m_memory.tryProcessNativeGifImageUploadChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    if (m_memory.tryProcessNativeGifPackedChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    m_memory.writeIORegister(GIF_CHCR, chcr);
    m_memory.processPendingTransfers();
    drainCompletedDmacHandlers(rdram);
}

void PS2Runtime::requestStop()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
    if (m_eeScheduler)
    {
        m_eeScheduler->requestStop();
    }
    m_chain.wakeWaiters();    // a game thread held by the chain throttle sees the stop
    m_gs.wakeRasterWaiters(); // ... or by the GS thread's flip throttle
}

R5900Context *PS2Runtime::currentEeContext()
{
    R5900Context *ctx = m_eeScheduler ? m_eeScheduler->currentContext() : nullptr;
    return ctx ? ctx : &m_cpuContext;
}

bool PS2Runtime::isStopRequested() const
{
    return m_stopRequested.load(std::memory_order_relaxed);
}

EeScheduler &PS2Runtime::eeScheduler()
{
    return *m_eeScheduler;
}

const EeScheduler &PS2Runtime::eeScheduler() const
{
    return *m_eeScheduler;
}

void PS2Runtime::postEeEvent(EeEvent event)
{
    m_eeScheduler->postEvent(event);
}

bool PS2Runtime::eeCheckpointDue(uint32_t cycles) noexcept
{
    return m_eeScheduler->checkpointDue(cycles);
}

[[noreturn]] void PS2Runtime::eeWaitVSyncTicks(uint32_t ticks, uint32_t resumePc)
{
    const uint64_t currentTick = m_eeScheduler->currentVSyncTick();
    const uint64_t waitTicks = std::max<uint64_t>(1u, ticks);
    m_eeScheduler->waitVSync(currentTick + waitTicks - 1u,
                             0,
                             [resumePc](R5900Context &context)
                             {
                                 context.pc = resumePc;
                             });
}

void PS2Runtime::addEeExitHandler(int threadId, uint32_t function, uint32_t argument)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers[threadId].push_back({function, argument});
}

std::vector<PS2Runtime::EeExitHandlerRegistration> PS2Runtime::takeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    auto it = m_eeExitHandlers.find(threadId);
    if (it == m_eeExitHandlers.end())
    {
        return {};
    }
    auto handlers = std::move(it->second);
    m_eeExitHandlers.erase(it);
    return handlers;
}

void PS2Runtime::removeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers.erase(threadId);
}

bool PS2Runtime::findEeSyscallOverride(uint32_t syscallNumber, uint32_t &handler) const
{
    std::lock_guard lock(m_eeKernelStateMutex);
    const auto it = m_eeSyscallOverrides.find(syscallNumber);
    if (it == m_eeSyscallOverrides.end())
    {
        return false;
    }
    handler = it->second;
    return true;
}

void PS2Runtime::setEeSyscallOverride(uint8_t *rdram, uint32_t syscallNumber, uint32_t handler)
{
    constexpr uint32_t kTableBase = 0x80011F80u & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
    const int64_t address = static_cast<int64_t>(kTableBase) + offset;

    std::lock_guard lock(m_eeKernelStateMutex);
    if (handler == 0u)
    {
        m_eeSyscallOverrides.erase(syscallNumber);
    }
    else
    {
        m_eeSyscallOverrides[syscallNumber] = handler;
    }
    if (!rdram || address < 0 || address + 4 > kMirrorLimit)
    {
        return;
    }
    const uint32_t guestAddress = static_cast<uint32_t>(address);
    std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
    if (handler == 0u)
    {
        m_eeSyscallMirrorAddresses.erase(guestAddress);
    }
    else
    {
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::initializeEeKernelState(uint8_t *rdram)
{
    if (!rdram)
    {
        return;
    }
    constexpr uint32_t kTableGuestBase = 0x80011F80u;
    constexpr uint32_t kTableBase = kTableGuestBase & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    constexpr uint32_t kProbeBase = 0x000002F0u;

    std::lock_guard lock(m_eeKernelStateMutex);
    for (const uint32_t address : m_eeSyscallMirrorAddresses)
    {
        const uint32_t zero = 0u;
        std::memcpy(rdram + address, &zero, sizeof(zero));
    }
    m_eeSyscallMirrorAddresses.clear();
    const uint32_t high = kTableGuestBase >> 16;
    const uint32_t low = kTableGuestBase & 0xFFFFu;
    std::memcpy(rdram + kProbeBase, &high, sizeof(high));
    std::memcpy(rdram + kProbeBase + 8u, &low, sizeof(low));
    m_eeSyscallMirrorAddresses.insert(kProbeBase);
    m_eeSyscallMirrorAddresses.insert(kProbeBase + 8u);

    for (const auto &[syscallNumber, handler] : m_eeSyscallOverrides)
    {
        const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
        const int64_t address = static_cast<int64_t>(kTableBase) + offset;
        if (address < 0 || address + 4 > kMirrorLimit)
        {
            continue;
        }
        const uint32_t guestAddress = static_cast<uint32_t>(address);
        std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::HandleIntegerOverflow(R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_INTEGER_OVERFLOW);
}

void ps2HostPreferPerformance();      // Kernel/HostPower.cpp
void ps2HostPreferPerformanceCores(); // Kernel/HostPower.cpp

void PS2Runtime::run()
{
    ps2HostPreferPerformance();
    m_stopRequested.store(false, std::memory_order_relaxed);
    ps2_stubs::resetSifState();
    resetIop();
    ps2_stubs::resetAudioStubState();
    ps2_stubs::resetMpegStubState();
    initializeEeKernelState(m_memory.getRDRAM());
    m_cpuContext.r[4] = _mm_setzero_si128();
    m_cpuContext.r[5] = _mm_setzero_si128();
    m_cpuContext.r[29] = _mm_set_epi64x(0, static_cast<int64_t>(PS2_RAM_SIZE - 0x10u));
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)), std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[29], 0)), std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[28], 0)), std::memory_order_relaxed);

    RUNTIME_LOG("Starting execution at address 0x" << std::hex << m_cpuContext.pc << std::dec);

    // A blank image to use as a framebuffer
    Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLANK);
    Texture2D frameTex = LoadTextureFromImage(blank);
    UnloadImage(blank);
    // The picture is scaled to the window by a non-whole factor (a 224-line field on a 1440-line screen is ×6.4), so
    // blocky nearest-pixel scaling makes uneven blocks: smoothed by default (VENDETTA_FILTER: fsr on the graphics card,
    // bilinear, or nearest = the blocky look).
    {
        const GSPresentFilter filter = m_gs.presentFilter();
        s_frameTextureFilter = filter == GSPresentFilter::Nearest ? TEXTURE_FILTER_POINT : TEXTURE_FILTER_BILINEAR;
        SetTextureFilter(frameTex, s_frameTextureFilter);
        std::cerr << "[present] filter="
                  << (filter == GSPresentFilter::Fsr ? "fsr" : filter == GSPresentFilter::Nearest ? "nearest" : "bilinear")
                  << " renderer=" << gsRendererModeName(m_gs.rendererMode()) << std::endl;
    }

    std::atomic<bool> gameThreadFinished{false};

#if defined(PS2X_VULKAN_PRESENT)
    // shown through Vulkan (vulkanPresentWanted): the window and its presenter were made at start; raylib's hidden
    // window doesn't wait for the screen, so the loop waits once
    VulkanPresenter *presenter = s_presenter.get();
    GLFWwindow *presenterWindow = s_vulkanWindow;
#endif

    m_chain.start(); // the VIF1 -> VU1 -> GS chain worker (VENDETTA_CHAIN_ASYNC=1)
    std::thread gameThread([&]()
                           {
        ThreadNaming::SetCurrentThreadName("GameThread");
        ps2_host_profile::recordThisThread();
        ps2HostPreferPerformanceCores();
        ChainWorker::markProducerThread();
        try
        {
            m_eeScheduler->reset(m_memory.getRDRAM(), m_cpuContext);
            m_eeScheduler->run();
            uint32_t pc = m_debugPc.load(std::memory_order_relaxed);
            RUNTIME_LOG("Game thread returned. PC=0x" << std::hex << pc
                      << " RA=0x" << static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)) << std::dec << std::endl);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during program execution: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Error during program execution: unknown exception" << std::endl;
        }
        gameThreadFinished.store(true, std::memory_order_release); });

    uint64_t tick = 0;
    while (!isStopRequested() && !gameThreadFinished.load(std::memory_order_acquire))
    {
        PS2_IF_AGRESSIVE_LOGS({
            tick++;
            if ((tick % 120) == 0)
            {
                uint64_t curDma = m_memory.dmaStartCount();
                uint64_t curGif = m_memory.gifCopyCount();
                uint64_t curGs = m_memory.gsWriteCount();
                uint64_t curVif = m_memory.vifWriteCount();
                const GSRegisters &gs = m_memory.gs();
                const uint32_t dbgPc = m_debugPc.load(std::memory_order_relaxed);
                const uint32_t dbgRa = m_debugRa.load(std::memory_order_relaxed);
                const uint32_t dbgSp = m_debugSp.load(std::memory_order_relaxed);
                const uint32_t dbgGp = m_debugGp.load(std::memory_order_relaxed);
                const auto eeSnapshot = m_eeScheduler->snapshot();

                RUNTIME_LOG("[run:tick] tick=" << tick
                                               << " pc=0x" << std::hex << dbgPc
                                               << " ra=0x" << dbgRa
                                               << " sp=0x" << dbgSp
                                               << " gp=0x" << dbgGp
                                               << " dispfb1=0x" << gs.dispfb1
                                               << " display1=0x" << gs.display1
                                               << std::dec
                                               << " activeThreads=" << eeSnapshot.threads.size()
                                               << " dma=" << curDma
                                               << " gif=" << curGif
                                               << " gsw=" << curGs
                                               << " vif=" << curVif
                                               << std::endl);

            }
        });
        uint32_t presentWidth = FB_WIDTH;
        uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
        // The window shows each picture once, as soon as the game has finished it: it waits for a new one (up to
        // 20 ms — a game frame comes every 16.7 ms — so input and the window are still served at least 50 times a
        // second when the picture stands still, a menu waiting for a press). With its own 60-a-second timer, which
        // drifts against the game's 59.94 and jitters by a millisecond, a picture was now and then shown twice and the
        // next one never — a slight stutter although every frame was on time. VENDETTA_WINDOW_PACING=timer = the old way.
        bool freshPicture = UploadFrame(frameTex, this, presentWidth, presentHeight);
        if (!windowTimerPacing())
        {
            const auto waitStart = std::chrono::steady_clock::now();
            while (!freshPicture && !isStopRequested() && !gameThreadFinished.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() - waitStart < std::chrono::milliseconds(20))
            {
                std::this_thread::sleep_for(std::chrono::microseconds(250));
                freshPicture = UploadFrame(frameTex, this, presentWidth, presentHeight);
            }
        }
        ps2_host_profile::count(freshPicture ? ps2_host_profile::WindowFrames : ps2_host_profile::WindowRepeats);

        BeginDrawing();
        ClearBackground(BLACK);
        const float srcWidth = static_cast<float>(std::max<uint32_t>(1u, presentWidth));
        const float srcHeight = static_cast<float>(std::max<uint32_t>(1u, presentHeight));
        // the framebuffer's size: in fullscreen raylib's screen size can stay at the window's size from before (Windows
        // clamps a framed 2560 x 1440 window to the screen minus its title bar), which left a bar under the picture
        const float screenWidth = static_cast<float>(GetRenderWidth());
        const float screenHeight = static_cast<float>(GetRenderHeight());
        // A PS2 picture fills a 4:3 TV whatever the frame buffer's width (512-wide menus and 640-wide movies alike);
        // scaling by pixels made 512 × 448 screens look squeezed. VENDETTA_ASPECT=16:9 stretches to widescreen; the
        // setting shape = "stretch" (VENDETTA_ASPECT=fill) fills the window whatever its shape.
        static const float displayAspect = []
        {
            const char *aspect = std::getenv("VENDETTA_ASPECT");
            if (aspect && std::string(aspect) == "16:9")
                return 16.0f / 9.0f;
            return ps2_host_settings::get().stretch ? 0.0f : 4.0f / 3.0f;
        }();
        // widescreen: a frame the game drew wide (squeezed into its buffer) is shown at exactly the target shape, filling
        // a screen of that shape (the TV picture's small covered-share correction would leave thin bars at the sides)
        const float wideAspect = ps2_widescreen::wideNow() ? ps2_widescreen::targetAspect() : 0.0f;
        const float aspect = wideAspect > 0.0f        ? wideAspect
                             : displayAspect > 0.0f ? displayAspect * s_presentAspectScale
                                                    : screenWidth / std::max(1.0f, screenHeight);
        // whole pixels: an FSR picture is drawn at exactly this size, one texel per pixel
        const float dstHeight = std::round(std::min(screenHeight, screenWidth / aspect));
        const float dstWidth = std::round(std::min(screenWidth, dstHeight * aspect));
#if defined(PS2X_VULKAN_PRESENT)
        if (!presenterWindow) // the Vulkan window sets FSR's size itself
#endif
        if (GSGpuRenderer *gpu = m_gs.gpuRendererForSettings())
            gpu->setOutputSize(static_cast<uint32_t>(dstWidth), static_cast<uint32_t>(dstHeight));
        const Rectangle srcRect{0.0f, 0.0f, srcWidth, srcHeight};
        const Rectangle dstRect{
            std::floor((screenWidth - dstWidth) * 0.5f),
            std::floor((screenHeight - dstHeight) * 0.5f),
            dstWidth,
            dstHeight};
        {
            // once per change of layout: where the picture goes (and what raylib says the window is)
            static float s_loggedLayout[6] = {};
            const float layout[6] = {screenWidth, screenHeight, dstRect.x, dstRect.y, dstWidth, dstHeight};
            if (!std::equal(std::begin(layout), std::end(layout), std::begin(s_loggedLayout)))
            {
                std::copy(std::begin(layout), std::end(layout), std::begin(s_loggedLayout));
                std::cerr << "[present] picture " << dstWidth << "x" << dstHeight << " at " << dstRect.x << "," << dstRect.y
                          << " in " << screenWidth << "x" << screenHeight << " (raylib's screen " << GetScreenWidth() << "x"
                          << GetScreenHeight() << "); source " << srcWidth << "x" << srcHeight << std::endl;
            }
        }
        DrawTexturePro(frameTex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        if (m_debugUiInitialized && m_debugUiDrawCallback)
        {
            m_debugUiDrawCallback(*this, m_debugUiUserData);
        }
        EndDrawing();

#if defined(PS2X_VULKAN_PRESENT)
        if (presenterWindow)
            host_input::poll(); // the controllers' state for the pads (raylib's frame polled the events)
        // minimised (Alt-Tab from fullscreen): nothing to show (the swapchain would wait for a size on this thread)
        if (presenterWindow && s_sharedFrame && freshPicture && !glfwGetWindowAttrib(presenterWindow, GLFW_ICONIFIED))
        {
            GSGpuRenderer *gpu = m_gs.gpuRendererForSettings();
            if (presenter && gpu)
            {
                int fbWidth = 0, fbHeight = 0;
                glfwGetFramebufferSize(presenterWindow, &fbWidth, &fbHeight);
                const float vkAspect = aspect > 0.0f ? aspect : 4.0f / 3.0f;
                const float vkHeight = std::round(std::min(static_cast<float>(fbHeight), fbWidth / vkAspect));
                const float vkWidth = std::round(std::min(static_cast<float>(fbWidth), vkHeight * vkAspect));
                gpu->setOutputSize(static_cast<uint32_t>(vkWidth), static_cast<uint32_t>(vkHeight)); // FSR at that size
                presenter->present(*s_sharedFrame, *gpu, static_cast<int>(std::floor((fbWidth - vkWidth) * 0.5f)),
                                   static_cast<int>(std::floor((fbHeight - vkHeight) * 0.5f)), static_cast<int>(vkWidth),
                                   static_cast<int>(vkHeight), m_gs.presentFilter() == GSPresentFilter::Nearest);
            }
        }
        if (presenterWindow && glfwWindowShouldClose(presenterWindow))
        {
            requestStop();
            break;
        }
#endif

        if (WindowShouldClose())
        {
            RUNTIME_LOG("[run] window close requested, breaking out of loop");
            requestStop();
            break;
        }
    }

    requestStop();
    if (gameThread.joinable())
    {
        gameThread.join();
    }
    ps2_pad_rumble::stopAll(); // the game is done with the pads: no controller left vibrating
    m_chain.shutdown();          // the chain worker first (it feeds the GS)...
    m_gs.shutdownRasterThread(); // ... then the GS thread
    m_gs.shutdownGpu();          // ... then the graphics card renderer

    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }
#if defined(PS2X_VULKAN_PRESENT)
    s_presenter.reset();
    if (presenterWindow)
        glfwDestroyWindow(presenterWindow);
    s_vulkanWindow = nullptr;
#endif
    UnloadTexture(frameTex);
    CloseWindow();

    RUNTIME_LOG("[run] exiting loop");
}
