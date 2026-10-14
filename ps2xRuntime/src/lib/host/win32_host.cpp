// Native Win32 host (window, Direct3D 11, XInput, XAudio2). Ported from WWE Crush Hour PC (KaikoClanworth1,
// GPL-3.0, https://github.com/KaikoClanworth1/wwe-crush-hour-pc) for the HCTP native-renderer build.
#include "host/win32_host.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#include <tlhelp32.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_3.h>
#include <xaudio2.h>
#include <Xinput.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

extern std::atomic<double> g_hitchLastFlip; // ps2_runtime.cpp (PS2X_HITCH_LOG): host time of the last flip

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "xaudio2.lib")
#pragma comment(lib, "xinput.lib")
#pragma comment(lib, "winmm.lib")

using Microsoft::WRL::ComPtr;

uint64_t ps2Vif1KickCount(); // ps2_memory.cpp: game frames (VIF1 frame kicks) so far

// Crash report: an unhandled exception prints the faulting thread's stack as module+offset (x64 unwind
// data, no PDB needed) to stdout; tools/crashsym.py resolves the offsets with ps2EntryRunner.map.
static LONG WINAPI ps2xCrashFilter(EXCEPTION_POINTERS *ep)
{
    CONTEXT c = *ep->ContextRecord;
    std::printf("[crash] code=0x%08lx addr=%p thread=%lu\n", ep->ExceptionRecord->ExceptionCode,
                ep->ExceptionRecord->ExceptionAddress, GetCurrentThreadId());
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2)
        std::printf("[crash] %s address 0x%llx\n", ep->ExceptionRecord->ExceptionInformation[0] ? "write" : "read",
                    static_cast<unsigned long long>(ep->ExceptionRecord->ExceptionInformation[1]));
    for (int i = 0; i < 48 && c.Rip; ++i)
    {
        HMODULE mod = nullptr;
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(c.Rip), &mod))
            GetModuleFileNameA(mod, name, MAX_PATH);
        const char *base = std::strrchr(name, '\\');
        std::printf("[crash]   #%02d %s+0x%llx\n", i, base ? base + 1 : name,
                    static_cast<unsigned long long>(c.Rip - reinterpret_cast<DWORD64>(mod)));
        DWORD64 imageBase = 0;
        PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(c.Rip, &imageBase, nullptr);
        if (!fn)
        {
            c.Rip = *reinterpret_cast<DWORD64 *>(c.Rsp); // leaf function
            c.Rsp += 8;
            continue;
        }
        void *handlerData = nullptr;
        DWORD64 establisher = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, c.Rip, fn, &c, &handlerData, &establisher, nullptr);
    }
    std::fflush(stdout);
    return EXCEPTION_CONTINUE_SEARCH;
}

// HCTP: Crush Hour's mod menu (mod_menu.cpp) is not part of this port; its hooks stay as no-ops.
namespace modmenu
{
    inline void filterInput(ps2host::PadState *, int, bool) {}
    inline bool panel(uint32_t, const uint32_t *&, uint32_t &, uint32_t &, bool &) { return false; }
}

namespace ps2host
{
    constexpr int kAutoAnchorHits = 32;
    // Anchored auto-input: the dispatcher stamps g_autoAnchorAt when the guest first calls g_autoAnchorPc.
    uint32_t g_autoAnchorPc = 0;
    int g_autoAnchorCount = 1;          // PS2X_AUTOINPUT_ANCHOR=<hex pc>[:<n>] = default call index for "@t"
    std::atomic<double> g_autoAnchorAt[kAutoAnchorHits]; // time of the 1st..32nd call of the anchor pc
    uint32_t g_autoWatchPc[8] = {};     // PS2X_AUTOINPUT_WATCH=<hex pc>,... : log call times (finding anchors)
    int g_autoWatchN = 0;
    namespace
    {
        HostConfig g_config;
        bool g_background = false;      // PS2X_BACKGROUND: hidden, never activated (automated tests)
        std::string g_shotDir;          // PS2X_SCREENSHOT_DIR: periodic back-buffer dumps
        double g_shotInterval = 5.0;    // PS2X_SCREENSHOT_SEC
        double g_nextShot = 0.0;
        int g_shotIndex = 0;
        struct AutoPress { double t, until; uint16_t button; int anchorHit; int pad; }; // anchorHit 0 = absolute time
        std::vector<AutoPress> g_autoPress; // PS2X_AUTOINPUT="12:START,@0.5:RIGHT,@3+1.0:CROSS" ('@[n+]t' = t s after anchor call n)
        HWND g_hwnd = nullptr;
        std::atomic<bool> g_quit{false};
        bool g_ready = false;

        ComPtr<ID3D11Device> g_device;
        ComPtr<ID3D11DeviceContext> g_ctx;
        ComPtr<IDXGISwapChain1> g_swap;
        HANDLE g_frameWaitable = nullptr;
        ComPtr<ID3D11RenderTargetView> g_backRtv;
        ComPtr<ID3D11VertexShader> g_blitVs;
        ComPtr<ID3D11PixelShader> g_blitPs;
        ComPtr<ID3D11SamplerState> g_linearSampler;
        ComPtr<ID3D11Buffer> g_blitCb;
        ComPtr<ID3D11Texture2D> g_uploadTex;
        ComPtr<ID3D11ShaderResourceView> g_uploadSrv;
        uint32_t g_uploadW = 0, g_uploadH = 0;
        std::recursive_mutex g_d3dMutex;
        std::atomic<uint64_t> g_d3dEpoch{1}; // bumped whenever the host changes immediate-context state

        ComPtr<IXAudio2> g_xaudio;
        IXAudio2MasteringVoice *g_master = nullptr;

        std::mutex g_inputMutex;
        PadState g_pads[kMaxPads];
        bool g_keys[256]{};

        LARGE_INTEGER g_qpcFreq{};
        LARGE_INTEGER g_qpcStart{};
        double g_nextFrameTime = 0.0;
        double g_lastFrameMs = 0.0;
        double g_lastPresentTime = 0.0;

        // FPS counter: on-screen box (F1 toggles, ShowFPS in [Display]) and the window title.
        bool g_showFps = true;
        std::wstring g_baseTitle;
        std::atomic<int> g_titleFps{-1}, g_titleGameFps{-1};
        int g_shownTitleFps = -2, g_shownTitleGameFps = -2;
        double g_fpsWindowStart = 0.0;
        uint32_t g_fpsPresents = 0;
        uint64_t g_fpsKickStart = 0;
        double g_presentFps = 0.0, g_gameFps = 0.0;
        ComPtr<ID3D11Texture2D> g_fpsTex;
        ComPtr<ID3D11ShaderResourceView> g_fpsSrv;
        ComPtr<ID3D11SamplerState> g_pointSampler;

        const char *kBlitShader = R"(
cbuffer Blit : register(b0) { float4 uvRect; };
Texture2D tex : register(t0);
SamplerState samp : register(s0);
struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
VsOut vs_main(uint id : SV_VertexID)
{
    VsOut o;
    float2 p = float2((id << 1) & 2, id & 2);
    o.pos = float4(p * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = uvRect.xy + p * uvRect.zw;
    return o;
}
float4 ps_main(VsOut i) : SV_Target
{
    float4 c = tex.Sample(samp, i.uv);
    return float4(c.rgb, 1.0);
}
)";

        std::wstring widen(const char *s)
        {
            if (!s)
                return L"";
            int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
            std::wstring w(static_cast<size_t>(std::max(0, n - 1)), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
            return w;
        }

        LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
        {
            switch (msg)
            {
            case WM_CLOSE:
                g_quit.store(true);
                return 0;
            case WM_DESTROY:
                g_quit.store(true);
                PostQuitMessage(0);
                return 0;
            case WM_KEYDOWN:
            case WM_SYSKEYDOWN:
                if (wp < 256)
                {
                    std::lock_guard lock(g_inputMutex);
                    g_keys[wp] = true;
                }
                if (wp == VK_F1 && !(lp & (1 << 30)))
                    g_showFps = !g_showFps;
                if (wp == VK_RETURN && (lp & (1 << 29)))
                {
                    // Alt+Enter toggles borderless fullscreen.
                    g_config.fullscreen = !g_config.fullscreen;
                    LONG style = g_config.fullscreen ? (WS_POPUP | WS_VISIBLE) : (WS_OVERLAPPEDWINDOW | WS_VISIBLE);
                    SetWindowLongPtrW(hwnd, GWL_STYLE, style);
                    if (g_config.fullscreen)
                    {
                        MONITORINFO mi{sizeof(mi)};
                        GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mi);
                        SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                                     mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top,
                                     SWP_FRAMECHANGED);
                    }
                    else
                    {
                        RECT r{0, 0, (LONG)g_config.width, (LONG)g_config.height};
                        AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
                        SetWindowPos(hwnd, HWND_TOP, 64, 64, r.right - r.left, r.bottom - r.top, SWP_FRAMECHANGED);
                    }
                    return 0;
                }
                break;
            case WM_KEYUP:
            case WM_SYSKEYUP:
                if (wp < 256)
                {
                    std::lock_guard lock(g_inputMutex);
                    g_keys[wp] = false;
                }
                break;
            case WM_KILLFOCUS:
            {
                std::lock_guard lock(g_inputMutex);
                std::memset(g_keys, 0, sizeof(g_keys));
                break;
            }
            default:
                break;
            }
            return DefWindowProcW(hwnd, msg, wp, lp);
        }

        bool compileShader(const char *src, const char *entry, const char *target, ComPtr<ID3DBlob> &out)
        {
            ComPtr<ID3DBlob> err;
            HRESULT hr = D3DCompile(src, std::strlen(src), nullptr, nullptr, nullptr, entry, target,
                                    D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &out, &err);
            if (FAILED(hr))
            {
                std::fprintf(stderr, "[host] shader compile failed (%s): %s\n", entry,
                             err ? static_cast<const char *>(err->GetBufferPointer()) : "?");
                return false;
            }
            return true;
        }

        void createBackBufferView()
        {
            g_backRtv.Reset();
            ComPtr<ID3D11Texture2D> back;
            g_swap->GetBuffer(0, IID_PPV_ARGS(&back));
            g_device->CreateRenderTargetView(back.Get(), nullptr, &g_backRtv);
        }

        void resizeIfNeeded()
        {
            RECT rc{};
            GetClientRect(g_hwnd, &rc);
            UINT w = std::max<LONG>(1, rc.right - rc.left);
            UINT h = std::max<LONG>(1, rc.bottom - rc.top);
            DXGI_SWAP_CHAIN_DESC1 desc{};
            g_swap->GetDesc1(&desc);
            if (desc.Width == w && desc.Height == h)
                return;
            g_ctx->OMSetRenderTargets(0, nullptr, nullptr);
            g_backRtv.Reset();
            g_swap->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
            createBackBufferView();
        }

        double nowSeconds()
        {
            LARGE_INTEGER t;
            QueryPerformanceCounter(&t);
            return static_cast<double>(t.QuadPart - g_qpcStart.QuadPart) / static_cast<double>(g_qpcFreq.QuadPart);
        }

        void frameLimit()
        {
            if (g_config.targetFps == 0)
                return;
            const double period = 1.0 / static_cast<double>(g_config.targetFps);
            double now = nowSeconds();
            if (g_nextFrameTime == 0.0 || now - g_nextFrameTime > 0.25)
                g_nextFrameTime = now;
            g_nextFrameTime += period;
            while (true)
            {
                now = nowSeconds();
                double remain = g_nextFrameTime - now;
                if (remain <= 0.0)
                    break;
                if (remain > 0.002)
                    Sleep(static_cast<DWORD>((remain - 0.0015) * 1000.0));
                else
                    std::this_thread::yield();
            }
            g_lastFrameMs = (now - g_lastPresentTime) * 1000.0;
            g_lastPresentTime = now;
        }

        double nowSeconds();

        void saveScreenshot()
        {
            ComPtr<ID3D11Texture2D> back;
            if (FAILED(g_swap->GetBuffer(0, IID_PPV_ARGS(&back))))
                return;
            D3D11_TEXTURE2D_DESC td{};
            back->GetDesc(&td);
            td.Usage = D3D11_USAGE_STAGING;
            td.BindFlags = 0;
            td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            td.MiscFlags = 0;
            ComPtr<ID3D11Texture2D> staging;
            if (FAILED(g_device->CreateTexture2D(&td, nullptr, &staging)))
                return;
            g_ctx->CopyResource(staging.Get(), back.Get());
            D3D11_MAPPED_SUBRESOURCE m{};
            if (FAILED(g_ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m)))
                return;
            // Copy the pixels out and write the file on a worker: a 6 MB write to a busy disk under the D3D
            // lock stalled the GS thread for seconds (test runs only).
            const uint32_t w = td.Width, h = td.Height;
            std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4u);
            for (uint32_t y = 0; y < h; ++y)
                std::memcpy(pixels.data() + static_cast<size_t>(y) * w * 4u,
                            static_cast<const uint8_t *>(m.pData) + static_cast<size_t>(y) * m.RowPitch, w * 4u);
            g_ctx->Unmap(staging.Get(), 0);
            char path[512];
            std::snprintf(path, sizeof(path), "%s/shot_%04d.bmp", g_shotDir.c_str(), g_shotIndex++);
            std::thread([pixels = std::move(pixels), w, h, file = std::string(path)]()
            {
                FILE *f = std::fopen(file.c_str(), "wb");
                if (!f)
                    return;
                const uint32_t row = w * 3u, pad = (4u - row % 4u) % 4u;
                const uint32_t dataSize = (row + pad) * h;
                BITMAPFILEHEADER fh{};
                fh.bfType = 0x4D42;
                fh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
                fh.bfSize = fh.bfOffBits + dataSize;
                BITMAPINFOHEADER ih{};
                ih.biSize = sizeof(ih);
                ih.biWidth = static_cast<LONG>(w);
                ih.biHeight = static_cast<LONG>(h);
                ih.biPlanes = 1;
                ih.biBitCount = 24;
                std::fwrite(&fh, sizeof(fh), 1, f);
                std::fwrite(&ih, sizeof(ih), 1, f);
                std::vector<uint8_t> line(row + pad, 0);
                for (int y = static_cast<int>(h) - 1; y >= 0; --y)
                {
                    const uint8_t *src = pixels.data() + static_cast<size_t>(y) * w * 4u;
                    for (uint32_t x = 0; x < w; ++x)
                    {
                        line[x * 3 + 0] = src[x * 4 + 2];
                        line[x * 3 + 1] = src[x * 4 + 1];
                        line[x * 3 + 2] = src[x * 4 + 0];
                    }
                    std::fwrite(line.data(), 1, line.size(), f);
                }
                std::fclose(f);
            }).detach();
        }

        // 5x7 glyphs (one byte per row, low 5 bits, bit 4 = left column) for the FPS box.
        const uint8_t *glyph(char c)
        {
            static const uint8_t digits[10][7] = {
                {14, 17, 19, 21, 25, 17, 14}, {4, 12, 4, 4, 4, 4, 14},  {14, 17, 1, 2, 4, 8, 31},
                {31, 2, 4, 2, 1, 17, 14},     {2, 6, 10, 18, 31, 2, 2}, {31, 16, 30, 1, 1, 17, 14},
                {6, 8, 16, 30, 17, 17, 14},   {31, 1, 2, 4, 8, 8, 8},   {14, 17, 17, 14, 17, 17, 14},
                {14, 17, 17, 15, 1, 2, 12}};
            static const uint8_t F[7] = {31, 16, 16, 30, 16, 16, 16}, P[7] = {30, 17, 17, 30, 16, 16, 16},
                                 S[7] = {15, 16, 16, 14, 1, 1, 30}, G[7] = {14, 17, 16, 23, 17, 17, 15},
                                 A[7] = {14, 17, 17, 31, 17, 17, 17}, M[7] = {17, 27, 21, 21, 17, 17, 17},
                                 E[7] = {31, 16, 16, 30, 16, 16, 31}, blank[7] = {0, 0, 0, 0, 0, 0, 0};
            if (c >= '0' && c <= '9')
                return digits[c - '0'];
            switch (c)
            {
            case 'F': return F;
            case 'P': return P;
            case 'S': return S;
            case 'G': return G;
            case 'A': return A;
            case 'M': return M;
            case 'E': return E;
            default: return blank;
            }
        }

        // Counts presents and game frames (VIF1 frame kicks) and draws the FPS box over the back buffer.
        void updateAndDrawFps(float bh)
        {
            const double now = nowSeconds();
            ++g_fpsPresents;
            if (g_fpsWindowStart == 0.0)
            {
                g_fpsWindowStart = now;
                g_fpsKickStart = ps2Vif1KickCount();
            }
            else if (now - g_fpsWindowStart >= 0.5)
            {
                const double dt = now - g_fpsWindowStart;
                const uint64_t kicks = ps2Vif1KickCount();
                g_presentFps = g_fpsPresents / dt;
                g_gameFps = static_cast<double>(kicks - g_fpsKickStart) / dt;
                g_fpsPresents = 0;
                g_fpsWindowStart = now;
                g_fpsKickStart = kicks;
                g_titleFps.store(static_cast<int>(g_presentFps + 0.5));
                g_titleGameFps.store(static_cast<int>(g_gameFps + 0.5));
            }
            if (!g_showFps)
                return;

            char lines[2][16];
            std::snprintf(lines[0], sizeof(lines[0]), "FPS  %3d", static_cast<int>(g_presentFps + 0.5));
            std::snprintf(lines[1], sizeof(lines[1]), "GAME %3d", static_cast<int>(g_gameFps + 0.5));
            constexpr uint32_t kCols = 8, kW = kCols * 6 + 3, kH = 2 * 9 + 3;
            uint32_t px[kW * kH];
            for (uint32_t &p : px)
                p = 0xFF000000u; // opaque black box
            for (int l = 0; l < 2; ++l)
            {
                // RGBA in memory (little endian ABGR): green normally, orange when the game drops below 55.
                const uint32_t color = (l == 1 && g_gameFps < 55.0) ? 0xFF3090FFu : 0xFF60FF60u;
                for (uint32_t ci = 0; ci < kCols && lines[l][ci]; ++ci)
                {
                    const uint8_t *g = glyph(lines[l][ci]);
                    for (uint32_t y = 0; y < 7; ++y)
                        for (uint32_t x = 0; x < 5; ++x)
                            if (g[y] & (16u >> x))
                                px[(2 + l * 9 + y) * kW + 2 + ci * 6 + x] = color;
                }
            }
            if (!g_fpsTex)
            {
                D3D11_TEXTURE2D_DESC td{};
                td.Width = kW;
                td.Height = kH;
                td.MipLevels = 1;
                td.ArraySize = 1;
                td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_DEFAULT;
                td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                if (FAILED(g_device->CreateTexture2D(&td, nullptr, &g_fpsTex)))
                    return;
                g_device->CreateShaderResourceView(g_fpsTex.Get(), nullptr, &g_fpsSrv);
                D3D11_SAMPLER_DESC sd{};
                sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
                sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
                sd.MaxLOD = D3D11_FLOAT32_MAX;
                g_device->CreateSamplerState(&sd, &g_pointSampler);
            }
            g_ctx->UpdateSubresource(g_fpsTex.Get(), 0, nullptr, px, kW * 4u, 0);
            const float s = std::max(2.0f, std::floor(bh / 360.0f)); // integer scale: 3x at 1080p
            D3D11_VIEWPORT vp{8.0f, 8.0f, kW * s, kH * s, 0.0f, 1.0f};
            g_ctx->RSSetViewports(1, &vp);
            D3D11_MAPPED_SUBRESOURCE m{};
            if (SUCCEEDED(g_ctx->Map(g_blitCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
            {
                const float rect[4] = {0.0f, 0.0f, 1.0f, 1.0f};
                std::memcpy(m.pData, rect, sizeof(rect));
                g_ctx->Unmap(g_blitCb.Get(), 0);
            }
            ID3D11ShaderResourceView *srv = g_fpsSrv.Get();
            g_ctx->PSSetShaderResources(0, 1, &srv);
            ID3D11SamplerState *smp = g_pointSampler.Get();
            g_ctx->PSSetSamplers(0, 1, &smp);
            g_ctx->Draw(3, 0);
            ID3D11ShaderResourceView *nullSrv = nullptr;
            g_ctx->PSSetShaderResources(0, 1, &nullSrv);
        }

        // Widescreen=1: the GS backend shows frames without a 3D scene (movies, loading screens) 4:3 with bars.
        std::atomic<bool> g_presentWide{true};
        int g_internalScale = 0; // [Display] InternalScale (0 = from the output size)

        // The mod menu panel (mod_menu.cpp renders it): centred over the frame, premultiplied alpha.
        ComPtr<ID3D11Texture2D> g_modTex;
        ComPtr<ID3D11ShaderResourceView> g_modSrv;
        ComPtr<ID3D11PixelShader> g_alphaPs;
        ComPtr<ID3D11BlendState> g_premulBlend;
        uint32_t g_modW = 0, g_modH = 0;

        void drawModMenu(float bw, float bh)
        {
            const uint32_t *px = nullptr;
            uint32_t w = 0, h = 0;
            bool changed = false;
            if (!modmenu::panel(static_cast<uint32_t>(bh), px, w, h, changed))
                return;
            if (!g_alphaPs)
            {
                static const char *kAlphaPs = R"(
Texture2D tex : register(t0);
SamplerState samp : register(s0);
float4 ps_main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target { return tex.Sample(samp, uv); }
)";
                ComPtr<ID3DBlob> b;
                if (!compileShader(kAlphaPs, "ps_main", "ps_5_0", b))
                    return;
                g_device->CreatePixelShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &g_alphaPs);
                D3D11_BLEND_DESC bd{};
                bd.RenderTarget[0].BlendEnable = TRUE;
                bd.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
                bd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
                bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
                bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
                bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
                bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
                bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
                g_device->CreateBlendState(&bd, &g_premulBlend);
            }
            if (!g_modTex || w != g_modW || h != g_modH)
            {
                g_modTex.Reset();
                g_modSrv.Reset();
                D3D11_TEXTURE2D_DESC td{};
                td.Width = w;
                td.Height = h;
                td.MipLevels = 1;
                td.ArraySize = 1;
                td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_DEFAULT;
                td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                if (FAILED(g_device->CreateTexture2D(&td, nullptr, &g_modTex)))
                    return;
                g_device->CreateShaderResourceView(g_modTex.Get(), nullptr, &g_modSrv);
                g_modW = w;
                g_modH = h;
                changed = true;
            }
            if (changed)
                g_ctx->UpdateSubresource(g_modTex.Get(), 0, nullptr, px, w * 4u, 0);
            D3D11_VIEWPORT vp{std::floor((bw - static_cast<float>(w)) * 0.5f), std::floor((bh - static_cast<float>(h)) * 0.5f),
                              static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f};
            g_ctx->RSSetViewports(1, &vp);
            D3D11_MAPPED_SUBRESOURCE m{};
            if (SUCCEEDED(g_ctx->Map(g_blitCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
            {
                const float rect[4] = {0.0f, 0.0f, 1.0f, 1.0f};
                std::memcpy(m.pData, rect, sizeof(rect));
                g_ctx->Unmap(g_blitCb.Get(), 0);
            }
            g_ctx->OMSetBlendState(g_premulBlend.Get(), nullptr, 0xFFFFFFFFu);
            g_ctx->PSSetShader(g_alphaPs.Get(), nullptr, 0);
            ID3D11ShaderResourceView *srv = g_modSrv.Get();
            g_ctx->PSSetShaderResources(0, 1, &srv);
            ID3D11SamplerState *smp = g_pointSampler ? g_pointSampler.Get() : g_linearSampler.Get();
            g_ctx->PSSetSamplers(0, 1, &smp);
            g_ctx->Draw(3, 0);
            ID3D11ShaderResourceView *nullSrv = nullptr;
            g_ctx->PSSetShaderResources(0, 1, &nullSrv);
            g_ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
            g_ctx->PSSetShader(g_blitPs.Get(), nullptr, 0);
        }

        void drawBlit(ID3D11ShaderResourceView *srv, uint32_t srcW, uint32_t srcH, float u1, float v1)
        {
            g_d3dEpoch.fetch_add(1, std::memory_order_relaxed); // GS backend state cache must rebind
            resizeIfNeeded();
            DXGI_SWAP_CHAIN_DESC1 desc{};
            g_swap->GetDesc1(&desc);
            const float bw = static_cast<float>(desc.Width);
            const float bh = static_cast<float>(desc.Height);

            float vw = bw, vh = bh;
            if (g_config.keepAspect)
            {
                const float aspect = (g_config.widescreen && g_presentWide.load(std::memory_order_relaxed)) ? (16.0f / 9.0f) : (4.0f / 3.0f);
                if (bw / bh > aspect)
                    vw = bh * aspect;
                else
                    vh = bw / aspect;
            }
            (void)srcW;
            (void)srcH;

            const float clear[4] = {0, 0, 0, 1};
            g_ctx->ClearRenderTargetView(g_backRtv.Get(), clear);
            D3D11_VIEWPORT vp{(bw - vw) * 0.5f, (bh - vh) * 0.5f, vw, vh, 0.0f, 1.0f};
            g_ctx->RSSetViewports(1, &vp);
            ID3D11RenderTargetView *rtv = g_backRtv.Get();
            g_ctx->OMSetRenderTargets(1, &rtv, nullptr);
            g_ctx->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
            g_ctx->OMSetDepthStencilState(nullptr, 0);
            g_ctx->RSSetState(nullptr);

            D3D11_MAPPED_SUBRESOURCE m{};
            if (SUCCEEDED(g_ctx->Map(g_blitCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
            {
                float rect[4] = {0.0f, 0.0f, u1, v1};
                std::memcpy(m.pData, rect, sizeof(rect));
                g_ctx->Unmap(g_blitCb.Get(), 0);
            }
            g_ctx->IASetInputLayout(nullptr);
            g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            g_ctx->VSSetShader(g_blitVs.Get(), nullptr, 0);
            g_ctx->PSSetShader(g_blitPs.Get(), nullptr, 0);
            ID3D11Buffer *cb = g_blitCb.Get();
            g_ctx->VSSetConstantBuffers(0, 1, &cb);
            g_ctx->PSSetShaderResources(0, 1, &srv);
            ID3D11SamplerState *s = g_linearSampler.Get();
            g_ctx->PSSetSamplers(0, 1, &s);
            g_ctx->GSSetShader(nullptr, nullptr, 0);
            g_ctx->Draw(3, 0);
            ID3D11ShaderResourceView *nullSrv = nullptr;
            g_ctx->PSSetShaderResources(0, 1, &nullSrv);
            updateAndDrawFps(bh);
            drawModMenu(bw, bh);

            if (!g_shotDir.empty() && nowSeconds() >= g_nextShot)
            {
                g_nextShot = nowSeconds() + g_shotInterval;
                saveScreenshot();
            }
            g_swap->Present(g_config.vsync ? 1 : 0, 0);
        }

        // Called outside the D3D lock: waits for the swap chain (vsync) and applies the FPS cap.
        void waitForNextFrame()
        {
            if (g_frameWaitable && g_config.vsync)
                WaitForSingleObjectEx(g_frameWaitable, 100, TRUE);
            frameLimit();
        }

        uint8_t axisToByte(SHORT v, bool invert)
        {
            int iv = invert ? -static_cast<int>(v) - 1 : static_cast<int>(v);
            int b = (iv + 32768) >> 8;
            return static_cast<uint8_t>(std::clamp(b, 0, 255));
        }

        uint8_t applyDeadzone(uint8_t v)
        {
            int d = static_cast<int>(v) - 128;
            if (std::abs(d) < 24)
                return 0x80;
            return v;
        }
    }

    const HostConfig &config() { return g_config; }

    bool loadConfig(const char *iniPath)
    {
        if (!iniPath || GetFileAttributesA(iniPath) == INVALID_FILE_ATTRIBUTES)
            return false;
        // GetPrivateProfile* looks a relative file name up in the Windows directory, not the working directory.
        char fullPath[MAX_PATH];
        if (GetFullPathNameA(iniPath, MAX_PATH, fullPath, nullptr) != 0)
            iniPath = fullPath;
        g_config.width = GetPrivateProfileIntA("Display", "Width", g_config.width, iniPath);
        g_config.height = GetPrivateProfileIntA("Display", "Height", g_config.height, iniPath);
        g_config.fullscreen = GetPrivateProfileIntA("Display", "Fullscreen", g_config.fullscreen, iniPath) != 0;
        g_config.vsync = GetPrivateProfileIntA("Display", "VSync", g_config.vsync, iniPath) != 0;
        g_config.targetFps = GetPrivateProfileIntA("Display", "TargetFPS", g_config.targetFps, iniPath);
        g_config.keepAspect = GetPrivateProfileIntA("Display", "KeepAspect", g_config.keepAspect, iniPath) != 0;
        g_config.widescreen = GetPrivateProfileIntA("Display", "Widescreen", g_config.widescreen, iniPath) != 0;
        g_showFps = GetPrivateProfileIntA("Display", "ShowFPS", g_showFps ? 1 : 0, iniPath) != 0;
        // InternalScale=N: the GS is rendered at N x the PS2 resolution (640x448 -> 1920x1344 at 3), 1..6;
        // 0 = from the output size (init). The GS backend reads PS2X_GS_SCALE when it starts (an explicit
        // PS2X_GS_SCALE wins).
        g_internalScale = static_cast<int>(GetPrivateProfileIntA("Display", "InternalScale", 0, iniPath));
        if (g_internalScale >= 1 && g_internalScale <= 6 && !std::getenv("PS2X_GS_SCALE"))
            _putenv_s("PS2X_GS_SCALE", std::to_string(g_internalScale).c_str());
        if (const char *ws = std::getenv("PS2X_WIDESCREEN")) // test override
            g_config.widescreen = *ws != '0';
        return true;
    }

    bool init(const char *title)
    {
        if (g_ready)
            return true;
        SetUnhandledExceptionFilter(ps2xCrashFilter);

        QueryPerformanceFrequency(&g_qpcFreq);
        QueryPerformanceCounter(&g_qpcStart);
        timeBeginPeriod(1);
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

        HINSTANCE inst = GetModuleHandleW(nullptr);
        WNDCLASSEXW wc{sizeof(wc)};
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = wndProc;
        wc.hInstance = inst;
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        wc.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(1));
        wc.lpszClassName = L"PS2XNativeWindow";
        RegisterClassExW(&wc);

        DWORD style = WS_OVERLAPPEDWINDOW | WS_VISIBLE;
        int x = CW_USEDEFAULT, y = CW_USEDEFAULT;
        int w = static_cast<int>(g_config.width), h = static_cast<int>(g_config.height);
        if (g_config.fullscreen)
        {
            style = WS_POPUP | WS_VISIBLE;
            x = 0;
            y = 0;
            w = GetSystemMetrics(SM_CXSCREEN);
            h = GetSystemMetrics(SM_CYSCREEN);
        }
        if (g_internalScale == 0 && !std::getenv("PS2X_GS_SCALE"))
        {
            // InternalScale=0: enough GS pixels for the picture on screen. The 640x448 frame fills the screen's
            // height and 4:3 (or, with Widescreen, 16:9) of its width: 1080p -> 3, 1440p -> 4, 2160p (4K) -> 6.
            const double outW = static_cast<double>(w), outH = static_cast<double>(h);
            const double picW = std::min(outW, outH * (g_config.widescreen ? 16.0 / 9.0 : 4.0 / 3.0));
            const int scale = std::clamp(static_cast<int>(std::ceil(std::max(picW / 640.0, outH / 448.0) - 0.05)), 1, 6);
            _putenv_s("PS2X_GS_SCALE", std::to_string(scale).c_str());
            std::printf("[host] output %dx%d: internal scale %d%c", w, h, scale, 10);
        }
        if (!g_config.fullscreen)
        {
            RECT r{0, 0, w, h};
            AdjustWindowRect(&r, style, FALSE);
            w = r.right - r.left;
            h = r.bottom - r.top;
        }
        if (const char *bg = std::getenv("PS2X_BACKGROUND"); bg && *bg && *bg != '0')
        {
            g_background = true;
            style &= ~WS_VISIBLE;
            g_config.fullscreen = false;
        }
        if (const char *d = std::getenv("PS2X_SCREENSHOT_DIR"))
        {
            g_shotDir = d;
            CreateDirectoryA(d, nullptr);
            if (const char *iv = std::getenv("PS2X_SCREENSHOT_SEC"))
                g_shotInterval = std::max(0.5, std::atof(iv));
            g_nextShot = g_shotInterval;
        }
        for (auto &a : g_autoAnchorAt)
            a.store(-1.0);
        if (const char *an = std::getenv("PS2X_AUTOINPUT_ANCHOR"))
        {
            char *endp = nullptr;
            g_autoAnchorPc = static_cast<uint32_t>(std::strtoul(an, &endp, 16));
            if (endp && *endp == ':')
                g_autoAnchorCount = std::max(1, std::atoi(endp + 1));
        }
        if (const char *aw = std::getenv("PS2X_AUTOINPUT_WATCH"))
        {
            const char *q = aw;
            while (*q && g_autoWatchN < 8)
            {
                char *endp = nullptr;
                const unsigned long v = std::strtoul(q, &endp, 16);
                if (endp == q)
                    break;
                g_autoWatchPc[g_autoWatchN++] = static_cast<uint32_t>(v);
                q = (*endp == ',') ? endp + 1 : endp;
            }
        }
        if (const char *ai = std::getenv("PS2X_AUTOINPUT"))
        {
            // "seconds:BUTTON" (0.25 s press) or "from-to:BUTTON" (held) entries separated by commas.
            std::string spec = ai;
            size_t pos = 0;
            while (pos < spec.size())
            {
                size_t end = spec.find(',', pos);
                std::string item = spec.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
                size_t colon = item.find(':');
                if (colon != std::string::npos)
                {
                    std::string when = item.substr(0, colon);
                    // "@seconds" entries count from the first guest call of PS2X_AUTOINPUT_ANCHOR=<hex pc>
                    // (menu timing drifts with load; e.g. anchor OnInitLevelSelect_Tag for the arena chooser).
                    int anchorHit = 0;
                    if (!when.empty() && when[0] == '@')
                    {
                        when.erase(0, 1);
                        anchorHit = g_autoAnchorCount;
                        const size_t plus = when.find('+');
                        if (plus != std::string::npos)
                        {
                            anchorHit = std::max(1, std::atoi(when.c_str()));
                            when.erase(0, plus + 1);
                        }
                        anchorHit = std::min(anchorHit, kAutoAnchorHits);
                    }
                    const size_t dash = when.find('-', 1);
                    const double t = std::atof(when.c_str());
                    const double until = dash != std::string::npos ? std::atof(when.c_str() + dash + 1) : t + 0.25;
                    // "BUTTON#n": pad n (1-based; default 1), e.g. "CROSS#2" for a split-screen player 2.
                    std::string b = item.substr(colon + 1);
                    int padIndex = 0;
                    const size_t hash = b.find('#');
                    if (hash != std::string::npos)
                    {
                        padIndex = std::clamp(std::atoi(b.c_str() + hash + 1) - 1, 0, kMaxPads - 1);
                        b.erase(hash);
                    }
                    static const std::pair<const char *, uint16_t> names[] = {
                        {"START", kBtnStart}, {"SELECT", kBtnSelect}, {"CROSS", kBtnCross}, {"CIRCLE", kBtnCircle},
                        {"SQUARE", kBtnSquare}, {"TRIANGLE", kBtnTriangle}, {"UP", kBtnUp}, {"DOWN", kBtnDown},
                        {"LEFT", kBtnLeft}, {"RIGHT", kBtnRight}, {"L1", kBtnL1}, {"R1", kBtnR1}, {"L2", kBtnL2}, {"R2", kBtnR2}};
                    for (const auto &n : names)
                        if (b == n.first)
                            g_autoPress.push_back({t, until, n.second, anchorHit, padIndex});
                }
                if (end == std::string::npos)
                    break;
                pos = end + 1;
            }
        }
        g_baseTitle = widen(title);
        g_hwnd = CreateWindowExW(g_background ? (WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW) : 0, wc.lpszClassName, widen(title).c_str(), style, x, y, w, h,
                                 nullptr, nullptr, inst, nullptr);
        if (!g_hwnd)
            return false;

        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        if (GetEnvironmentVariableA("PS2X_D3D_DEBUG", nullptr, 0) > 0)
            flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels, 2,
                                       D3D11_SDK_VERSION, &g_device, nullptr, &g_ctx);
        if (FAILED(hr))
        {
            hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags, levels, 2,
                                   D3D11_SDK_VERSION, &g_device, nullptr, &g_ctx);
            if (FAILED(hr))
                return false;
        }

        ComPtr<IDXGIDevice> dxgiDev;
        g_device.As(&dxgiDev);
        ComPtr<IDXGIAdapter> adapter;
        dxgiDev->GetAdapter(&adapter);
        ComPtr<IDXGIFactory2> factory;
        adapter->GetParent(IID_PPV_ARGS(&factory));

        RECT rc{};
        GetClientRect(g_hwnd, &rc);
        DXGI_SWAP_CHAIN_DESC1 sd{};
        sd.Width = std::max<LONG>(1, rc.right - rc.left);
        sd.Height = std::max<LONG>(1, rc.bottom - rc.top);
        sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 2;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        hr = factory->CreateSwapChainForHwnd(g_device.Get(), g_hwnd, &sd, nullptr, nullptr, &g_swap);
        if (FAILED(hr))
            return false;
        {
            ComPtr<IDXGISwapChain2> swap2;
            if (SUCCEEDED(g_swap.As(&swap2)))
            {
                swap2->SetMaximumFrameLatency(1);
                g_frameWaitable = swap2->GetFrameLatencyWaitableObject();
            }
        }
        factory->MakeWindowAssociation(g_hwnd, DXGI_MWA_NO_ALT_ENTER);
        createBackBufferView();

        ComPtr<ID3DBlob> vsb, psb;
        if (!compileShader(kBlitShader, "vs_main", "vs_5_0", vsb) ||
            !compileShader(kBlitShader, "ps_main", "ps_5_0", psb))
            return false;
        g_device->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &g_blitVs);
        g_device->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &g_blitPs);

        D3D11_SAMPLER_DESC samp{};
        samp.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        samp.AddressU = samp.AddressV = samp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        samp.MaxLOD = D3D11_FLOAT32_MAX;
        g_device->CreateSamplerState(&samp, &g_linearSampler);

        D3D11_BUFFER_DESC cbd{};
        cbd.ByteWidth = 16;
        cbd.Usage = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        g_device->CreateBuffer(&cbd, nullptr, &g_blitCb);

        // XAudio2 (optional: a missing audio device must not stop the game).
        if (SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) || true)
        {
            if (SUCCEEDED(XAudio2Create(&g_xaudio, 0, XAUDIO2_DEFAULT_PROCESSOR)))
            {
                if (FAILED(g_xaudio->CreateMasteringVoice(&g_master, 2, 48000)))
                {
                    g_master = nullptr;
                    g_xaudio.Reset();
                }
                else if (const char *mute = std::getenv("PS2X_MUTE"); mute && *mute && *mute != '0')
                {
                    g_master->SetVolume(0.0f); // silent automated test runs
                }
            }
        }

        if (!g_background)
        {
            ShowWindow(g_hwnd, SW_SHOW);
            UpdateWindow(g_hwnd);
        }
        g_ready = true;
        std::fprintf(stdout, "[host] D3D11 %ux%u ready, XAudio2 %s\n", sd.Width, sd.Height,
                     g_master ? "ready" : "unavailable");
        return true;
    }

    void shutdown()
    {
        if (!g_ready)
            return;
        if (g_master)
        {
            g_master->DestroyVoice();
            g_master = nullptr;
        }
        g_xaudio.Reset();
        g_uploadSrv.Reset();
        g_uploadTex.Reset();
        g_backRtv.Reset();
        g_swap.Reset();
        g_ctx.Reset();
        g_device.Reset();
        if (g_frameWaitable)
            CloseHandle(g_frameWaitable);
        g_frameWaitable = nullptr;
        if (g_hwnd)
            DestroyWindow(g_hwnd);
        g_hwnd = nullptr;
        timeEndPeriod(1);
        g_ready = false;
    }

    bool isReady() { return g_ready; }

    bool pumpMessages()
    {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
                g_quit.store(true);
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        pollInput();
        const int fps = g_titleFps.load(), gameFps = g_titleGameFps.load();
        if (g_hwnd && fps >= 0 && (fps != g_shownTitleFps || gameFps != g_shownTitleGameFps))
        {
            g_shownTitleFps = fps;
            g_shownTitleGameFps = gameFps;
            const std::wstring t = g_baseTitle + L"  |  " + std::to_wstring(fps) + L" fps  (game " + std::to_wstring(gameFps) + L")";
            SetWindowTextW(g_hwnd, t.c_str());
        }
        return !g_quit.load();
    }

    bool quitRequested() { return g_quit.load(); }

    void presentRgba(const uint8_t *rgba, uint32_t width, uint32_t height, uint32_t stride)
    {
        if (!g_ready || !rgba || width == 0 || height == 0)
            return;
        waitForNextFrame();
        std::lock_guard lock(g_d3dMutex);
        if (!g_uploadTex || g_uploadW != width || g_uploadH != height)
        {
            g_uploadSrv.Reset();
            g_uploadTex.Reset();
            D3D11_TEXTURE2D_DESC td{};
            td.Width = width;
            td.Height = height;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            if (FAILED(g_device->CreateTexture2D(&td, nullptr, &g_uploadTex)))
                return;
            g_device->CreateShaderResourceView(g_uploadTex.Get(), nullptr, &g_uploadSrv);
            g_uploadW = width;
            g_uploadH = height;
        }
        g_ctx->UpdateSubresource(g_uploadTex.Get(), 0, nullptr, rgba, stride, 0);
        drawBlit(g_uploadSrv.Get(), width, height, 1.0f, 1.0f);
    }

    void setPresentWide(bool wide) { g_presentWide.store(wide, std::memory_order_relaxed); }

    void presentTexture(void *srv, uint32_t width, uint32_t height, float u1, float v1)
    {
        if (!g_ready || !srv)
            return;
        std::lock_guard lock(g_d3dMutex);
        drawBlit(static_cast<ID3D11ShaderResourceView *>(srv), width, height, u1, v1);
    }

    std::recursive_mutex &d3dLock() { return g_d3dMutex; }
    uint64_t d3dStateEpoch() { return g_d3dEpoch.load(std::memory_order_relaxed); }
    void beginFrame() { waitForNextFrame(); }
    ID3D11Device *device() { return g_device.Get(); }
    ID3D11DeviceContext *context() { return g_ctx.Get(); }
    void *windowHandle() { return g_hwnd; }

    void pollInput()
    {
        PadState pads[kMaxPads];
        // Hidden test runs (PS2X_BACKGROUND) and PS2X_NO_XINPUT=1 ignore real controllers: XInput is global, so a
        // pad the user plays something else with would otherwise drive the test too.
        static const bool s_noXinput = std::getenv("PS2X_BACKGROUND") != nullptr || std::getenv("PS2X_NO_XINPUT") != nullptr;
        for (int i = 0; i < kMaxPads && !s_noXinput; ++i)
        {
            XINPUT_STATE xs{};
            if (XInputGetState(static_cast<DWORD>(i), &xs) != ERROR_SUCCESS)
                continue;
            PadState &p = pads[i];
            p.connected = true;
            const WORD b = xs.Gamepad.wButtons;
            auto set = [&](WORD xb, uint16_t pb)
            { if (b & xb) p.buttons |= pb; };
            set(XINPUT_GAMEPAD_DPAD_UP, kBtnUp);
            set(XINPUT_GAMEPAD_DPAD_DOWN, kBtnDown);
            set(XINPUT_GAMEPAD_DPAD_LEFT, kBtnLeft);
            set(XINPUT_GAMEPAD_DPAD_RIGHT, kBtnRight);
            set(XINPUT_GAMEPAD_A, kBtnCross);
            set(XINPUT_GAMEPAD_B, kBtnCircle);
            set(XINPUT_GAMEPAD_X, kBtnSquare);
            set(XINPUT_GAMEPAD_Y, kBtnTriangle);
            set(XINPUT_GAMEPAD_LEFT_SHOULDER, kBtnL1);
            set(XINPUT_GAMEPAD_RIGHT_SHOULDER, kBtnR1);
            set(XINPUT_GAMEPAD_LEFT_THUMB, kBtnL3);
            set(XINPUT_GAMEPAD_RIGHT_THUMB, kBtnR3);
            set(XINPUT_GAMEPAD_START, kBtnStart);
            set(XINPUT_GAMEPAD_BACK, kBtnSelect);
            p.l2 = xs.Gamepad.bLeftTrigger;
            p.r2 = xs.Gamepad.bRightTrigger;
            if (p.l2 > 40)
                p.buttons |= kBtnL2;
            if (p.r2 > 40)
                p.buttons |= kBtnR2;
            p.lx = applyDeadzone(axisToByte(xs.Gamepad.sThumbLX, false));
            p.ly = applyDeadzone(axisToByte(xs.Gamepad.sThumbLY, true));
            p.rx = applyDeadzone(axisToByte(xs.Gamepad.sThumbRX, false));
            p.ry = applyDeadzone(axisToByte(xs.Gamepad.sThumbRY, true));
        }

        if (!g_autoPress.empty())
        {
            const double t = nowSeconds();
            // PS2X_AUTOINPUT_STOP_AT_ANCHOR=1: timed (unanchored) presses end at the first anchor call -- the START
            // taps that get a slow boot past the title must not reach the menu the anchored presses navigate.
            static const bool s_stopAtAnchor = std::getenv("PS2X_AUTOINPUT_STOP_AT_ANCHOR") != nullptr;
            const bool anchored = g_autoAnchorAt[0].load() >= 0.0;
            for (const auto &ap : g_autoPress)
            {
                if (!ap.anchorHit && s_stopAtAnchor && anchored)
                    continue;
                const double anchorAt = ap.anchorHit ? g_autoAnchorAt[ap.anchorHit - 1].load() : 0.0;
                if (ap.anchorHit && anchorAt < 0.0)
                    continue;
                const double rt = ap.anchorHit ? t - anchorAt : t;
                if (ap.pad > 0)
                    pads[ap.pad].connected = true; // a scripted pad 2..4 is plugged in from boot
                if (rt >= ap.t && rt < ap.until)
                    pads[ap.pad].buttons |= ap.button;
            }
        }
        std::lock_guard lock(g_inputMutex);
        // Keyboard drives player 1 (merged with XInput pad 0).
        PadState &k = pads[0];
        auto key = [&](int vk, uint16_t pb)
        { if (g_keys[vk]) { k.buttons |= pb; k.connected = true; } };
        key(VK_UP, kBtnUp);
        key(VK_DOWN, kBtnDown);
        key(VK_LEFT, kBtnLeft);
        key(VK_RIGHT, kBtnRight);
        key('X', kBtnCross);
        key(VK_SPACE, kBtnCross);
        key('C', kBtnCircle);
        key(VK_BACK, kBtnCircle);
        key('Z', kBtnSquare);
        key('V', kBtnTriangle);
        key('Q', kBtnL1);
        key('E', kBtnR1);
        key('1', kBtnL2);
        key('3', kBtnR2);
        key(VK_RETURN, kBtnStart);
        key(VK_RSHIFT, kBtnSelect);
        key(VK_TAB, kBtnSelect);
        float ax = 0, ay = 0;
        if (g_keys['D']) ax += 1;
        if (g_keys['A']) ax -= 1;
        if (g_keys['S']) ay += 1;
        if (g_keys['W']) ay -= 1;
        if (ax != 0 || ay != 0)
        {
            k.lx = static_cast<uint8_t>(std::clamp((ax + 1.0f) * 127.5f, 0.0f, 255.0f));
            k.ly = static_cast<uint8_t>(std::clamp((ay + 1.0f) * 127.5f, 0.0f, 255.0f));
        }
        // Keyboard player is always considered present so the game has a pad on port 0.
        k.connected = true;
        // The mod menu (F2, BACK + RB, or the main menu's m plank) takes player 1's input while it is up.
        modmenu::filterInput(pads, kMaxPads, g_keys[VK_F2]);
        for (int i = 0; i < kMaxPads; ++i)
            g_pads[i] = pads[i];
    }

    PadState pad(int index)
    {
        if (index < 0 || index >= kMaxPads)
            return {};
        std::lock_guard lock(g_inputMutex);
        return g_pads[index];
    }

    int connectedPadCount()
    {
        std::lock_guard lock(g_inputMutex);
        int n = 0;
        for (const auto &p : g_pads)
            n += p.connected ? 1 : 0;
        return n;
    }

    bool keyDown(int vk)
    {
        if (vk < 0 || vk >= 256)
            return false;
        std::lock_guard lock(g_inputMutex);
        return g_keys[vk];
    }

    void setVibration(int index, uint8_t smallMotor, uint8_t largeMotor)
    {
        if (index < 0 || index >= kMaxPads)
            return;
        XINPUT_VIBRATION v{};
        v.wLeftMotorSpeed = static_cast<WORD>(largeMotor * 257);
        v.wRightMotorSpeed = static_cast<WORD>(smallMotor ? 0xFFFF : 0);
        XInputSetState(static_cast<DWORD>(index), &v);
    }

    IXAudio2 *xaudio() { return g_xaudio.Get(); }
    IXAudio2MasteringVoice *masteringVoice() { return g_master; }
    bool audioReady() { return g_master != nullptr; }

    double timeSeconds() { return nowSeconds(); }
    double lastFrameMs() { return g_lastFrameMs; }

    // ---- Sampling profiler (PS2X_PROFILE=<output file>) ------------------------------------------
    namespace
    {
        std::atomic<bool> g_profRun{false};
        std::thread g_profThread;
        HANDLE g_profTarget = nullptr;
        std::string g_profPath;
        std::unordered_map<std::string, uint64_t> g_profStacks;
    }

    void profileCurrentThread()
    {
        const char *out = std::getenv("PS2X_PROFILE");
        if (!out || !*out || g_profRun.load())
            return;
        g_profPath = out;
        DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &g_profTarget,
                        THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0);
        g_profRun.store(true);
        g_profThread = std::thread([]()
        {
            std::unordered_map<uint64_t, uint64_t> hist;
            const uint64_t base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
            uint64_t samples = 0;
            auto dump = [&]()
            {
                if (FILE *f = std::fopen(g_profPath.c_str(), "w"))
                {
                    std::fprintf(f, "base %llx samples %llu\n", (unsigned long long)base, (unsigned long long)samples);
                    for (const auto &[rip, n] : hist)
                        std::fprintf(f, "%llx %llu\n", (unsigned long long)rip, (unsigned long long)n);
                    for (const auto &[stack, n] : g_profStacks)
                        std::fprintf(f, "S %llu%s\n", (unsigned long long)n, stack.c_str());
                    std::fclose(f);
                }
            };
            auto lastDump = std::chrono::steady_clock::now();
            // PS2X_PROFILE_SEC=<s>: start sampling that long after launch (e.g. once the menu is up).
            static const double s_profStart = std::getenv("PS2X_PROFILE_SEC") ? std::atof(std::getenv("PS2X_PROFILE_SEC")) : 0.0;
            const auto profBoot = std::chrono::steady_clock::now();
            while (g_profRun.load() && std::chrono::duration<double>(std::chrono::steady_clock::now() - profBoot).count() < s_profStart)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            while (g_profRun.load())
            {
                // PS2X_PROFILE_STALL=1 (with PS2X_HITCH_LOG): sample only while no frame has been flipped for 50 ms.
                static const bool s_stallOnly = std::getenv("PS2X_PROFILE_STALL") != nullptr;
                if (s_stallOnly)
                {
                    const double lastFlip = g_hitchLastFlip.load(std::memory_order_relaxed);
                    if (lastFlip <= 0.0 || nowSeconds() - lastFlip < 0.05)
                    {
                        Sleep(1);
                        if (std::chrono::steady_clock::now() - lastDump > std::chrono::seconds(5))
                        {
                            dump();
                            lastDump = std::chrono::steady_clock::now();
                        }
                        continue;
                    }
                }
                // PS2X_PROFILE_ALL=1 samples every thread of the process (guest threads run on their own
                // host threads, so the game thread alone mostly shows hand-off waits).
                static const bool s_all = std::getenv("PS2X_PROFILE_ALL") != nullptr;
                static std::vector<HANDLE> s_threads;
                static auto s_lastEnum = std::chrono::steady_clock::time_point{};
                if (s_all && std::chrono::steady_clock::now() - s_lastEnum > std::chrono::milliseconds(500))
                {
                    s_lastEnum = std::chrono::steady_clock::now();
                    for (HANDLE h : s_threads)
                        CloseHandle(h);
                    s_threads.clear();
                    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                    if (snap != INVALID_HANDLE_VALUE)
                    {
                        THREADENTRY32 te{};
                        te.dwSize = sizeof(te);
                        for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te))
                        {
                            if (te.th32OwnerProcessID != GetCurrentProcessId() || te.th32ThreadID == GetCurrentThreadId())
                                continue;
                            if (HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID))
                                s_threads.push_back(h);
                        }
                        CloseHandle(snap);
                    }
                }
                std::vector<HANDLE> targets = s_all ? s_threads : std::vector<HANDLE>{g_profTarget};
                for (HANDLE target : targets)
                {
                if (SuspendThread(target) != static_cast<DWORD>(-1))
                {
                    CONTEXT c{};
                    c.ContextFlags = CONTEXT_FULL;
                    if (GetThreadContext(target, &c))
                    {
                        // Attribute samples outside the exe to the first caller frame inside it.
                        const uint64_t modBase = base;
                        const uint64_t modEnd = base + 0x10000000ull;
                        uint64_t attributed = c.Rip;
                        CONTEXT u = c;
                        for (int depth = 0; depth < 24 && !(u.Rip >= modBase && u.Rip < modEnd); ++depth)
                        {
                            DWORD64 imageBase = 0;
                            PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(u.Rip, &imageBase, nullptr);
                            if (!fn)
                            {
                                u.Rip = *reinterpret_cast<DWORD64 *>(u.Rsp);
                                u.Rsp += 8;
                            }
                            else
                            {
                                PVOID handlerData = nullptr;
                                DWORD64 establisher = 0;
                                RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, u.Rip, fn, &u, &handlerData, &establisher, nullptr);
                            }
                            if (u.Rip == 0)
                                break;
                        }
                        if (c.Rip < modBase || c.Rip >= modEnd)
                            attributed = (u.Rip >= modBase && u.Rip < modEnd) ? (u.Rip | 0x8000000000000000ull) : 0x7FFFFFFFFFFFFFFFull;
                        // No allocation while the target is suspended (it may hold the heap lock).
                        uint64_t frames[10];
                        int nframes = 0;
                        CONTEXT w = (c.Rip >= modBase && c.Rip < modEnd) ? c : u;
                        for (; nframes < 10 && w.Rip >= modBase && w.Rip < modEnd; ++nframes)
                        {
                            frames[nframes] = w.Rip;
                            DWORD64 imageBase = 0;
                            PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(w.Rip, &imageBase, nullptr);
                            if (!fn)
                            {
                                ++nframes;
                                break;
                            }
                            PVOID handlerData = nullptr;
                            DWORD64 establisher = 0;
                            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, w.Rip, fn, &w, &handlerData, &establisher, nullptr);
                        }
                        ResumeThread(target);
                        ++hist[attributed];
                        std::string key;
                        for (int i = 0; i < nframes; ++i)
                        {
                            char buf[24];
                            std::snprintf(buf, sizeof(buf), " %llx", static_cast<unsigned long long>(frames[i]));
                            key += buf;
                        }
                        ++g_profStacks[key];
                        ++samples;
                        continue;
                    }
                    ResumeThread(target);
                }
                }
                Sleep(1);
                if (std::chrono::steady_clock::now() - lastDump > std::chrono::seconds(5))
                {
                    dump();
                    lastDump = std::chrono::steady_clock::now();
                }
            }
            dump();
        });
        g_profThread.detach();
    }

}
