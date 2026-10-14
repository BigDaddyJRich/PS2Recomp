#pragma once

// Native Windows host layer: Win32 window, Direct3D 11 presentation, XInput pads,
// keyboard, and XAudio2 audio. Replaces the raylib host on Windows builds.

#include <cstdint>
#include <mutex>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct IXAudio2;
struct IXAudio2MasteringVoice;

namespace ps2host
{
    constexpr int kMaxPads = 4;

    struct HostConfig
    {
        uint32_t width = 1920;
        uint32_t height = 1080;
        bool fullscreen = false;
        bool vsync = true;
        uint32_t targetFps = 60;
        bool keepAspect = true; // 4:3 pillarbox for the PS2 framebuffer
        bool widescreen = false;
    };

    // PS2 DualShock 2 buttons in libpad bit order (active-high here; callers invert).
    enum PadButton : uint16_t
    {
        kBtnSelect = 1u << 0,
        kBtnL3 = 1u << 1,
        kBtnR3 = 1u << 2,
        kBtnStart = 1u << 3,
        kBtnUp = 1u << 4,
        kBtnRight = 1u << 5,
        kBtnDown = 1u << 6,
        kBtnLeft = 1u << 7,
        kBtnL2 = 1u << 8,
        kBtnR2 = 1u << 9,
        kBtnL1 = 1u << 10,
        kBtnR1 = 1u << 11,
        kBtnTriangle = 1u << 12,
        kBtnCircle = 1u << 13,
        kBtnCross = 1u << 14,
        kBtnSquare = 1u << 15,
    };

    struct PadState
    {
        bool connected = false;
        uint16_t buttons = 0; // active-high PadButton mask
        uint8_t lx = 0x80, ly = 0x80, rx = 0x80, ry = 0x80;
        uint8_t l2 = 0, r2 = 0; // analog trigger pressure
    };

    const HostConfig &config();
    bool loadConfig(const char *iniPath);

    bool init(const char *title);
    void shutdown();
    bool isReady();

    // Pump Win32 messages. Returns false once the user asked to quit.
    bool pumpMessages();
    bool quitRequested();

    // Present a guest framebuffer (RGBA8, tightly packed rows of `stride` bytes).
    // Scaled with bilinear filtering into the 1080p back buffer.
    void presentRgba(const uint8_t *rgba, uint32_t width, uint32_t height, uint32_t stride);
    // Blocks (outside any lock) until the next frame should be presented: vsync + FPS cap.
    // presentRgba() calls this itself; presentTexture() callers must call it first.
    void beginFrame();
    // Present whatever is currently bound as the hardware-rendered guest frame.
    void presentTexture(void *shaderResourceView, uint32_t width, uint32_t height,
                        float u1 = 1.0f, float v1 = 1.0f);

    // Serializes all use of the immediate context (game thread draws vs host presents).
    std::recursive_mutex &d3dLock();
    ID3D11Device *device();
    ID3D11DeviceContext *context();
    void *windowHandle();

    // Input -------------------------------------------------------------
    void pollInput();
    PadState pad(int index);        // XInput pad (index 0..3) merged with keyboard on pad 0
    int connectedPadCount();
    bool keyDown(int virtualKey);
    void setVibration(int index, uint8_t smallMotor, uint8_t largeMotor);

    // Audio -------------------------------------------------------------
    IXAudio2 *xaudio();
    IXAudio2MasteringVoice *masteringVoice();
    bool audioReady();

    // Timing ------------------------------------------------------------
    double timeSeconds();
    double lastFrameMs();

    // Debug: start sampling the calling thread's instruction pointer (PS2X_PROFILE=<file>).
    void profileCurrentThread();
}
