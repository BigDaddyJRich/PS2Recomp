// The raylib subset on the native Win32 host (see host/raylib_shim.h).
#include "host/raylib_shim.h"
#include "runtime/gs/gs_d3d11_backend.h"
#include "runtime/gs/ps2_gif_arbiter.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <xaudio2.h>
#ifdef PlaySound
#undef PlaySound
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Diagnostics globals the ported host / D3D11 renderer read (Crush Hour keeps them in its runtime): the FPS counter and
// hitch log, the GS trace windows and the transfer log's GIF path label.
std::atomic<double> g_hitchLastFlip{-1.0};
std::atomic<double> g_ps2xBank1LoadedAt{-1.0};
std::atomic<double> g_ps2xTraceAnchorAt{-1.0};
thread_local GifPathId g_gifCurrentPath = GifPathId::Path2;
namespace
{
    std::atomic<uint64_t> g_shimPresents{0};
}
// game frames (the host's FPS counter, GAME line): frames the game presented, not window refreshes
uint64_t ps2Vif1KickCount()
{
    return GSD3D11Backend::IsActive() ? GSD3D11Backend::LatchedFrameCount() : g_shimPresents.load(std::memory_order_relaxed);
}

// ---- window / monitor ----------------------------------------------------------------------------------------------
void InitWindow(int, int, const char *) {} // ps2host::init made the window
void CloseWindow() { ps2host::shutdown(); }
bool WindowShouldClose() { return !ps2host::pumpMessages() || ps2host::quitRequested(); }
bool IsWindowReady() { return ps2host::isReady(); }
bool IsWindowState(unsigned int flag) { return flag == FLAG_FULLSCREEN_MODE ? ps2host::config().fullscreen : false; }
void SetWindowState(unsigned int) {}
void ClearWindowState(unsigned int) {}
void SetConfigFlags(unsigned int) {}
void SetWindowTitle(const char *) {} // the host's title shows the game and the frame rate
void SetWindowPosition(int, int) {}
void SetWindowMonitor(int) {}
void SetWindowSize(int, int) {}
void ToggleFullscreen() {}
void ToggleBorderlessWindowed() {}
void MaximizeWindow() {}
void HideCursor() {}
void ShowCursor() {}
int GetScreenWidth() { return static_cast<int>(ps2host::config().width); }
int GetScreenHeight() { return static_cast<int>(ps2host::config().height); }
int GetRenderWidth() { return GetScreenWidth(); }
int GetRenderHeight() { return GetScreenHeight(); }
int GetMonitorCount() { return std::max(1, GetSystemMetrics(SM_CMONITORS)); }
int GetCurrentMonitor() { return 0; }
Vector2 GetMonitorPosition(int) { return Vector2{0.0f, 0.0f}; }
int GetMonitorWidth(int) { return GetSystemMetrics(SM_CXSCREEN); }
int GetMonitorHeight(int) { return GetSystemMetrics(SM_CYSCREEN); }
int GetMonitorRefreshRate(int)
{
    DEVMODEW mode{};
    mode.dmSize = sizeof(mode);
    return EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode) && mode.dmDisplayFrequency > 1 ? static_cast<int>(mode.dmDisplayFrequency) : 60;
}
void SetTargetFPS(int) {} // the host paces presentation (HostConfig::targetFps / vsync)

// ---- drawing -------------------------------------------------------------------------------------------------------
namespace
{
    struct ShimTexture
    {
        int width = 0, height = 0;
        std::vector<uint8_t> rgba;
    };
    std::mutex g_texMutex;
    std::unordered_map<unsigned int, ShimTexture> g_textures;
    unsigned int g_nextTextureId = 1;
}

void BeginDrawing() {}
void EndDrawing()
{
    const uint64_t window = g_shimPresents.fetch_add(1, std::memory_order_relaxed) + 1u;
    // every 5 s: window refreshes and game frames per second ([native-fps], the measured frame rate)
    static double s_last = -1.0;
    static uint64_t s_lastWindow = 0, s_lastGame = 0;
    const double now = ps2host::timeSeconds();
    if (s_last < 0.0)
    {
        s_last = now;
        s_lastWindow = window;
        s_lastGame = ps2Vif1KickCount();
    }
    else if (now - s_last >= 5.0)
    {
        const uint64_t game = ps2Vif1KickCount();
        std::fprintf(stderr, "[native-fps] window=%.1f game=%.1f%c", (window - s_lastWindow) / (now - s_last),
                     (game - s_lastGame) / (now - s_last), 10);
        s_last = now;
        s_lastWindow = window;
        s_lastGame = game;
    }
}
void ClearBackground(Color) {}

Image GenImageColor(int width, int height, Color color)
{
    Image image{};
    image.width = std::max(width, 0);
    image.height = std::max(height, 0);
    image.mipmaps = 1;
    image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    const size_t pixels = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
    auto *data = static_cast<uint8_t *>(std::malloc(pixels * 4u + 4u));
    for (size_t i = 0; i < pixels; ++i)
    {
        data[i * 4 + 0] = color.r;
        data[i * 4 + 1] = color.g;
        data[i * 4 + 2] = color.b;
        data[i * 4 + 3] = color.a;
    }
    image.data = data;
    return image;
}

void UnloadImage(Image image) { std::free(image.data); }

Texture2D LoadTextureFromImage(Image image)
{
    std::lock_guard lock(g_texMutex);
    Texture2D tex{};
    tex.id = g_nextTextureId++;
    tex.width = image.width;
    tex.height = image.height;
    tex.mipmaps = 1;
    tex.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    ShimTexture &t = g_textures[tex.id];
    t.width = image.width;
    t.height = image.height;
    t.rgba.assign(static_cast<size_t>(image.width) * static_cast<size_t>(image.height) * 4u, 0u);
    if (image.data)
        std::memcpy(t.rgba.data(), image.data, t.rgba.size());
    return tex;
}

void UnloadTexture(Texture2D texture)
{
    std::lock_guard lock(g_texMutex);
    g_textures.erase(texture.id);
}

void UpdateTexture(Texture2D texture, const void *pixels)
{
    std::lock_guard lock(g_texMutex);
    auto it = g_textures.find(texture.id);
    if (it != g_textures.end() && pixels)
        std::memcpy(it->second.rgba.data(), pixels, it->second.rgba.size());
}

void SetTextureFilter(Texture2D, int) {} // the host scales with a linear sampler

void DrawTexturePro(Texture2D texture, struct Rectangle source, struct Rectangle, Vector2, float, Color)
{
    // The native GS module may own a Vulkan child swapchain over the Win32
    // host window. In that mode do not upload the raylib texture or present a
    // second D3D11 frame underneath the direct Vulkan image.
    if (ps2host::directVulkanPresentation())
        return;
    // the Direct3D 11 GS renderer shows its own latched frame (no CPU pixels)
    if (GSD3D11Backend::IsActive() && GSD3D11Backend::PresentLatchedToHost())
        return;
    std::lock_guard lock(g_texMutex);
    auto it = g_textures.find(texture.id);
    if (it == g_textures.end())
        return;
    const ShimTexture &t = it->second;
    const uint32_t w = static_cast<uint32_t>(std::clamp<float>(source.width, 1.0f, static_cast<float>(t.width)));
    const uint32_t h = static_cast<uint32_t>(std::clamp<float>(std::abs(source.height), 1.0f, static_cast<float>(t.height)));
    ps2host::presentRgba(t.rgba.data(), w, h, static_cast<uint32_t>(t.width) * 4u);
}

// ---- input ---------------------------------------------------------------------------------------------------------
bool IsKeyDown(int key) { return ps2host::keyDown(key); }

bool IsKeyPressed(int key)
{
    static bool s_was[256] = {};
    if (key < 0 || key >= 256)
        return false;
    const bool down = ps2host::keyDown(key);
    const bool pressed = down && !s_was[key];
    s_was[key] = down;
    return pressed;
}

bool IsGamepadAvailable(int gamepad) { return ps2host::pad(gamepad).connected; }
const char *GetGamepadName(int) { return "XInput"; }

bool IsGamepadButtonDown(int gamepad, int button)
{
    const ps2host::PadState p = ps2host::pad(gamepad);
    if (!p.connected)
        return false;
    uint16_t bit = 0;
    switch (button)
    {
    case GAMEPAD_BUTTON_LEFT_FACE_UP: bit = ps2host::kBtnUp; break;
    case GAMEPAD_BUTTON_LEFT_FACE_RIGHT: bit = ps2host::kBtnRight; break;
    case GAMEPAD_BUTTON_LEFT_FACE_DOWN: bit = ps2host::kBtnDown; break;
    case GAMEPAD_BUTTON_LEFT_FACE_LEFT: bit = ps2host::kBtnLeft; break;
    case GAMEPAD_BUTTON_RIGHT_FACE_UP: bit = ps2host::kBtnTriangle; break;
    case GAMEPAD_BUTTON_RIGHT_FACE_RIGHT: bit = ps2host::kBtnCircle; break;
    case GAMEPAD_BUTTON_RIGHT_FACE_DOWN: bit = ps2host::kBtnCross; break;
    case GAMEPAD_BUTTON_RIGHT_FACE_LEFT: bit = ps2host::kBtnSquare; break;
    case GAMEPAD_BUTTON_LEFT_TRIGGER_1: bit = ps2host::kBtnL1; break;
    case GAMEPAD_BUTTON_LEFT_TRIGGER_2: bit = ps2host::kBtnL2; break;
    case GAMEPAD_BUTTON_RIGHT_TRIGGER_1: bit = ps2host::kBtnR1; break;
    case GAMEPAD_BUTTON_RIGHT_TRIGGER_2: bit = ps2host::kBtnR2; break;
    case GAMEPAD_BUTTON_MIDDLE_LEFT: bit = ps2host::kBtnSelect; break;
    case GAMEPAD_BUTTON_MIDDLE_RIGHT: bit = ps2host::kBtnStart; break;
    case GAMEPAD_BUTTON_LEFT_THUMB: bit = ps2host::kBtnL3; break;
    case GAMEPAD_BUTTON_RIGHT_THUMB: bit = ps2host::kBtnR3; break;
    default: return false;
    }
    return (p.buttons & bit) != 0;
}

float GetGamepadAxisMovement(int gamepad, int axis)
{
    const ps2host::PadState p = ps2host::pad(gamepad);
    if (!p.connected)
        return 0.0f;
    auto stick = [](uint8_t v) { return std::clamp((static_cast<float>(v) - 127.5f) / 127.5f, -1.0f, 1.0f); };
    auto trigger = [](uint8_t v) { return static_cast<float>(v) / 127.5f - 1.0f; }; // raylib: -1 released .. 1 full
    switch (axis)
    {
    case GAMEPAD_AXIS_LEFT_X: return stick(p.lx);
    case GAMEPAD_AXIS_LEFT_Y: return stick(p.ly);
    case GAMEPAD_AXIS_RIGHT_X: return stick(p.rx);
    case GAMEPAD_AXIS_RIGHT_Y: return stick(p.ry);
    case GAMEPAD_AXIS_LEFT_TRIGGER: return trigger(p.l2);
    case GAMEPAD_AXIS_RIGHT_TRIGGER: return trigger(p.r2);
    default: return 0.0f;
    }
}

// ---- audio ---------------------------------------------------------------------------------------------------------
namespace
{
    // One stream: an XAudio2 source voice with kBuffers buffers in flight, each refilled from the callback when the
    // voice has played it (OnBufferEnd runs on XAudio2's thread, as raylib's callback runs on its mixer thread).
    struct ShimStream final : IXAudio2VoiceCallback
    {
        static constexpr int kBuffers = 3;
        static constexpr unsigned kFrames = 1024;
        IXAudio2SourceVoice *voice = nullptr;
        AudioCallback callback = nullptr;
        unsigned channels = 2, bytesPerSample = 2;
        std::vector<uint8_t> buffers[kBuffers];
        int next = 0;
        std::atomic<bool> playing{false};
        std::mutex m;

        // silent = prime with zeros: PlayAudioStream runs on the caller's thread, which may hold the lock the
        // callback takes (native ADX start) - raylib only calls the callback from its audio thread, so do the same.
        void submit(int index, bool silent = false)
        {
            std::vector<uint8_t> &b = buffers[index];
            AudioCallback cb = nullptr;
            if (!silent)
            {
                std::lock_guard lock(m);
                cb = callback;
            }
            if (cb)
                cb(b.data(), kFrames);
            else
                std::fill(b.begin(), b.end(), 0u);
            XAUDIO2_BUFFER xb{};
            xb.AudioBytes = static_cast<UINT32>(b.size());
            xb.pAudioData = b.data();
            xb.pContext = reinterpret_cast<void *>(static_cast<intptr_t>(index));
            voice->SubmitSourceBuffer(&xb);
        }

        void STDMETHODCALLTYPE OnBufferEnd(void *context) override
        {
            if (playing.load(std::memory_order_acquire))
                submit(static_cast<int>(reinterpret_cast<intptr_t>(context)));
        }
        void STDMETHODCALLTYPE OnVoiceProcessingPassStart(UINT32) override {}
        void STDMETHODCALLTYPE OnVoiceProcessingPassEnd() override {}
        void STDMETHODCALLTYPE OnStreamEnd() override {}
        void STDMETHODCALLTYPE OnBufferStart(void *) override {}
        void STDMETHODCALLTYPE OnLoopEnd(void *) override {}
        void STDMETHODCALLTYPE OnVoiceError(void *, HRESULT) override {}
    };
}

void InitAudioDevice() {} // ps2host::init started XAudio2
void CloseAudioDevice() {}
bool IsAudioDeviceReady() { return ps2host::audioReady(); }

void SetMasterVolume(float volume)
{
    if (IXAudio2MasteringVoice *master = ps2host::masteringVoice())
        master->SetVolume(std::clamp(volume, 0.0f, 1.0f));
}

AudioStream LoadAudioStream(unsigned int sampleRate, unsigned int sampleSize, unsigned int channels)
{
    AudioStream out{};
    out.sampleRate = sampleRate;
    out.sampleSize = sampleSize;
    out.channels = channels;
    IXAudio2 *xa = ps2host::xaudio();
    if (!xa || (sampleSize != 16u && sampleSize != 32u) || channels == 0u || channels > 2u || sampleRate == 0u)
        return out;
    auto *s = new ShimStream();
    s->channels = channels;
    s->bytesPerSample = sampleSize / 8u;
    WAVEFORMATEX fmt{};
    fmt.wFormatTag = sampleSize == 32u ? WAVE_FORMAT_IEEE_FLOAT : WAVE_FORMAT_PCM;
    fmt.nChannels = static_cast<WORD>(channels);
    fmt.nSamplesPerSec = sampleRate;
    fmt.wBitsPerSample = static_cast<WORD>(sampleSize);
    fmt.nBlockAlign = static_cast<WORD>(channels * s->bytesPerSample);
    fmt.nAvgBytesPerSec = sampleRate * fmt.nBlockAlign;
    if (FAILED(xa->CreateSourceVoice(&s->voice, &fmt, 0, XAUDIO2_DEFAULT_FREQ_RATIO, s)))
    {
        delete s;
        return out;
    }
    for (auto &b : s->buffers)
        b.assign(ShimStream::kFrames * fmt.nBlockAlign, 0u);
    out.buffer = s;
    return out;
}

void UnloadAudioStream(AudioStream stream)
{
    auto *s = static_cast<ShimStream *>(stream.buffer);
    if (!s)
        return;
    s->playing.store(false, std::memory_order_release);
    if (s->voice)
    {
        s->voice->Stop();
        s->voice->FlushSourceBuffers();
        s->voice->DestroyVoice(); // waits for the callbacks to finish
    }
    delete s;
}

void StopAudioStream(AudioStream stream)
{
    auto *s = static_cast<ShimStream *>(stream.buffer);
    if (!s || !s->voice || !s->playing.exchange(false))
        return;
    s->voice->Stop();
    s->voice->FlushSourceBuffers();
}

void SetAudioStreamCallback(AudioStream stream, AudioCallback callback)
{
    if (auto *s = static_cast<ShimStream *>(stream.buffer))
    {
        std::lock_guard lock(s->m);
        s->callback = callback;
    }
}

void PlayAudioStream(AudioStream stream)
{
    auto *s = static_cast<ShimStream *>(stream.buffer);
    if (!s || !s->voice || s->playing.exchange(true))
        return;
    for (int i = 0; i < ShimStream::kBuffers; ++i)
        s->submit(i, true);
    s->voice->Start();
}

Wave LoadWaveFromMemory(const char *, const unsigned char *, int) { return Wave{}; }
void UnloadWave(Wave) {}
Sound LoadSoundFromWave(Wave) { return Sound{}; }
void UnloadSound(Sound) {}
void PlaySound(Sound) {}
void StopSound(Sound) {}
bool IsSoundPlaying(Sound) { return false; }
void SetSoundVolume(Sound, float) {}
void SetSoundPitch(Sound, float) {}
