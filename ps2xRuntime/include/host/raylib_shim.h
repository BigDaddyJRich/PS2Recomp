#pragma once

// The subset of the raylib API the runtime uses, implemented on the native Win32 host (host/win32_host.h:
// Direct3D 11 window, XInput, XAudio2) for the PS2X_NATIVE_HOST build. Same names, types and layouts as raylib, so the
// runtime, the input code and the game overrides compile unchanged against either host.
//
// - Window / monitor calls map onto ps2host (sizes, quit); the window itself is made by ps2host::init.
// - Textures are CPU pixel buffers; DrawTexturePro presents the Direct3D 11 GS renderer's latched frame when that
//   renderer is active, else the texture's pixels (the CPU renderer's picture).
// - KEY_* are Win32 virtual-key codes; gamepads are ps2host's XInput pads (keyboard merged into pad 0).
// - AudioStream = an XAudio2 source voice refilled from the stream's callback; Sound / Wave (the HLE VAG sample
//   player, unused by HCTP's IOP sound path) are no-ops.

#include "host/win32_host.h"

#include <cstdint>

// <windows.h> / <mmsystem.h> name macros that would rename the raylib functions below
#ifdef PlaySound
#undef PlaySound
#endif

struct Vector2
{
    float x;
    float y;
};

struct Rectangle
{
    float x;
    float y;
    float width;
    float height;
};

struct Color
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

struct Image
{
    void *data;
    int width;
    int height;
    int mipmaps;
    int format;
};

struct Texture2D
{
    unsigned int id;
    int width;
    int height;
    int mipmaps;
    int format;
};
using Texture = Texture2D;

struct AudioStream
{
    void *buffer; // the shim's stream object
    void *processor;
    unsigned int sampleRate;
    unsigned int sampleSize;
    unsigned int channels;
};

struct Wave
{
    unsigned int frameCount;
    unsigned int sampleRate;
    unsigned int sampleSize;
    unsigned int channels;
    void *data;
};

struct Sound
{
    AudioStream stream;
    unsigned int frameCount;
};

using AudioCallback = void (*)(void *bufferData, unsigned int frames);

#define WHITE Color{255, 255, 255, 255}
#define BLACK Color{0, 0, 0, 255}
#define BLANK Color{0, 0, 0, 0}
#define MAGENTA Color{255, 0, 255, 255}

enum
{
    FLAG_VSYNC_HINT = 0x00000040,
    FLAG_FULLSCREEN_MODE = 0x00000002,
    FLAG_WINDOW_RESIZABLE = 0x00000004,
    FLAG_WINDOW_UNDECORATED = 0x00000008,
    FLAG_WINDOW_HIDDEN = 0x00000080,
    FLAG_WINDOW_MINIMIZED = 0x00000200,
    FLAG_WINDOW_MAXIMIZED = 0x00000400,
    FLAG_WINDOW_UNFOCUSED = 0x00000800,
    FLAG_WINDOW_TOPMOST = 0x00001000,
    FLAG_BORDERLESS_WINDOWED_MODE = 0x00008000,
};

enum
{
    TEXTURE_FILTER_POINT = 0,
    TEXTURE_FILTER_BILINEAR,
};

enum
{
    PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 = 7,
};

// Win32 virtual-key codes
enum
{
    KEY_SPACE = 0x20,
    KEY_ESCAPE = 0x1B,
    KEY_ENTER = 0x0D,
    KEY_TAB = 0x09,
    KEY_LEFT = 0x25,
    KEY_UP = 0x26,
    KEY_RIGHT = 0x27,
    KEY_DOWN = 0x28,
    KEY_A = 'A', KEY_B = 'B', KEY_C = 'C', KEY_D = 'D', KEY_E = 'E', KEY_F = 'F', KEY_G = 'G', KEY_H = 'H',
    KEY_I = 'I', KEY_J = 'J', KEY_K = 'K', KEY_L = 'L', KEY_M = 'M', KEY_N = 'N', KEY_O = 'O', KEY_P = 'P',
    KEY_Q = 'Q', KEY_R = 'R', KEY_S = 'S', KEY_T = 'T', KEY_U = 'U', KEY_V = 'V', KEY_W = 'W', KEY_X = 'X',
    KEY_Y = 'Y', KEY_Z = 'Z',
    KEY_ZERO = '0', KEY_ONE = '1', KEY_TWO = '2', KEY_THREE = '3', KEY_FOUR = '4',
    KEY_FIVE = '5', KEY_SIX = '6', KEY_SEVEN = '7', KEY_EIGHT = '8', KEY_NINE = '9',
    KEY_KP_0 = 0x60, KEY_KP_1, KEY_KP_2, KEY_KP_3, KEY_KP_4, KEY_KP_5, KEY_KP_6, KEY_KP_7, KEY_KP_8, KEY_KP_9,
    KEY_F1 = 0x70, KEY_F2, KEY_F3, KEY_F4, KEY_F5, KEY_F6, KEY_F7, KEY_F8, KEY_F9, KEY_F10, KEY_F11, KEY_F12,
    KEY_LEFT_SHIFT = 0xA0,
    KEY_RIGHT_SHIFT = 0xA1,
    KEY_LEFT_CONTROL = 0xA2,
    KEY_RIGHT_CONTROL = 0xA3,
    KEY_LEFT_ALT = 0xA4,
    KEY_RIGHT_ALT = 0xA5,
    KEY_BACKSPACE = 0x08,
};

// raylib's gamepad numbering
enum
{
    GAMEPAD_BUTTON_UNKNOWN = 0,
    GAMEPAD_BUTTON_LEFT_FACE_UP,
    GAMEPAD_BUTTON_LEFT_FACE_RIGHT,
    GAMEPAD_BUTTON_LEFT_FACE_DOWN,
    GAMEPAD_BUTTON_LEFT_FACE_LEFT,
    GAMEPAD_BUTTON_RIGHT_FACE_UP,
    GAMEPAD_BUTTON_RIGHT_FACE_RIGHT,
    GAMEPAD_BUTTON_RIGHT_FACE_DOWN,
    GAMEPAD_BUTTON_RIGHT_FACE_LEFT,
    GAMEPAD_BUTTON_LEFT_TRIGGER_1,
    GAMEPAD_BUTTON_LEFT_TRIGGER_2,
    GAMEPAD_BUTTON_RIGHT_TRIGGER_1,
    GAMEPAD_BUTTON_RIGHT_TRIGGER_2,
    GAMEPAD_BUTTON_MIDDLE_LEFT,
    GAMEPAD_BUTTON_MIDDLE,
    GAMEPAD_BUTTON_MIDDLE_RIGHT,
    GAMEPAD_BUTTON_LEFT_THUMB,
    GAMEPAD_BUTTON_RIGHT_THUMB,
};

enum
{
    GAMEPAD_AXIS_LEFT_X = 0,
    GAMEPAD_AXIS_LEFT_Y,
    GAMEPAD_AXIS_RIGHT_X,
    GAMEPAD_AXIS_RIGHT_Y,
    GAMEPAD_AXIS_LEFT_TRIGGER,
    GAMEPAD_AXIS_RIGHT_TRIGGER,
};

// window / monitor
void InitWindow(int width, int height, const char *title);
void CloseWindow();
bool WindowShouldClose();
bool IsWindowReady();
bool IsWindowState(unsigned int flag);
void SetWindowState(unsigned int flags);
void ClearWindowState(unsigned int flags);
void SetConfigFlags(unsigned int flags);
void SetWindowTitle(const char *title);
void SetWindowPosition(int x, int y);
void SetWindowMonitor(int monitor);
void SetWindowSize(int width, int height);
void ToggleFullscreen();
void ToggleBorderlessWindowed();
void MaximizeWindow();
void HideCursor();
void ShowCursor();
int GetScreenWidth();
int GetScreenHeight();
int GetRenderWidth();
int GetRenderHeight();
int GetMonitorCount();
int GetCurrentMonitor();
Vector2 GetMonitorPosition(int monitor);
int GetMonitorWidth(int monitor);
int GetMonitorHeight(int monitor);
int GetMonitorRefreshRate(int monitor);
void SetTargetFPS(int fps);

// drawing
void BeginDrawing();
void EndDrawing();
void ClearBackground(Color color);
Image GenImageColor(int width, int height, Color color);
void UnloadImage(Image image);
Texture2D LoadTextureFromImage(Image image);
void UnloadTexture(Texture2D texture);
void UpdateTexture(Texture2D texture, const void *pixels);
void SetTextureFilter(Texture2D texture, int filter);
void DrawTexturePro(Texture2D texture, Rectangle source, Rectangle dest, Vector2 origin, float rotation, Color tint);

// input
bool IsKeyDown(int key);
bool IsKeyPressed(int key);
bool IsGamepadAvailable(int gamepad);
const char *GetGamepadName(int gamepad);
bool IsGamepadButtonDown(int gamepad, int button);
float GetGamepadAxisMovement(int gamepad, int axis);

// audio
void InitAudioDevice();
void CloseAudioDevice();
bool IsAudioDeviceReady();
void SetMasterVolume(float volume);
AudioStream LoadAudioStream(unsigned int sampleRate, unsigned int sampleSize, unsigned int channels);
void UnloadAudioStream(AudioStream stream);
void SetAudioStreamCallback(AudioStream stream, AudioCallback callback);
void PlayAudioStream(AudioStream stream);
void StopAudioStream(AudioStream stream);
Wave LoadWaveFromMemory(const char *fileType, const unsigned char *fileData, int dataSize);
void UnloadWave(Wave wave);
Sound LoadSoundFromWave(Wave wave);
void UnloadSound(Sound sound);
void PlaySound(Sound sound);
void StopSound(Sound sound);
bool IsSoundPlaying(Sound sound);
void SetSoundVolume(Sound sound, float volume);
void SetSoundPitch(Sound sound, float pitch);
