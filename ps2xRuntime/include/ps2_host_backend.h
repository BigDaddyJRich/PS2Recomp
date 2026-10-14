#pragma once

// The host API: raylib, or on the native Windows host (PS2X_NATIVE_HOST) its subset on Win32 / D3D11 / XAudio2.
#if defined(PS2X_NATIVE_HOST)
#include "host/raylib_shim.h"
#else
#include "raylib.h"
#endif
