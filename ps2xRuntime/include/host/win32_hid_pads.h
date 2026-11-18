#pragma once

// PlayStation 4 / 5 controllers over HID for the native Win32 host (win32_hid_pads.cpp). They are the host's
// controllers kHidFirstIndex.. (after the 4 XInput users).
#include "host/win32_host.h"

namespace ps2host::hidpads
{
    constexpr int kHidFirstIndex = kXInputPads;
    constexpr int kHidPads = kMaxPads - kXInputPads;

    void start(); // looks for controllers every 2 s from a background thread
    void stop();
    // the Nth HID controller's state (index 0..kHidPads-1); false = none there
    bool read(int index, PadState &out, const char *&name);
    void setVibration(int index, uint8_t smallMotor, uint8_t largeMotor);
}
