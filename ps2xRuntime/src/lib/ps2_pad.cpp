#include "runtime/ps2_pad.h"
#include "runtime/ps2_host_settings.h"
#include "ps2_host_backend.h"
#include "ps2_host_input.h"
#include <cstdlib>
#include <cstring>

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;

    int multitapPort()
    {
        const int port = ps2_host_settings::get().multitapPort;
        return (port >= 0 && port < ps2_pad_layout::kPortCount) ? port : -1;
    }

    // VENDETTA_PADS=<n>: the first n places count as plugged in (tests drive them with VENDETTA_PADSCRIPT)
    int testPadCount()
    {
        static const int count = []
        {
            const char *value = std::getenv("VENDETTA_PADS");
            return (value && *value) ? std::atoi(value) : 0;
        }();
        return count;
    }

    // VENDETTA_KEYBOARD=<player> (settings.toml [controls] keyboard): which player the keyboard plays as, 1-4, or 0 =
    // nobody; that player's controller still works too. Unset: player 1. VENDETTA_NO_HOST_INPUT=1 (test runs): nobody.
    int keyboardPadIndex()
    {
        static const int index = []
        {
            const char *ignored = std::getenv("VENDETTA_NO_HOST_INPUT");
            if (ignored && ignored[0] != '0')
                return -1;
            const char *value = std::getenv("VENDETTA_KEYBOARD");
            const int player = (value && *value) ? std::atoi(value) : 1;
            return (player >= 1 && player <= 4) ? player - 1 : -1;
        }();
        return index;
    }
}

namespace ps2_pad_layout
{
    int slotCount(int port)
    {
        return (port >= 0 && port == multitapPort()) ? kSlotCount : 1;
    }

    bool padPlace(int index, int &port, int &slot)
    {
        if (index < 0)
            return false;
        // port 1 (and its slots when the multitap is there), then port 2 (and its slots)
        for (int p = 0; p < kPortCount; ++p)
        {
            const int slots = slotCount(p);
            if (index < slots)
            {
                port = p;
                slot = index;
                return true;
            }
            index -= slots;
        }
        return false;
    }

    int padIndex(int port, int slot)
    {
        if (port < 0 || port >= kPortCount || slot < 0 || slot >= slotCount(port))
            return -1;
        int index = slot;
        for (int p = 0; p < port; ++p)
            index += slotCount(p);
        return index;
    }

    int hostController(int port, int slot)
    {
        const int index = padIndex(port, slot);
        if (index < 0 || !host_input::windowReady())
            return -1;
        for (int gamepad = 0, seen = 0; gamepad < 4; ++gamepad)
        {
            if (host_input::gamepadAvailable(gamepad) && seen++ == index)
                return gamepad;
        }
        return -1;
    }

    bool connected(int port, int slot)
    {
        const int index = padIndex(port, slot);
        if (index < 0)
            return false;
        return index < testPadCount() || index == keyboardPadIndex() || hostController(port, slot) >= 0;
    }

    int xinputOrder(int port, int slot, const char **name)
    {
        const int gamepad = hostController(port, slot);
        if (name)
            *name = gamepad >= 0 ? host_input::gamepadName(gamepad) : nullptr;
        auto isXInput = [](int pad)
        {
            const char *padName = host_input::gamepadName(pad);
            if (!padName)
                return false;
            return std::strcmp(padName, "Xbox Controller") == 0 || std::strcmp(padName, "Wireless Xbox Controller") == 0 ||
                   std::strncmp(padName, "XInput ", 7) == 0 || std::strcmp(padName, "Unknown XInput Device") == 0;
        };
        if (gamepad < 0 || !isXInput(gamepad))
            return -1;
        int order = 0;
        for (int pad = 0; pad < gamepad; ++pad)
            if (host_input::gamepadAvailable(pad) && isXInput(pad))
                ++order;
        return order;
    }
}

bool PSPadBackend::readState(int port, int slot, uint8_t *data, size_t size)
{
    if (!data || size < 32)
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;

    uint16_t btns = 0xFFFFu;
    // Each pad has its own controller (ps2_pad_layout: the first in port 1, the next ones in port 2 / its multitap);
    // the keyboard plays as one player too (keyboardPadIndex), with that player's controller. (One controller used to
    // drive every port, so player 1 and player 2 moved together.)
    const int kGamepad = ps2_pad_layout::hostController(port, slot);
    const bool useGamepad = kGamepad >= 0;
    auto clearBit = [&btns](uint16_t mask)
    { btns &= ~mask; };

    if (useGamepad)
    {
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_UP))
            clearBit(PAD_UP);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_DOWN))
            clearBit(PAD_DOWN);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_LEFT))
            clearBit(PAD_LEFT);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_RIGHT))
            clearBit(PAD_RIGHT);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_DOWN))
            clearBit(PAD_CROSS);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT))
            clearBit(PAD_CIRCLE);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_LEFT))
            clearBit(PAD_SQUARE);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_UP))
            clearBit(PAD_TRIANGLE);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_1))
            clearBit(PAD_L1);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_1))
            clearBit(PAD_R1);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_2))
            clearBit(PAD_L2);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_2))
            clearBit(PAD_R2);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            clearBit(PAD_START);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_LEFT))
            clearBit(PAD_SELECT);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_THUMB))
            clearBit(PAD_L3);
        if (host_input::gamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_THUMB))
            clearBit(PAD_R3);

        float lx = host_input::gamepadAxis(kGamepad, GAMEPAD_AXIS_LEFT_X);
        float ly = host_input::gamepadAxis(kGamepad, GAMEPAD_AXIS_LEFT_Y);
        float rx = host_input::gamepadAxis(kGamepad, GAMEPAD_AXIS_RIGHT_X);
        float ry = host_input::gamepadAxis(kGamepad, GAMEPAD_AXIS_RIGHT_Y);
        data[6] = static_cast<uint8_t>(128 + lx * 127);
        data[7] = static_cast<uint8_t>(128 + ly * 127);
        data[4] = static_cast<uint8_t>(128 + rx * 127);
        data[5] = static_cast<uint8_t>(128 + ry * 127);
    }
    if (ps2_pad_layout::padIndex(port, slot) == keyboardPadIndex() && host_input::windowReady())
    {
        if (host_input::keyDown(KEY_UP) || host_input::keyDown(KEY_W))
            clearBit(PAD_UP);
        if (host_input::keyDown(KEY_DOWN) || host_input::keyDown(KEY_S))
            clearBit(PAD_DOWN);
        if (host_input::keyDown(KEY_LEFT) || host_input::keyDown(KEY_A))
            clearBit(PAD_LEFT);
        if (host_input::keyDown(KEY_RIGHT) || host_input::keyDown(KEY_D))
            clearBit(PAD_RIGHT);
        if (host_input::keyDown(KEY_X) || host_input::keyDown(KEY_SPACE))
            clearBit(PAD_CROSS);
        if (host_input::keyDown(KEY_C) || host_input::keyDown(KEY_ESCAPE))
            clearBit(PAD_CIRCLE);
        if (host_input::keyDown(KEY_Z) || host_input::keyDown(KEY_KP_0))
            clearBit(PAD_SQUARE);
        if (host_input::keyDown(KEY_V) || host_input::keyDown(KEY_KP_1))
            clearBit(PAD_TRIANGLE);
        if (host_input::keyDown(KEY_Q))
            clearBit(PAD_L1);
        if (host_input::keyDown(KEY_E))
            clearBit(PAD_R1);
        if (host_input::keyDown(KEY_LEFT_SHIFT))
            clearBit(PAD_L2);
        if (host_input::keyDown(KEY_RIGHT_SHIFT))
            clearBit(PAD_R2);
        if (host_input::keyDown(KEY_ENTER))
            clearBit(PAD_START);
        if (host_input::keyDown(KEY_TAB))
            clearBit(PAD_SELECT);
        // the right stick (taunts, Blazin'): I J K L, all the way over while held
        const int rx = (host_input::keyDown(KEY_L) ? 1 : 0) - (host_input::keyDown(KEY_J) ? 1 : 0);
        const int ry = (host_input::keyDown(KEY_K) ? 1 : 0) - (host_input::keyDown(KEY_I) ? 1 : 0);
        if (rx != 0 || ry != 0)
        {
            data[4] = static_cast<uint8_t>(rx > 0 ? 0xFF : (rx < 0 ? 0x00 : kPadStickCenter));
            data[5] = static_cast<uint8_t>(ry > 0 ? 0xFF : (ry < 0 ? 0x00 : kPadStickCenter));
        }
    }

    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    return true;
}
