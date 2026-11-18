#include "runtime/ps2_pad.h"
#include "runtime/ps2_host_settings.h"
#include "ps2_host_backend.h"
#include "ps2_host_input.h"
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

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

    // the PC's controllers the pads can use: raylib / GLFW give 4; the native Win32 host 8 (XInput users 1-4, then
    // PlayStation 4 / 5 controllers over HID), so every pad of a multitap setup can have its own
#if defined(PS2X_NATIVE_HOST)
    constexpr int kHostControllers = 8;
#else
    constexpr int kHostControllers = 4;
#endif

    int keyboardPadIndex();

    // binding order (ps2_host_settings::bindingName): the 16 buttons, then the sticks' 8 directions
    constexpr uint16_t kBindingMask[ps2_host_settings::kPadButtonBindings] = {
        PAD_UP, PAD_DOWN, PAD_LEFT, PAD_RIGHT, PAD_CROSS, PAD_CIRCLE, PAD_SQUARE, PAD_TRIANGLE,
        PAD_L1, PAD_R1, PAD_L2, PAD_R2, PAD_L3, PAD_R3, PAD_START, PAD_SELECT};

    std::string lowerTrim(const std::string &text)
    {
        std::string out;
        for (char c : text)
            if (!std::isspace(static_cast<unsigned char>(c)))
                out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return out;
    }

    // a key's name (settings.toml key_<button>) -> raylib's KEY_* (the native host's are Windows' virtual keys)
    int keyCode(const std::string &name)
    {
        const std::string n = lowerTrim(name);
        if (n.size() == 1 && n[0] >= 'a' && n[0] <= 'z')
            return KEY_A + (n[0] - 'a');
        if (n.size() == 1 && n[0] >= '0' && n[0] <= '9')
            return KEY_ZERO + (n[0] - '0');
        if (n.size() >= 2 && n[0] == 'f' && std::isdigit(static_cast<unsigned char>(n[1])))
        {
            const int f = std::atoi(n.c_str() + 1);
            if (f >= 1 && f <= 12)
                return KEY_F1 + (f - 1);
        }
        for (const char *prefix : {"numpad", "kp"})
        {
            const size_t len = std::strlen(prefix);
            if (n.size() == len + 1 && n.compare(0, len, prefix) == 0 && std::isdigit(static_cast<unsigned char>(n[len])))
                return KEY_KP_0 + (n[len] - '0');
        }
        struct Named
        {
            const char *name;
            int key;
        };
        static const Named named[] = {
            {"up", KEY_UP}, {"down", KEY_DOWN}, {"left", KEY_LEFT}, {"right", KEY_RIGHT}, {"space", KEY_SPACE},
            {"enter", KEY_ENTER}, {"return", KEY_ENTER}, {"tab", KEY_TAB}, {"backspace", KEY_BACKSPACE},
            {"escape", KEY_ESCAPE}, {"esc", KEY_ESCAPE}, {"lshift", KEY_LEFT_SHIFT}, {"leftshift", KEY_LEFT_SHIFT},
            {"rshift", KEY_RIGHT_SHIFT}, {"rightshift", KEY_RIGHT_SHIFT}, {"lctrl", KEY_LEFT_CONTROL},
            {"leftctrl", KEY_LEFT_CONTROL}, {"rctrl", KEY_RIGHT_CONTROL}, {"rightctrl", KEY_RIGHT_CONTROL},
            {"lalt", KEY_LEFT_ALT}, {"leftalt", KEY_LEFT_ALT}, {"ralt", KEY_RIGHT_ALT}, {"rightalt", KEY_RIGHT_ALT},
            {"insert", KEY_INSERT}, {"delete", KEY_DELETE}, {"home", KEY_HOME}, {"end", KEY_END},
            {"pageup", KEY_PAGE_UP}, {"pagedown", KEY_PAGE_DOWN}, {"semicolon", KEY_SEMICOLON}, {"equals", KEY_EQUAL},
            {"equal", KEY_EQUAL}, {"comma", KEY_COMMA}, {"minus", KEY_MINUS}, {"period", KEY_PERIOD},
            {"slash", KEY_SLASH}, {"grave", KEY_GRAVE}, {"lbracket", KEY_LEFT_BRACKET}, {"backslash", KEY_BACKSLASH},
            {"rbracket", KEY_RIGHT_BRACKET}, {"apostrophe", KEY_APOSTROPHE}, {"quote", KEY_APOSTROPHE}};
        for (const Named &k : named)
            if (n == k.name)
                return k.key;
        return -1;
    }

    // a controller button's name (pad_<button>, PlayStation names) -> its PAD_* bit; 0 = "None" / not known
    uint16_t padButtonMask(const std::string &name)
    {
        const std::string n = lowerTrim(name);
        for (int i = 0; i < ps2_host_settings::kPadButtonBindings; ++i)
            if (n == lowerTrim(ps2_host_settings::defaultPadButton(i)))
                return kBindingMask[i];
        return 0;
    }

    struct Bindings
    {
        std::vector<int> keys[ps2_host_settings::kBindings];
        uint16_t padSource[ps2_host_settings::kPadButtonBindings] = {}; // the controller's PAD_* bit for each button
    };

    // settings.toml [controls] key_* / pad_* (empty = the defaults), read once
    const Bindings &bindings()
    {
        static const Bindings b = []
        {
            Bindings out;
            const ps2_host_settings::Settings &settings = ps2_host_settings::get();
            int changed = 0;
            for (int i = 0; i < ps2_host_settings::kBindings; ++i)
            {
                const std::string &text = settings.keyBindings[i];
                const std::string list = text.empty() ? std::string(ps2_host_settings::defaultKeys(i)) : text;
                size_t start = 0;
                for (;;)
                {
                    const size_t comma = list.find(',', start);
                    const std::string one =
                        list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                    if (!lowerTrim(one).empty() && lowerTrim(one) != "none")
                    {
                        const int key = keyCode(one);
                        if (key >= 0)
                            out.keys[i].push_back(key);
                        else
                            std::cerr << "[settings] key_" << ps2_host_settings::bindingName(i) << ": no key called '"
                                      << one << "'" << std::endl;
                    }
                    if (comma == std::string::npos)
                        break;
                    start = comma + 1;
                }
                if (!text.empty() && lowerTrim(text) != lowerTrim(ps2_host_settings::defaultKeys(i)))
                    ++changed;
                if (i < ps2_host_settings::kPadButtonBindings)
                {
                    const std::string &pad = settings.padBindings[i];
                    out.padSource[i] = padButtonMask(pad.empty() ? std::string(ps2_host_settings::defaultPadButton(i)) : pad);
                    if (!pad.empty() && lowerTrim(pad) != lowerTrim(ps2_host_settings::defaultPadButton(i)))
                        ++changed;
                }
            }
            std::cerr << "[settings] controls: " << changed << " button binding(s) changed from the defaults, keyboard = player "
                      << (keyboardPadIndex() + 1) << std::endl;
            return out;
        }();
        return b;
    }

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
        for (int gamepad = 0, seen = 0; gamepad < kHostControllers; ++gamepad)
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

    const Bindings &binds = bindings();
    if (useGamepad)
    {
        // the controller's buttons as a PlayStation pad's (PAD_* bits), then through the pad_<button> bindings
        uint16_t held = 0;
        auto hold = [&](int button, uint16_t mask)
        {
            if (host_input::gamepadButtonDown(kGamepad, button))
                held |= mask;
        };
        hold(GAMEPAD_BUTTON_LEFT_FACE_UP, PAD_UP);
        hold(GAMEPAD_BUTTON_LEFT_FACE_DOWN, PAD_DOWN);
        hold(GAMEPAD_BUTTON_LEFT_FACE_LEFT, PAD_LEFT);
        hold(GAMEPAD_BUTTON_LEFT_FACE_RIGHT, PAD_RIGHT);
        hold(GAMEPAD_BUTTON_RIGHT_FACE_DOWN, PAD_CROSS);
        hold(GAMEPAD_BUTTON_RIGHT_FACE_RIGHT, PAD_CIRCLE);
        hold(GAMEPAD_BUTTON_RIGHT_FACE_LEFT, PAD_SQUARE);
        hold(GAMEPAD_BUTTON_RIGHT_FACE_UP, PAD_TRIANGLE);
        hold(GAMEPAD_BUTTON_LEFT_TRIGGER_1, PAD_L1);
        hold(GAMEPAD_BUTTON_RIGHT_TRIGGER_1, PAD_R1);
        hold(GAMEPAD_BUTTON_LEFT_TRIGGER_2, PAD_L2);
        hold(GAMEPAD_BUTTON_RIGHT_TRIGGER_2, PAD_R2);
        hold(GAMEPAD_BUTTON_MIDDLE_RIGHT, PAD_START);
        hold(GAMEPAD_BUTTON_MIDDLE_LEFT, PAD_SELECT);
        hold(GAMEPAD_BUTTON_LEFT_THUMB, PAD_L3);
        hold(GAMEPAD_BUTTON_RIGHT_THUMB, PAD_R3);
        for (int i = 0; i < ps2_host_settings::kPadButtonBindings; ++i)
            if (held & binds.padSource[i])
                clearBit(kBindingMask[i]);

        float lx = host_input::gamepadAxis(kGamepad, GAMEPAD_AXIS_LEFT_X);
        float ly = host_input::gamepadAxis(kGamepad, GAMEPAD_AXIS_LEFT_Y);
        float rx = host_input::gamepadAxis(kGamepad, GAMEPAD_AXIS_RIGHT_X);
        float ry = host_input::gamepadAxis(kGamepad, GAMEPAD_AXIS_RIGHT_Y);
        data[6] = static_cast<uint8_t>(128 + lx * 127);
        data[7] = static_cast<uint8_t>(128 + ly * 127);
        data[4] = static_cast<uint8_t>(128 + rx * 127);
        data[5] = static_cast<uint8_t>(128 + ry * 127);
    }
    // the keyboard's player: the key_<button> bindings (settings.toml [controls]); the sticks all the way over
    if (ps2_pad_layout::padIndex(port, slot) == keyboardPadIndex() && host_input::windowReady())
    {
        auto down = [&binds](int binding)
        {
            for (int key : binds.keys[binding])
                if (host_input::keyDown(key))
                    return true;
            return false;
        };
        for (int i = 0; i < ps2_host_settings::kPadButtonBindings; ++i)
            if (down(i))
                clearBit(kBindingMask[i]);
        auto stick = [&down](int negative, int positive, uint8_t &axis)
        {
            const int v = (down(positive) ? 1 : 0) - (down(negative) ? 1 : 0);
            if (v != 0)
                axis = static_cast<uint8_t>(v > 0 ? 0xFF : 0x00);
        };
        stick(18, 19, data[6]); // left stick x: lstick_left / lstick_right
        stick(16, 17, data[7]); // left stick y: lstick_up / lstick_down
        stick(22, 23, data[4]); // right stick x: rstick_left / rstick_right
        stick(20, 21, data[5]); // right stick y: rstick_up / rstick_down
    }

    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    return true;
}
