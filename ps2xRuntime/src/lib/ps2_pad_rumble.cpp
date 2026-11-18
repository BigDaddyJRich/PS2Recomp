#include "runtime/ps2_pad_rumble.h"
#include "runtime/ps2_pad.h"
#include "runtime/ps2_host_settings.h"

#include <array>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <Xinput.h>
#endif
#if defined(PS2X_NATIVE_HOST)
#include "host/win32_host.h"
#endif

namespace ps2_pad_rumble
{
    namespace
    {
        std::mutex g_mutex;

#if defined(_WIN32)
        struct Motors
        {
            WORD left = 0;  // low frequency: the PS2's large motor
            WORD right = 0; // high frequency: the small motor
            bool sent = false;
        };
        std::array<Motors, XUSER_MAX_COUNT> g_sent{};

        // XInput's connected controllers (asking an empty slot is slow, so at most every 2 s)
        DWORD g_connectedMask = 0;
        std::chrono::steady_clock::time_point g_maskTime{};

        DWORD connectedMask()
        {
            const auto now = std::chrono::steady_clock::now();
            if (g_maskTime == std::chrono::steady_clock::time_point{} || now - g_maskTime > std::chrono::seconds(2))
            {
                g_connectedMask = 0;
                for (DWORD user = 0; user < XUSER_MAX_COUNT; ++user)
                {
                    XINPUT_STATE state{};
                    if (XInputGetState(user, &state) == ERROR_SUCCESS)
                        g_connectedMask |= 1u << user;
                }
                g_maskTime = now;
            }
            return g_connectedMask;
        }

        // the XInput user for the Nth XInput controller GLFW lists (both count in XInput's order)
        int xinputUser(int order)
        {
            const DWORD mask = connectedMask();
            for (DWORD user = 0; user < XUSER_MAX_COUNT; ++user)
                if ((mask & (1u << user)) && order-- == 0)
                    return static_cast<int>(user);
            return -1;
        }

        void send(int user, WORD left, WORD right)
        {
            Motors &motors = g_sent[static_cast<size_t>(user)];
            if (motors.sent && motors.left == left && motors.right == right)
                return;
            XINPUT_VIBRATION vibration{left, right};
            XInputSetState(static_cast<DWORD>(user), &vibration);
            motors = Motors{left, right, true};
        }
#endif
    }

#if defined(PS2X_NATIVE_HOST)
    namespace
    {
        // what each host controller was last sent (the game sets its motors every frame)
        struct HostMotors
        {
            uint8_t small = 0, large = 0;
            bool sent = false;
        };
        std::array<HostMotors, ps2host::kMaxPads> g_hostSent{};

        void sendHost(int controller, uint8_t smallMotor, uint8_t largeMotor)
        {
            if (controller < 0 || controller >= ps2host::kMaxPads)
                return;
            HostMotors &m = g_hostSent[static_cast<size_t>(controller)];
            if (m.sent && m.small == smallMotor && m.large == largeMotor)
                return;
            ps2host::setVibration(controller, smallMotor, largeMotor);
            m = HostMotors{smallMotor, largeMotor, true};
        }
    }
#endif

    void set(int port, int slot, bool smallMotor, uint8_t largeMotor)
    {
#if defined(PS2X_NATIVE_HOST)
        // The native host knows each controller (XInput users 1-4, PlayStation 4 / 5 controllers over HID): the
        // motors go to the controller behind the pad (ps2_pad_layout::hostController), whatever kind it is.
        const int player = ps2_pad_layout::padIndex(port, slot);
        const ps2_host_settings::Settings &settings = ps2_host_settings::get();
        const bool allowed = player >= 0 && (player >= 4 || settings.vibration[player]);
        const int controller = ps2_pad_layout::hostController(port, slot);
        std::lock_guard<std::mutex> lock(g_mutex);
        static bool s_reported[ps2_pad_layout::kPortCount][ps2_pad_layout::kSlotCount] = {};
        if (port >= 0 && port < ps2_pad_layout::kPortCount && slot >= 0 && slot < ps2_pad_layout::kSlotCount &&
            !s_reported[port][slot] && (smallMotor || largeMotor))
        {
            s_reported[port][slot] = true;
            const char *name = controller >= 0 ? ps2host::padName(controller) : nullptr;
            std::cerr << "[rumble] pad " << (port + 1) << static_cast<char>('A' + slot) << " (player " << (player + 1)
                      << "): " << (name ? name : "no controller")
                      << (controller >= 0 ? " -> controller " + std::to_string(controller + 1) : std::string())
                      << (allowed ? "" : " (vibration off in the settings)") << std::endl;
        }
        if (controller < 0)
            return;
        sendHost(controller, allowed && smallMotor ? 0xFF : 0, allowed ? largeMotor : 0);
#elif defined(_WIN32)
        const int player = ps2_pad_layout::padIndex(port, slot);
        const ps2_host_settings::Settings &settings = ps2_host_settings::get();
        const bool allowed = player >= 0 && (player >= 4 || settings.vibration[player]);
        const char *name = nullptr;
        const int order = ps2_pad_layout::xinputOrder(port, slot, &name);

        std::lock_guard<std::mutex> lock(g_mutex);
        // once per pad: where its motors go
        static bool s_reported[ps2_pad_layout::kPortCount][ps2_pad_layout::kSlotCount] = {};
        const int user = order >= 0 ? xinputUser(order) : -1;
        if (port >= 0 && port < ps2_pad_layout::kPortCount && slot >= 0 && slot < ps2_pad_layout::kSlotCount &&
            !s_reported[port][slot] && (smallMotor || largeMotor))
        {
            s_reported[port][slot] = true;
            std::cerr << "[rumble] pad " << (port + 1) << static_cast<char>('A' + slot) << " (player " << (player + 1)
                      << "): " << (name ? name : "no controller")
                      << (user >= 0 ? " -> XInput controller " + std::to_string(user + 1)
                                    : std::string(name ? " (not an XInput controller: no vibration)" : ""))
                      << (allowed ? "" : " (vibration off in the settings)") << std::endl;
        }
        if (user < 0)
            return;
        if (!allowed)
        {
            send(user, 0, 0);
            return;
        }
        send(user, static_cast<WORD>(largeMotor * 257u), smallMotor ? 0xFFFFu : 0u);
#else
        (void)port;
        (void)slot;
        (void)smallMotor;
        (void)largeMotor;
#endif
    }

    void stopAll()
    {
#if defined(PS2X_NATIVE_HOST)
        std::lock_guard<std::mutex> lock(g_mutex);
        for (int controller = 0; controller < ps2host::kMaxPads; ++controller)
            if (g_hostSent[static_cast<size_t>(controller)].sent)
                sendHost(controller, 0, 0);
#elif defined(_WIN32)
        std::lock_guard<std::mutex> lock(g_mutex);
        for (int user = 0; user < static_cast<int>(XUSER_MAX_COUNT); ++user)
            if (g_sent[static_cast<size_t>(user)].sent)
                send(user, 0, 0);
#endif
    }
}
