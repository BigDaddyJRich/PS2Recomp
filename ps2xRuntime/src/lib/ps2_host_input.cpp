// The PC's keyboard and controllers for the pads: raylib's window, or a GLFW window read the way raylib reads it
// (see ps2_host_input.h).

#include "ps2_host_input.h"
#include "ps2_host_backend.h"

#if defined(PS2X_HAS_GLFW)
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <string>

namespace host_input
{
    namespace
    {
        constexpr int kGamepads = 4;
        constexpr int kButtons = 32; // raylib's GamepadButton values fit
        constexpr int kAxes = 6;

        struct Gamepad
        {
            bool ready = false;
            std::string name;
            std::array<bool, kButtons> buttons{};
            std::array<float, kAxes> axes{};
        };

        std::atomic<GLFWwindow *> g_window{nullptr};
        std::array<std::atomic<bool>, 512> g_keys{}; // GLFW_KEY_LAST is 348
        std::mutex g_padMutex;
        std::array<Gamepad, kGamepads> g_pads; // under g_padMutex

#if defined(PS2X_HAS_GLFW)
        GLFWkeyfun g_previousKeyCallback = nullptr;

        void onKey(GLFWwindow *window, int key, int scancode, int action, int mods)
        {
            if (key >= 0 && key < static_cast<int>(g_keys.size()))
                g_keys[static_cast<size_t>(key)].store(action != GLFW_RELEASE, std::memory_order_relaxed);
            if (g_previousKeyCallback)
                g_previousKeyCallback(window, key, scancode, action, mods);
        }

        // GLFW's gamepad buttons → raylib's GamepadButton (as raylib's own PollInputEvents)
        int raylibButton(int glfwButton)
        {
            switch (glfwButton)
            {
            case GLFW_GAMEPAD_BUTTON_Y: return GAMEPAD_BUTTON_RIGHT_FACE_UP;
            case GLFW_GAMEPAD_BUTTON_B: return GAMEPAD_BUTTON_RIGHT_FACE_RIGHT;
            case GLFW_GAMEPAD_BUTTON_A: return GAMEPAD_BUTTON_RIGHT_FACE_DOWN;
            case GLFW_GAMEPAD_BUTTON_X: return GAMEPAD_BUTTON_RIGHT_FACE_LEFT;
            case GLFW_GAMEPAD_BUTTON_LEFT_BUMPER: return GAMEPAD_BUTTON_LEFT_TRIGGER_1;
            case GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER: return GAMEPAD_BUTTON_RIGHT_TRIGGER_1;
            case GLFW_GAMEPAD_BUTTON_BACK: return GAMEPAD_BUTTON_MIDDLE_LEFT;
            case GLFW_GAMEPAD_BUTTON_GUIDE: return GAMEPAD_BUTTON_MIDDLE;
            case GLFW_GAMEPAD_BUTTON_START: return GAMEPAD_BUTTON_MIDDLE_RIGHT;
            case GLFW_GAMEPAD_BUTTON_DPAD_UP: return GAMEPAD_BUTTON_LEFT_FACE_UP;
            case GLFW_GAMEPAD_BUTTON_DPAD_RIGHT: return GAMEPAD_BUTTON_LEFT_FACE_RIGHT;
            case GLFW_GAMEPAD_BUTTON_DPAD_DOWN: return GAMEPAD_BUTTON_LEFT_FACE_DOWN;
            case GLFW_GAMEPAD_BUTTON_DPAD_LEFT: return GAMEPAD_BUTTON_LEFT_FACE_LEFT;
            case GLFW_GAMEPAD_BUTTON_LEFT_THUMB: return GAMEPAD_BUTTON_LEFT_THUMB;
            case GLFW_GAMEPAD_BUTTON_RIGHT_THUMB: return GAMEPAD_BUTTON_RIGHT_THUMB;
            default: return -1;
            }
        }
#endif

        bool ours() { return g_window.load(std::memory_order_acquire) != nullptr; }
    }

    void useWindow(GLFWwindow *window)
    {
#if defined(PS2X_HAS_GLFW)
        if (window)
            g_previousKeyCallback = glfwSetKeyCallback(window, onKey);
        g_window.store(window, std::memory_order_release);
        poll();
#else
        (void)window;
#endif
    }

    GLFWwindow *window() { return g_window.load(std::memory_order_acquire); }

    void poll()
    {
#if defined(PS2X_HAS_GLFW)
        if (!ours())
            return;
        std::array<Gamepad, kGamepads> pads;
        for (int i = 0; i < kGamepads; ++i)
        {
            Gamepad &pad = pads[static_cast<size_t>(i)];
            pad.ready = glfwJoystickPresent(i) == GLFW_TRUE;
            if (!pad.ready)
                continue;
            if (const char *name = glfwGetJoystickName(i))
                pad.name = name;
            GLFWgamepadstate state{};
            if (glfwGetGamepadState(i, &state) == GLFW_TRUE)
            {
                for (int k = 0; k <= GLFW_GAMEPAD_BUTTON_LAST; ++k)
                {
                    const int button = raylibButton(k);
                    if (button >= 0 && button < kButtons)
                        pad.buttons[static_cast<size_t>(button)] = state.buttons[k] == GLFW_PRESS;
                }
                for (int k = 0; k < kAxes; ++k)
                    pad.axes[static_cast<size_t>(k)] = state.axes[k];
                pad.buttons[GAMEPAD_BUTTON_LEFT_TRIGGER_2] = pad.axes[GAMEPAD_AXIS_LEFT_TRIGGER] > 0.1f;
                pad.buttons[GAMEPAD_BUTTON_RIGHT_TRIGGER_2] = pad.axes[GAMEPAD_AXIS_RIGHT_TRIGGER] > 0.1f;
            }
        }
        std::lock_guard<std::mutex> lock(g_padMutex);
        g_pads = std::move(pads);
#endif
    }

    bool windowReady() { return ours() || IsWindowReady(); }

    bool keyDown(int key)
    {
        if (!ours())
            return IsKeyDown(key);
        return key >= 0 && key < static_cast<int>(g_keys.size()) && g_keys[static_cast<size_t>(key)].load(std::memory_order_relaxed);
    }

    bool gamepadAvailable(int gamepad)
    {
        if (!ours())
            return IsGamepadAvailable(gamepad);
        if (gamepad < 0 || gamepad >= kGamepads)
            return false;
        std::lock_guard<std::mutex> lock(g_padMutex);
        return g_pads[static_cast<size_t>(gamepad)].ready;
    }

    const char *gamepadName(int gamepad)
    {
        if (!ours())
            return GetGamepadName(gamepad);
        if (gamepad < 0 || gamepad >= kGamepads)
            return nullptr;
        // a copy per controller on the asking thread: the snapshot may change under it
        thread_local std::array<std::string, kGamepads> copies;
        std::lock_guard<std::mutex> lock(g_padMutex);
        const Gamepad &pad = g_pads[static_cast<size_t>(gamepad)];
        if (!pad.ready)
            return nullptr;
        copies[static_cast<size_t>(gamepad)] = pad.name;
        return copies[static_cast<size_t>(gamepad)].c_str();
    }

    bool gamepadButtonDown(int gamepad, int button)
    {
        if (!ours())
            return IsGamepadButtonDown(gamepad, button);
        if (gamepad < 0 || gamepad >= kGamepads || button < 0 || button >= kButtons)
            return false;
        std::lock_guard<std::mutex> lock(g_padMutex);
        const Gamepad &pad = g_pads[static_cast<size_t>(gamepad)];
        return pad.ready && pad.buttons[static_cast<size_t>(button)];
    }

    float gamepadAxis(int gamepad, int axis)
    {
        if (!ours())
            return GetGamepadAxisMovement(gamepad, axis);
        // as raylib: the triggers rest at -1, the sticks at 0
        float value = (axis == GAMEPAD_AXIS_LEFT_TRIGGER || axis == GAMEPAD_AXIS_RIGHT_TRIGGER) ? -1.0f : 0.0f;
        if (gamepad < 0 || gamepad >= kGamepads || axis < 0 || axis >= kAxes)
            return value;
        std::lock_guard<std::mutex> lock(g_padMutex);
        const Gamepad &pad = g_pads[static_cast<size_t>(gamepad)];
        if (!pad.ready)
            return value;
        const float state = pad.axes[static_cast<size_t>(axis)];
        const float movement = value < 0.0f ? state : std::fabs(state);
        return movement > value ? state : value;
    }
}
