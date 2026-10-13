#pragma once

// The PC's keyboard and controllers for the pads (ps2_pad.cpp), with raylib's numbering (KEY_*, GAMEPAD_BUTTON_*,
// GAMEPAD_AXIS_*: raylib.h). Read from raylib's window as before, or - once a window is given (the Vulkan window,
// the game's window by default) - straight from GLFW the way raylib does it: keys from that window, controllers 0-3 as
// GLFW's joysticks with its Xbox-style gamepad mapping, L2 / R2 pressed past 0.1. GLFW's controller calls belong on the
// main thread: poll() there takes a snapshot each window frame; the pads read the snapshot from the game thread.

struct GLFWwindow;

namespace host_input
{
    void useWindow(GLFWwindow *window); // the main thread
    GLFWwindow *window();               // the window given (null: raylib's)
    void poll();                        // the main thread, each window frame (with a window given)

    bool windowReady();
    bool keyDown(int key);
    bool gamepadAvailable(int gamepad);
    const char *gamepadName(int gamepad); // valid until the next poll
    bool gamepadButtonDown(int gamepad, int button);
    float gamepadAxis(int gamepad, int axis);
}
