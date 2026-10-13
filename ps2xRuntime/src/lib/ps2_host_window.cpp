#include "runtime/ps2_host_window.h"

#if defined(PS2X_HAS_GLFW)
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <iostream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace ps2_host_window
{
    namespace
    {
        GLFWwindowfocusfun g_previous = nullptr;

        // Did the player cause the focus loss? A key Windows switches with (Alt, Tab, Ctrl, the Windows keys) or a mouse
        // button is down, or the player's last keyboard / mouse input was just now (the Windows key opens Start when it
        // is let go). A program taking the front by itself comes with neither.
        bool playerSwitchedAway()
        {
#if defined(_WIN32)
            for (int key : {VK_MENU, VK_TAB, VK_CONTROL, VK_LWIN, VK_RWIN, VK_LBUTTON, VK_RBUTTON, VK_MBUTTON})
                if (GetAsyncKeyState(key) & 0x8000)
                    return true;
            LASTINPUTINFO last{};
            last.cbSize = sizeof(last);
            if (GetLastInputInfo(&last) && GetTickCount() - last.dwTime < 1000u)
                return true;
            return false;
#else
            return true;
#endif
        }

        void onFocus(GLFWwindow *window, int focused)
        {
            if (g_previous)
                g_previous(window, focused);
            if (focused || !glfwGetWindowMonitor(window))
                return; // only a fullscreen window losing the front
            if (playerSwitchedAway())
            {
                glfwIconifyWindow(window); // the screen gets its own mode back; clicking it or Alt-Tab restores both
                return;
            }
            std::cerr << "[window] another program took the front by itself: the game stays fullscreen (a click or "
                         "Alt-Tab brings it back)"
                      << std::endl;
        }
    }

    void minimiseOnlyWhenThePlayerSwitchesAway(GLFWwindow *window)
    {
        if (!window)
            return;
        glfwSetWindowAttrib(window, GLFW_AUTO_ICONIFY, GLFW_FALSE);
        GLFWwindowfocusfun previous = glfwSetWindowFocusCallback(window, onFocus);
        if (previous != onFocus)
            g_previous = previous;
    }
}
#endif
