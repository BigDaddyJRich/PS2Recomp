#pragma once

struct GLFWwindow;

// The game's window in fullscreen, when it loses the front. It is minimised (the screen gets its own mode back) only
// when the player switched away: Alt-Tab, the Windows key, a click elsewhere. When another program takes the front
// by itself (Microsoft Copilot's Quick View does at every screen mode change), the game stays fullscreen on screen:
// one click or Alt-Tab brings it back, with no new mode change. Controllers keep working meanwhile; the keyboard goes
// to the program in front until then.
namespace ps2_host_window
{
    // Replaces GLFW's own minimise-on-focus-loss for this window (and keeps any focus callback already set).
    void minimiseOnlyWhenThePlayerSwitchesAway(GLFWwindow *window);
}
