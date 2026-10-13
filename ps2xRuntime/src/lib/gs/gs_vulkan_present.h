#pragma once

// Shows the graphics card's pictures in a window through Vulkan (vulkanPresentWanted): the window's own device,
// on the same graphics card as paraLLEl-GS's, draws the picture the drawing thread shares (gs_vulkan_shared.h) into
// Granite's swapchain - a DXGI flip-model swapchain on Windows when the card allows it (what Xbox Game Bar, overlays
// and G-Sync / FreeSync work with), else a Vulkan surface swapchain. The window must have no OpenGL context.

#include <cstdint>
#include <memory>
#include <string>

struct GLFWwindow;
struct GSGpuFrame;
struct ImDrawData;
class GSGpuRenderer;

// The game's window shown through Vulkan: the default with the graphics card renderer (VENDETTA_GS=gpu, the default);
// VENDETTA_PRESENT=opengl = raylib's window as before, =vulkan = asked for in any mode. The CPU renderer and
// VENDETTA_GS=both (the two compared) keep raylib's window.
bool vulkanPresentWanted();

// The window couldn't be shown through Vulkan (raylib's window instead): the drawing thread reads its pictures back
// again rather than sharing them. Before the first picture is shown.
void stopSharedPictures();

// The debug panel's frame (Dear ImGui's draw data), drawn over the picture at the next present; null = no panel. The
// main thread.
void setPresentUi(ImDrawData *data);

// The window's client area painted black with Windows' own drawing (a window without OpenGL shows nothing of its own
// until its first Vulkan picture): for a black start, and the black cover before a fullscreen mode change.
void paintWindowBlack(GLFWwindow *window);

class VulkanPresenter
{
public:
    VulkanPresenter();
    ~VulkanPresenter();

    // Made on the window's thread, on the graphics card with this name (the drawing thread's).
    bool init(GLFWwindow *window, const std::string &deviceName, bool vsync, std::string &whyNot);
    bool ready() const;
    // The frame's shared picture drawn into dst (pixels, in the window) on black, then presented; nearest = blocky
    // scaling (else smooth). The renderer is told which shared image is in use while it is drawn.
    void present(const GSGpuFrame &frame, GSGpuRenderer &renderer, int dstX, int dstY, int dstWidth, int dstHeight,
                 bool nearest);
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
