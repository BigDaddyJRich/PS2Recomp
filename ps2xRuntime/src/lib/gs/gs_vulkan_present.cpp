// The window's own Vulkan device: the drawing thread's shared pictures drawn into Granite's swapchain (see
// gs_vulkan_present.h).

#include "gs_vulkan_present.h"
#include "gs_vulkan_shared.h"
#include "runtime/gs/gs_gpu.h"

#include "context.hpp"
#include "device.hpp"
#include "wsi.hpp"
#include "gs_interface.hpp"
#include "shaders/slangmosh.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <windows.h>

#if defined(PS2X_PRESENT_UI)
#include "imgui.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
    // Granite's window interface over a GLFW window: a Win32 surface, the window's size, its handle (for DXGI). The
    // main loop polls the window's events itself.
    class GlfwPlatform final : public Vulkan::WSIPlatform
    {
    public:
        explicit GlfwPlatform(GLFWwindow *window) : m_window(window) {}

        VkSurfaceKHR create_surface(VkInstance instance, VkPhysicalDevice) override
        {
            VkWin32SurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
            info.hinstance = GetModuleHandleW(nullptr);
            info.hwnd = glfwGetWin32Window(m_window);
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            if (vkCreateWin32SurfaceKHR(instance, &info, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            return surface;
        }
        std::vector<const char *> get_instance_extensions() override { return {"VK_KHR_surface", "VK_KHR_win32_surface"}; }
        uint32_t get_surface_width() override { return size().first; }
        uint32_t get_surface_height() override { return size().second; }
        bool alive(Vulkan::WSI &) override { return true; }
        void poll_input() override {}
        void poll_input_async(Granite::InputTrackerHandler *) override {}
        uintptr_t get_native_window() override { return reinterpret_cast<uintptr_t>(glfwGetWin32Window(m_window)); }

        // a new size: the swapchain is made again at the next frame
        void checkResize()
        {
            const auto [width, height] = size();
            if (width != current_swapchain_width || height != current_swapchain_height)
                resize = true;
        }

    private:
        std::pair<uint32_t, uint32_t> size() const
        {
            int width = 0, height = 0;
            glfwGetFramebufferSize(m_window, &width, &height);
            return {static_cast<uint32_t>(std::max(width, 1)), static_cast<uint32_t>(std::max(height, 1))};
        }

        GLFWwindow *m_window;
    };

    // The graphics card with this name, and its UUID (each shared picture names its card's: they must match)
    VkPhysicalDevice findByName(VkInstance instance, const std::string &name, uint8_t (&uuid)[16])
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> gpus(count);
        vkEnumeratePhysicalDevices(instance, &count, gpus.data());
        for (VkPhysicalDevice gpu : gpus)
        {
            VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            props.pNext = &ids;
            vkGetPhysicalDeviceProperties2(gpu, &props);
            if (name == props.properties.deviceName)
            {
                std::memcpy(uuid, ids.deviceUUID, sizeof(uuid));
                return gpu;
            }
        }
        return VK_NULL_HANDLE;
    }

    // Our own handle to import (Granite closes what it imports; the drawing thread keeps its own).
    void *duplicate(void *handle)
    {
        HANDLE copy = nullptr;
        if (!handle || !DuplicateHandle(GetCurrentProcess(), handle, GetCurrentProcess(), &copy, 0, FALSE, DUPLICATE_SAME_ACCESS))
            return nullptr;
        return copy;
    }
}

namespace
{
    ImDrawData *g_ui = nullptr;
}

void setPresentUi(ImDrawData *data) { g_ui = data; }

void paintWindowBlack(GLFWwindow *window)
{
    HWND hwnd = glfwGetWin32Window(window);
    if (!hwnd)
        return;
    RECT rect{};
    GetClientRect(hwnd, &rect);
    if (HDC dc = GetDC(hwnd))
    {
        FillRect(dc, &rect, static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
        ReleaseDC(hwnd, dc);
    }
    GdiFlush();
}

struct VulkanPresenter::Impl
{
    std::unique_ptr<GlfwPlatform> platform;
    Vulkan::WSI wsi;
    Vulkan::Program *program = nullptr;
    Vulkan::ImageHandle images[4];
    uint64_t generation = 0;
    Vulkan::Semaphore timeline;
    bool ready = false;
    uint8_t uuid[16] = {};
    bool uuidReported = false;
    // the debug panel: paraLLEl-GS's UI shaders (textured, no sRGB conversion), Dear ImGui's textures by their id - 1
    Vulkan::Program *uiProgram = nullptr;
    std::vector<Vulkan::ImageHandle> uiTextures;

    void updateUiTextures();
    void drawUi(Vulkan::CommandBuffer &cmd, uint32_t width, uint32_t height);
};

VulkanPresenter::VulkanPresenter() : m_impl(std::make_unique<Impl>()) {}

VulkanPresenter::~VulkanPresenter() { shutdown(); }

bool VulkanPresenter::init(GLFWwindow *window, const std::string &deviceName, bool vsync, std::string &whyNot)
{
    Impl &m = *m_impl;
    m.platform = std::make_unique<GlfwPlatform>(window);
    if (!Vulkan::Context::init_loader(nullptr))
        return whyNot = "no Vulkan loader", false;
    auto context = Util::make_handle<Vulkan::Context>();
    context->set_num_thread_indices(1);
    const auto extensions = m.platform->get_instance_extensions();
    if (!context->init_instance(extensions.data(), static_cast<uint32_t>(extensions.size()), 0))
        return whyNot = "no Vulkan instance with a Win32 surface", false;
    const VkPhysicalDevice gpu = findByName(context->get_instance(), deviceName, m.uuid);
    if (gpu == VK_NULL_HANDLE)
        return whyNot = "the drawing thread's graphics card (" + deviceName + ") isn't listed", false;
    const char *deviceExtensions[] = {"VK_KHR_swapchain"};
    if (!context->init_device(gpu, VK_NULL_HANDLE, deviceExtensions, 1, 0))
        return whyNot = "no Vulkan device for the window", false;
    if (!context->get_enabled_device_features().supports_external)
        return whyNot = "the window's device can't import shared images", false;

    m.wsi.set_platform(m.platform.get());
    m.wsi.set_backbuffer_format(Vulkan::BackbufferFormat::UNORM);
    m.wsi.set_present_mode(vsync ? Vulkan::PresentMode::SyncToVBlank : Vulkan::PresentMode::UnlockedMaybeTear);
    if (!m.wsi.init_from_existing_context(context) || !m.wsi.init_device())
        return whyNot = "Granite's WSI didn't start", false;
    if (!m.wsi.init_surface_swapchain())
        return whyNot = "no swapchain for the window", false;

    Vulkan::Device &device = m.wsi.get_device();
    Vulkan::ResourceLayout layout;
    ParallelGS::Shaders<> suite(device, layout, 0);
    m.program = device.request_program(suite.quad, suite.blit_circuit);
    if (!m.program)
        return whyNot = "no program for the picture", false;
    m.uiProgram = device.request_program(suite.ui_vert, suite.ui_frag[1][0]);
    m.ready = true;
    std::cerr << "[present] Vulkan on \"" << context->get_gpu_props().deviceName << "\"" << std::endl;
    return true;
}

bool VulkanPresenter::ready() const { return m_impl && m_impl->ready; }

void VulkanPresenter::present(const GSGpuFrame &frame, GSGpuRenderer &renderer, int dstX, int dstY, int dstWidth,
                              int dstHeight, bool nearest)
{
    Impl &m = *m_impl;
    if (!m.ready || !frame.shared || frame.sharedImage < 0 || frame.sharedImage > 3)
        return;
    if (std::memcmp(frame.sharedDeviceUuid, m.uuid, sizeof(m.uuid)) != 0)
    {
        if (!m.uuidReported)
            std::cerr << "[present] the pictures come from another graphics card than the window's: not shown" << std::endl;
        m.uuidReported = true;
        return;
    }
    Vulkan::Device &device = m.wsi.get_device();

    // the shared images (again at a new size) and the timeline (once)
    if (frame.sharedGeneration != m.generation)
    {
        for (int i = 0; i < 4; ++i)
        {
            Vulkan::ImageCreateInfo info = sharedImageInfo(frame.width, frame.height, duplicate(frame.sharedHandles[i]));
            m.images[i] = info.external.handle ? device.create_image(info) : Vulkan::ImageHandle{};
            if (!m.images[i])
            {
                std::cerr << "[present] couldn't import shared picture " << i << std::endl;
                m.generation = 0;
                return;
            }
        }
        m.generation = frame.sharedGeneration;
    }
    if (!m.timeline)
    {
        m.timeline = device.request_semaphore_external(VK_SEMAPHORE_TYPE_TIMELINE,
                                                       VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT);
        Vulkan::ExternalHandle handle;
        handle.handle = duplicate(frame.sharedTimeline);
        handle.semaphore_handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        if (!m.timeline || !handle.handle || !m.timeline->import_from_handle(handle))
        {
            std::cerr << "[present] couldn't import the shared timeline" << std::endl;
            m.timeline.reset();
            return;
        }
    }

    renderer.setSharedReading(frame.sharedImage);
    m.platform->checkResize();
    if (!m.wsi.begin_frame())
    {
        renderer.setSharedReading(-1);
        return;
    }
    // drawn once the drawing thread's copy is there
    device.add_wait_semaphore(Vulkan::CommandBuffer::Type::Generic,
                              device.request_timeline_semaphore_as_binary(*m.timeline, frame.seq),
                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, true);
    Vulkan::CommandBufferHandle cmd = device.request_command_buffer();
    const Vulkan::Image &image = *m.images[frame.sharedImage];
    // taken over from the other device (it released it to the "external" queue family, in the sampling layout)
    VkImageMemoryBarrier2 acquire{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    acquire.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    acquire.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    acquire.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    acquire.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    acquire.dstQueueFamilyIndex = device.get_queue_info().family_indices[Vulkan::QUEUE_INDEX_GRAPHICS];
    acquire.image = image.get_image();
    acquire.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &acquire;
    cmd->barrier(dep);

    m.updateUiTextures();
    Vulkan::RenderPassInfo pass = device.get_swapchain_render_pass(Vulkan::SwapchainRenderPass::ColorOnly);
    pass.clear_attachments = 1u;
    pass.store_attachments = 1u;
    pass.clear_color[0] = {};
    pass.clear_color[0].float32[3] = 1.0f;
    cmd->begin_render_pass(pass);
    if (dstWidth > 0 && dstHeight > 0)
    {
        cmd->set_program(m.program);
        cmd->set_opaque_state();
        cmd->set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        cmd->set_texture(0, 0, image.get_view(), nearest ? Vulkan::StockSampler::NearestClamp : Vulkan::StockSampler::LinearClamp);
        cmd->set_viewport({static_cast<float>(dstX), static_cast<float>(dstY), static_cast<float>(dstWidth),
                           static_cast<float>(dstHeight), 0.0f, 1.0f});
        const int x0 = std::max(dstX, 0), y0 = std::max(dstY, 0);
        cmd->set_scissor({{x0, y0}, {static_cast<uint32_t>(dstX + dstWidth - x0), static_cast<uint32_t>(dstY + dstHeight - y0)}});
        cmd->draw(3);
    }
    m.drawUi(*cmd, device.get_swapchain_view().get_image().get_width(), device.get_swapchain_view().get_image().get_height());
    cmd->end_render_pass();
    Vulkan::Fence fence;
    device.submit(cmd, &fence);
    m.wsi.end_frame();
    // the drawing thread may use the image again once this frame's drawing is done
    fence->wait();
    renderer.setSharedReading(-1);
}

// Dear ImGui's textures (its font atlas and any others), created, updated (made again whole) and destroyed as it asks.
void VulkanPresenter::Impl::updateUiTextures()
{
#if defined(PS2X_PRESENT_UI)
    if (!g_ui || !g_ui->Textures)
        return;
    Vulkan::Device &device = wsi.get_device();
    for (ImTextureData *tex : *g_ui->Textures)
    {
        if (tex->Status == ImTextureStatus_OK)
            continue;
        if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0)
        {
            const size_t slot = static_cast<size_t>(tex->GetTexID());
            if (slot > 0 && slot <= uiTextures.size())
                uiTextures[slot - 1].reset();
            tex->SetTexID(ImTextureID_Invalid);
            tex->SetStatus(ImTextureStatus_Destroyed);
            continue;
        }
        if (tex->Status != ImTextureStatus_WantCreate && tex->Status != ImTextureStatus_WantUpdates)
            continue;
        // RGBA, or white with the atlas's alpha
        std::vector<uint8_t> rgba;
        const uint8_t *pixels = static_cast<const uint8_t *>(tex->GetPixels());
        if (tex->Format == ImTextureFormat_Alpha8)
        {
            rgba.resize(static_cast<size_t>(tex->Width) * tex->Height * 4u);
            for (size_t i = 0; i < static_cast<size_t>(tex->Width) * tex->Height; ++i)
            {
                rgba[i * 4u + 0u] = rgba[i * 4u + 1u] = rgba[i * 4u + 2u] = 255u;
                rgba[i * 4u + 3u] = pixels[i];
            }
            pixels = rgba.data();
        }
        Vulkan::ImageCreateInfo info = Vulkan::ImageCreateInfo::immutable_2d_image(
            static_cast<unsigned>(tex->Width), static_cast<unsigned>(tex->Height), VK_FORMAT_R8G8B8A8_UNORM);
        Vulkan::ImageInitialData initial{pixels, 0, 0};
        Vulkan::ImageHandle image = device.create_image(info, &initial);
        size_t slot = static_cast<size_t>(tex->GetTexID());
        if (slot == 0 || slot > uiTextures.size())
        {
            uiTextures.push_back(image);
            slot = uiTextures.size();
        }
        else
            uiTextures[slot - 1] = image;
        tex->SetTexID(static_cast<ImTextureID>(slot));
        tex->SetStatus(ImTextureStatus_OK);
    }
#endif
}

// The panel over the picture: each list's vertices and indices, each command clipped to its rectangle (in pixels),
// alpha blended.
void VulkanPresenter::Impl::drawUi(Vulkan::CommandBuffer &cmd, uint32_t width, uint32_t height)
{
#if defined(PS2X_PRESENT_UI)
    ImDrawData *data = g_ui;
    g_ui = nullptr;
    if (!data || !data->Valid || !uiProgram || data->TotalVtxCount == 0 || data->DisplaySize.x <= 0.0f || data->DisplaySize.y <= 0.0f)
        return;
    cmd.set_program(uiProgram);
    cmd.set_opaque_state();
    cmd.set_cull_mode(VK_CULL_MODE_NONE);
    cmd.set_blend_enable(true);
    cmd.set_blend_factors(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);
    cmd.set_blend_op(VK_BLEND_OP_ADD);
    cmd.set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    cmd.set_viewport({0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f});
    const float invResolution[2] = {1.0f / data->DisplaySize.x, 1.0f / data->DisplaySize.y};
    cmd.push_constants(invResolution, 0, sizeof(invResolution));
    const ImVec2 origin = data->DisplayPos;
    const ImVec2 scale = data->FramebufferScale;
    for (const ImDrawList *list : data->CmdLists)
    {
        void *vertices = cmd.allocate_vertex_data(0, list->VtxBuffer.Size * sizeof(ImDrawVert), sizeof(ImDrawVert));
        std::memcpy(vertices, list->VtxBuffer.Data, list->VtxBuffer.Size * sizeof(ImDrawVert));
        void *indices = cmd.allocate_index_data(list->IdxBuffer.Size * sizeof(ImDrawIdx),
                                                sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        std::memcpy(indices, list->IdxBuffer.Data, list->IdxBuffer.Size * sizeof(ImDrawIdx));
        cmd.set_vertex_attrib(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, pos));
        cmd.set_vertex_attrib(1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, uv));
        cmd.set_vertex_attrib(2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(ImDrawVert, col));
        for (const ImDrawCmd &draw : list->CmdBuffer)
        {
            if (draw.UserCallback)
                continue;
            const float x0 = std::max(0.0f, (draw.ClipRect.x - origin.x) * scale.x);
            const float y0 = std::max(0.0f, (draw.ClipRect.y - origin.y) * scale.y);
            const float x1 = std::min(static_cast<float>(width), (draw.ClipRect.z - origin.x) * scale.x);
            const float y1 = std::min(static_cast<float>(height), (draw.ClipRect.w - origin.y) * scale.y);
            if (x1 <= x0 || y1 <= y0)
                continue;
            const size_t slot = static_cast<size_t>(draw.GetTexID());
            if (slot == 0 || slot > uiTextures.size() || !uiTextures[slot - 1])
                continue;
            cmd.set_texture(0, 0, uiTextures[slot - 1]->get_view(), Vulkan::StockSampler::LinearClamp);
            cmd.set_scissor({{static_cast<int32_t>(x0), static_cast<int32_t>(y0)},
                             {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}});
            cmd.draw_indexed(draw.ElemCount, 1, draw.IdxOffset, static_cast<int32_t>(draw.VtxOffset), 0);
        }
    }
#else
    (void)cmd;
    (void)width;
    (void)height;
#endif
}

void VulkanPresenter::shutdown()
{
    if (!m_impl || !m_impl->ready)
        return;
    Impl &m = *m_impl;
    m.wsi.get_device().wait_idle();
    for (auto &image : m.images)
        image.reset();
    m.uiTextures.clear();
    m.timeline.reset();
    m.ready = false;
}
