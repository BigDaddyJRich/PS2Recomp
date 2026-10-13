#pragma once

// The pictures the graphics card thread (paraLLEl-GS's device) shares with the window's own Vulkan device
// (shown through Vulkan): both sides describe the images exactly alike, as importing one requires.

#include "image.hpp"

#include <cstdint>

// An R8G8B8A8 picture, copied in (transfer) on one device and sampled on the other; with import = a Windows handle
// from the other device, that image's memory is used.
inline Vulkan::ImageCreateInfo sharedImageInfo(uint32_t width, uint32_t height, void *import = nullptr)
{
    Vulkan::ImageCreateInfo info = Vulkan::ImageCreateInfo::immutable_2d_image(width, height, VK_FORMAT_R8G8B8A8_UNORM);
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.misc = Vulkan::IMAGE_MISC_EXTERNAL_MEMORY_BIT;
    info.external.memory_handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    info.external.handle = import;
    return info;
}
