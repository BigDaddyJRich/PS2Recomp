#include "runtime/gs/gs_gpu.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace
{
    bool envFlag(const char *name, bool fallback)
    {
        const char *value = std::getenv(name);
        if (!value || !*value)
            return fallback;
        return std::strcmp(value, "0") != 0;
    }
}

GSRendererMode gsRendererModeFromEnvironment()
{
    const char *value = std::getenv("VENDETTA_GS");
    if (!value || !*value || std::strcmp(value, "gpu") == 0)
        return GSRendererMode::Gpu;
    if (std::strcmp(value, "cpu") == 0)
        return GSRendererMode::Cpu;
    if (std::strcmp(value, "both") == 0)
        return GSRendererMode::Both;
    std::cerr << "[gpu-gs] VENDETTA_GS=" << value << " not understood (cpu, gpu or both); drawing on the CPU" << std::endl;
    return GSRendererMode::Cpu;
}

const char *gsRendererModeName(GSRendererMode mode)
{
    switch (mode)
    {
    case GSRendererMode::Gpu:
        return "gpu";
    case GSRendererMode::Both:
        return "both";
    default:
        return "cpu";
    }
}

GSPresentFilter gsPresentFilterFromEnvironment()
{
    const char *value = std::getenv("VENDETTA_FILTER");
    if (value && std::strcmp(value, "nearest") == 0)
        return GSPresentFilter::Nearest;
    if (value && std::strcmp(value, "bilinear") == 0)
        return GSPresentFilter::Bilinear;
    if (value && *value && std::strcmp(value, "fsr") != 0)
        std::cerr << "[present] VENDETTA_FILTER=" << value << " not understood (fsr, bilinear or nearest)" << std::endl;
    return GSPresentFilter::Fsr;
}

GSGpuOptions GSGpuOptions::fromEnvironment()
{
    GSGpuOptions options;
    if (const char *ssaa = std::getenv("VENDETTA_SSAA"); ssaa && *ssaa)
    {
        const unsigned long rate = std::strtoul(ssaa, nullptr, 10);
        if (rate == 1 || rate == 2 || rate == 4 || rate == 8 || rate == 16)
            options.superSampling = static_cast<uint32_t>(rate);
        else
            std::cerr << "[gpu-gs] VENDETTA_SSAA=" << ssaa << " not one of 1 2 4 8 16; using "
                      << options.superSampling << std::endl;
    }
    options.orderedGrid = envFlag("VENDETTA_SSAA_ORDERED", true);
    options.superSampledTextures = envFlag("VENDETTA_SSAA_TEXTURES", false);
    options.highResolutionScanout = envFlag("VENDETTA_HIRES", true);
    options.fsr = gsPresentFilterFromEnvironment() == GSPresentFilter::Fsr;
    if (const char *sharpen = std::getenv("VENDETTA_SHARPEN"); sharpen && *sharpen)
        options.sharpness = std::clamp(static_cast<float>(std::atof(sharpen)), 0.0f, 2.0f);
    options.libraryLog = envFlag("VENDETTA_GPU_LOG", false);
    options.sync = envFlag("VENDETTA_GPU_SYNC", false);
    if (const char *ring = std::getenv("VENDETTA_GPU_RING_MB"); ring && *ring)
        options.ringBytes = static_cast<size_t>(std::max(1ul, std::strtoul(ring, nullptr, 10))) << 20u;
    if (const char *gpu = std::getenv("VENDETTA_GPU"))
        options.gpuSelector = gpu;
    return options;
}
