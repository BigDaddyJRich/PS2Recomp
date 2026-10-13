#include "runtime/gs/gs_gpu.h"

// Built without paraLLEl-GS (no PS2X_PARALLEL_GS_DIR): the CPU renderer only.
std::unique_ptr<GSGpuRenderer> makeGpuRenderer(const GSGpuOptions &, std::string &whyNot)
{
    whyNot = "built without paraLLEl-GS (configure with -DPS2X_PARALLEL_GS_DIR=<its source tree>)";
    return nullptr;
}
