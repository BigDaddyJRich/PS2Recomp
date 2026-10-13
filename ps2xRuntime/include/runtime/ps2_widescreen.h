#pragma once
// Widescreen for games drawn for a 4:3 TV: the game's view is widened by game-specific code (its projection), which
// marks each frame it draws that way; the window then shows those frames at the target shape and everything else
// (menus, movies) at 4:3. The game's frame buffer keeps its size: a wide frame is drawn squeezed into it and stretched
// back by the window.
//
// VENDETTA_WIDESCREEN=<w>:<h> (e.g. 16:9, 21:9, 16:10) or 1 (= 16:9); unset / 0 = off.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace ps2_widescreen
{
    // The shape (width / height) wide frames are shown at; 0 = widescreen off.
    inline float targetAspect()
    {
        static const float aspect = []
        {
            const char *value = std::getenv("VENDETTA_WIDESCREEN");
            if (!value || !*value || (value[0] == '0' && value[1] == '\0'))
                return 0.0f;
            float w = 0.0f, h = 0.0f;
            if (std::sscanf(value, "%f:%f", &w, &h) == 2 && w > 0.0f && h > 0.0f)
                return w / h;
            return 16.0f / 9.0f;
        }();
        return aspect;
    }

    namespace detail
    {
        inline std::atomic<int64_t> &lastWideFrame()
        {
            static std::atomic<int64_t> value{INT64_MIN / 2};
            return value;
        }
        inline int64_t nowMs()
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }
    }

    // Game code: the frame being drawn is wide.
    inline void markWideFrame() { detail::lastWideFrame().store(detail::nowMs(), std::memory_order_relaxed); }

    // A wide frame was drawn in the last `holdMs` milliseconds (the window shows the target shape).
    inline bool wideNow(int64_t holdMs = 250)
    {
        return targetAspect() > 0.0f && detail::nowMs() - detail::lastWideFrame().load(std::memory_order_relaxed) <= holdMs;
    }
}
