#include <utility>
#include <cstdlib>
#include "runtime/ps2_host_profile.h"
#include <vector>
#include <string>
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <cmath>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace GSInternal;

// The per-pixel helpers are forced inline: the drawing functions are large, and the compiler's own inlining left these
// small helpers as calls inside them.
#if defined(_MSC_VER)
#define GS_FORCEINLINE __forceinline
#else
#define GS_FORCEINLINE inline __attribute__((always_inline))
#endif

namespace
{
    GS_FORCEINLINE float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    GS_FORCEINLINE u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    GS_FORCEINLINE u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    GS_FORCEINLINE u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    GS_FORCEINLINE uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    GS_FORCEINLINE int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    GS_FORCEINLINE bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    GS_FORCEINLINE TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    GS_FORCEINLINE uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    GS_FORCEINLINE bool isFourBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    GS_FORCEINLINE bool isEightBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H;
    }

    // = std::lround for |value| < 2^24 (here a blend of 0..255 values), without the library call: truncate, then step
    // away from zero when the dropped fraction (value - truncated, exact) is at least a half
    GS_FORCEINLINE int roundHalfAwayFromZero(float value)
    {
        const int truncated = static_cast<int>(value);
        const float fraction = value - static_cast<float>(truncated);
        return fraction >= 0.5f ? truncated + 1 : (fraction <= -0.5f ? truncated - 1 : truncated);
    }

    GS_FORCEINLINE uint8_t lerpChannel(uint8_t c00, uint8_t c10, uint8_t c01, uint8_t c11, float fx, float fy)
    {
        const float top = static_cast<float>(c00) + (static_cast<float>(c10) - static_cast<float>(c00)) * fx;
        const float bottom = static_cast<float>(c01) + (static_cast<float>(c11) - static_cast<float>(c01)) * fx;
        return clampU8(roundHalfAwayFromZero(top + (bottom - top) * fy));
    }

    // lerpChannel on each 8-bit channel of four RGBA texels: with SSE4.1 the four channels in four lanes (the same
    // float operations per channel in the same order, the same rounding and clamp)
    GS_FORCEINLINE uint32_t bilinearBlend(uint32_t c00, uint32_t c10, uint32_t c01, uint32_t c11, float fx, float fy)
    {
#if defined(__AVX2__)
        const auto channels = [](uint32_t texel)
        { return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(texel)))); };
        const __m128 f00 = channels(c00), f10 = channels(c10), f01 = channels(c01), f11 = channels(c11);
        const __m128 wx = _mm_set1_ps(fx), wy = _mm_set1_ps(fy);
        const __m128 top = _mm_add_ps(f00, _mm_mul_ps(_mm_sub_ps(f10, f00), wx));
        const __m128 bottom = _mm_add_ps(f01, _mm_mul_ps(_mm_sub_ps(f11, f01), wx));
        const __m128 value = _mm_add_ps(top, _mm_mul_ps(_mm_sub_ps(bottom, top), wy));
        // roundHalfAwayFromZero per lane (a compare is all ones = -1 where it holds)
        const __m128i truncated = _mm_cvttps_epi32(value);
        const __m128 fraction = _mm_sub_ps(value, _mm_cvtepi32_ps(truncated));
        const __m128i up = _mm_castps_si128(_mm_cmpge_ps(fraction, _mm_set1_ps(0.5f)));
        const __m128i down = _mm_castps_si128(_mm_cmple_ps(fraction, _mm_set1_ps(-0.5f)));
        const __m128i rounded = _mm_add_epi32(_mm_sub_epi32(truncated, up), down);
        // clampU8 per lane, then the four bytes
        const __m128i clamped = _mm_min_epi32(_mm_max_epi32(rounded, _mm_setzero_si128()), _mm_set1_epi32(255));
        const __m128i bytes = _mm_packus_epi16(_mm_packus_epi32(clamped, clamped), _mm_setzero_si128());
        return static_cast<uint32_t>(_mm_cvtsi128_si32(bytes));
#else
        uint32_t out = 0u;
        for (uint32_t shift = 0u; shift < 32u; shift += 8u)
        {
            out |= static_cast<uint32_t>(lerpChannel(static_cast<uint8_t>(c00 >> shift), static_cast<uint8_t>(c10 >> shift),
                                                     static_cast<uint8_t>(c01 >> shift), static_cast<uint8_t>(c11 >> shift),
                                                     fx, fy))
                   << shift;
        }
        return out;
#endif
    }
}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        const uint32_t magv = static_cast<uint32_t>((display64 >> 27) & 0x03u);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = (dh + 1u) / (magv + 1u);
        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }
}

namespace
{
    uint32_t drawingThreadCount();

    // Work counters of one drawing thread, added to the profiler's once per band (one atomic add per triangle on
    // every band thread made them a shared hot spot).
    struct DrawCounters
    {
        uint64_t sprites = 0u;
        uint64_t spritePixels = 0u;
        uint64_t triangles = 0u;
        uint64_t triangleBoxPixels = 0u;
        uint64_t trianglePixels = 0u;
    };
    thread_local DrawCounters t_drawCounters{};

    // A quick 64-bit hash of a block whose size is a multiple of 32 bytes (for comparing GS memory, not security).
    uint64_t hashBlock(const uint8_t *data, size_t size)
    {
        uint64_t lanes[4] = {0x243F6A8885A308D3ull, 0x13198A2E03707344ull, 0xA4093822299F31D0ull, 0x082EFA98EC4E6C89ull};
        for (size_t offset = 0; offset + 32u <= size; offset += 32u)
            for (uint32_t lane = 0; lane < 4u; ++lane)
            {
                uint64_t word;
                std::memcpy(&word, data + offset + lane * 8u, sizeof(word));
                lanes[lane] = (lanes[lane] ^ word) * 0x9E3779B97F4A7C15ull;
                lanes[lane] ^= lanes[lane] >> 29u;
            }
        uint64_t hash = size;
        for (uint64_t lane : lanes)
            hash = (hash ^ lane) * 0xFF51AFD7ED558CCDull;
        return hash ^ (hash >> 32u);
    }
}

GSCpuBackend::GSCpuBackend(uint32_t drawingThreads, bool referenceMode)
    : m_referenceMode(referenceMode)
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            m_readVramFuncs[i] = ReadNull;
            m_writeVramFuncs[i] = WriteNull;
            break;
        }
    }
    Reset();
    m_drawingThreads = drawingThreads != 0u ? drawingThreads : drawingThreadCount();
    StartWorkers();
}

GSCpuBackend::~GSCpuBackend()
{
    StopWorkers();
}

// ---- Parallel drawing ------------------------------------------------------------------------------------------
// Submit() queues primitives; the queue is drawn before anything else touches GS memory (every other entry point
// flushes it), when it grows large, or when a new primitive would read texture pages a queued one writes (or write
// pages a queued one reads). Big queues are drawn by the worker threads + the caller, each taking its own bands of
// rows; small ones on the caller alone. VENDETTA_GS_THREADS=n sets the number of drawing threads (1 = off).

namespace
{
    uint32_t drawingThreadCount()
    {
        if (const char *env = std::getenv("VENDETTA_GS_THREADS"))
            return std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(env, nullptr, 10)), 1u, 32u);
        const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
        return std::clamp<uint32_t>(hw / 2u, 1u, 8u);
    }

    // The GS page (8 KB) of a pixel format, in pixels: 32-bit 64 x 32, 16-bit 64 x 64, 8-bit 128 x 64, 4-bit
    // 128 x 128 (the 8H / 4HL / 4HH textures live in 32-bit pages).
    void pageSize(uint32_t psm, uint32_t &width, uint32_t &height)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            width = 64u;
            height = 64u;
            return;
        case GS_PSM_T8:
            width = 128u;
            height = 64u;
            return;
        case GS_PSM_T4:
            width = 128u;
            height = 128u;
            return;
        default:
            width = 64u;
            height = 32u;
            return;
        }
    }

    // The pages [begin, end) a buffer at block `baseBlock` (64 words) with width `bw` (x 64 pixels), format `psm`
    // and `height` rows covers: whole page rows from its first page, plus the page it starts in when not aligned.
    // (An overlap only makes the draw queue flush early, so this may be slightly generous, never short.)
    std::pair<uint32_t, uint32_t> bufferPages(uint32_t baseBlock, uint32_t bw, uint32_t psm, uint32_t height)
    {
        uint32_t pageWidth = 64u, pageHeight = 32u;
        pageSize(psm, pageWidth, pageHeight);
        const uint32_t pagesPerRow = std::max(1u, (std::max(1u, bw) * 64u + pageWidth - 1u) / pageWidth);
        const uint32_t pageRows = std::max(1u, (height + pageHeight - 1u) / pageHeight);
        const uint32_t first = baseBlock / 32u;
        const uint32_t last = first + pagesPerRow * pageRows + ((baseBlock % 32u) != 0u ? 1u : 0u);
        return {first, last};
    }

    uint64_t estimatePixels(const GSPrimitiveBatch &batch)
    {
        const auto &ctx = batch.state.context;
        const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        float minX = 1.0e9f, minY = 1.0e9f, maxX = -1.0e9f, maxY = -1.0e9f;
        const uint32_t count = std::max<uint32_t>(1u, std::min<uint32_t>(batch.vertexCount, 3u));
        for (uint32_t i = 0; i < count; ++i)
        {
            minX = std::min(minX, batch.vertices[i].x - ofx);
            maxX = std::max(maxX, batch.vertices[i].x - ofx);
            minY = std::min(minY, batch.vertices[i].y - ofy);
            maxY = std::max(maxY, batch.vertices[i].y - ofy);
        }
        minX = std::max(minX, static_cast<float>(ctx.scissor.x0));
        minY = std::max(minY, static_cast<float>(ctx.scissor.y0));
        maxX = std::min(maxX, static_cast<float>(ctx.scissor.x1));
        maxY = std::min(maxY, static_cast<float>(ctx.scissor.y1));
        if (maxX < minX || maxY < minY)
            return 1u;
        return static_cast<uint64_t>(maxX - minX + 1.0f) * static_cast<uint64_t>(maxY - minY + 1.0f);
    }
}

void GSCpuBackend::StartWorkers()
{
    for (uint32_t i = 1; i < m_drawingThreads; ++i)
        m_workers.emplace_back([this, i]() { WorkerMain(i); });
}

void GSCpuBackend::StopWorkers()
{
    {
        std::lock_guard<std::mutex> lock(m_workMutex);
        m_stopWorkers = true;
    }
    m_workCv.notify_all();
    for (std::thread &worker : m_workers)
        if (worker.joinable())
            worker.join();
    m_workers.clear();
}

void GSCpuBackend::WorkerMain(uint32_t index)
{
    uint64_t seen = 0;
    for (;;)
    {
        uint32_t bandCount = 1u;
        {
            std::unique_lock<std::mutex> lock(m_workMutex);
            m_workCv.wait(lock, [&]() { return m_stopWorkers || m_workGeneration != seen; });
            if (m_stopWorkers)
                return;
            seen = m_workGeneration;
            bandCount = m_workBandCount;
        }
        DrawQueuedBand(RowBand{index, bandCount});
        {
            std::lock_guard<std::mutex> lock(m_workMutex);
            if (--m_workersPending == 0u)
                m_doneCv.notify_one();
        }
    }
}

void GSCpuBackend::DrawQueuedBand(const RowBand &band)
{
    // Every drawing thread draws in the same rounding mode (round to nearest), whatever its caller's mode: the band
    // workers start in it, and the band drawn by the flushing thread used to take its caller's.
    const unsigned fpEnv = gsEnterDrawFpEnv();
    t_drawCounters = {};
    if (m_queuedClut.empty())
    {
        for (const GSPrimitiveBatch &batch : m_queue)
            DrawPrimitive(batch, band, m_clut.data());
    }
    else
    {
        for (size_t index = 0; index < m_queue.size(); ++index)
            DrawPrimitive(m_queue[index], band, m_clutVersions[m_queuedClut[index]].data());
    }
    gsLeaveDrawFpEnv(fpEnv);
    const DrawCounters &counters = t_drawCounters;
    if (counters.sprites != 0u)
    {
        ps2_host_profile::count(ps2_host_profile::Sprites, counters.sprites);
        ps2_host_profile::count(ps2_host_profile::SpritePixels, counters.spritePixels);
    }
    if (counters.triangles != 0u)
    {
        ps2_host_profile::count(ps2_host_profile::Triangles, counters.triangles);
        ps2_host_profile::count(ps2_host_profile::TriangleBoxPixels, counters.triangleBoxPixels);
    }
    if (counters.trianglePixels != 0u)
        ps2_host_profile::count(ps2_host_profile::TrianglePixels, counters.trianglePixels);
}

void GSCpuBackend::QueueBatchUnlocked(const GSPrimitiveBatch &batch)
{
    const auto &ctx = batch.state.context;
    const uint32_t height = static_cast<uint32_t>(ctx.scissor.y1) + 1u;
    const auto framePages = bufferPages(ctx.frame.fbp * 32u, ctx.frame.fbw, ctx.frame.psm, height);
    const auto depthPages = bufferPages(ctx.zbuf.zbp * 32u, ctx.frame.fbw, ctx.zbuf.psm, height);
    // the layout a target's words are drawn through: the bands split by row y, and one word is on different rows
    // in two layouts (HCTP's 640-wide Z24 buffer and the PSMT4HL/HH font + HUD textures drawn into its top byte)
    auto targetLayout = [](uint32_t baseBlock, uint32_t bw, uint32_t psm)
    {
        uint32_t pageWidth = 64u, pageHeight = 32u;
        pageSize(psm, pageWidth, pageHeight);
        return (static_cast<uint64_t>(baseBlock) << 32u) | (static_cast<uint64_t>(bw & 0xFFFFu) << 16u) |
               (static_cast<uint64_t>(pageWidth) << 8u) | static_cast<uint64_t>(pageHeight & 0xFFu);
    };
    const PageRange frame{framePages.first, framePages.second, targetLayout(ctx.frame.fbp * 32u, ctx.frame.fbw, ctx.frame.psm)};
    const PageRange depth{depthPages.first, depthPages.second, targetLayout(ctx.zbuf.zbp * 32u, ctx.frame.fbw, ctx.zbuf.psm)};
    PageRange texture{};
    const bool textured = batch.state.prim.tme != 0u;
    if (textured)
    {
        const auto texturePages = bufferPages(ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, batch.state.textureHeight);
        texture = PageRange{texturePages.first, texturePages.second};
    }

    const bool full = m_queue.size() >= kMaxQueuedBatches;
    bool hazard = false;
    if (!full && textured)
        for (const PageRange &target : m_queuedTargets)
            if (target.overlaps(texture))
            {
                hazard = true;
                break;
            }
    if (!full && !hazard)
        for (const PageRange &read : m_queuedTextures)
            if (read.overlaps(frame) || read.overlaps(depth))
            {
                hazard = true;
                break;
            }
    // write after write through another layout: the two draws' bands would own the same words on different rows,
    // so threads raced their read-modify-writes (the dots on menu / HUD fonts: row 0 of each page lost its top byte)
    if (!full && !hazard)
        for (const PageRange &target : m_queuedTargets)
            if ((target.overlaps(frame) && target.layout != frame.layout) ||
                (target.overlaps(depth) && target.layout != depth.layout))
            {
                hazard = true;
                break;
            }
    if (full)
        FlushQueueUnlocked(ps2_host_profile::GsFlushFull);
    else if (hazard)
        FlushQueueUnlocked(ps2_host_profile::GsHazardFlushes);

    if (!m_referenceMode)
    {
        if (m_currentClutVersion == kNoClutVersion)
        {
            if (m_clutVersions.size() >= kMaxClutVersions)
                FlushQueueUnlocked(ps2_host_profile::GsFlushClut);
            m_clutVersions.push_back(m_clut);
            m_currentClutVersion = static_cast<uint16_t>(m_clutVersions.size() - 1u);
        }
        m_queuedClut.push_back(m_currentClutVersion);
    }
    m_queue.push_back(batch);
    // Keep each page range once: a frame uses only a handful (its frame / depth buffers and textures), and every new
    // primitive scans these lists — storing one entry per primitive made that scan grow with the queue (thousands of
    // checks per triangle for a fight's ~12,000 small triangles a frame).
    auto addRange = [](std::vector<PageRange> &ranges, const PageRange &range)
    {
        for (const PageRange &known : ranges)
            if (known.begin == range.begin && known.end == range.end && known.layout == range.layout)
                return;
        ranges.push_back(range);
    };
    addRange(m_queuedTargets, frame);
    addRange(m_queuedTargets, depth);
    if (textured)
        addRange(m_queuedTextures, texture);
    m_queuedPixels += estimatePixels(batch);
}

void GSCpuBackend::FlushQueueIfTouchedUnlocked(const PageRange &pages, FlushCause cause)
{
    if (m_queue.empty())
        return;
    for (const PageRange &target : m_queuedTargets)
        if (target.overlaps(pages))
        {
            FlushQueueUnlocked(cause);
            return;
        }
    for (const PageRange &read : m_queuedTextures)
        if (read.overlaps(pages))
        {
            FlushQueueUnlocked(cause);
            return;
        }
}

GSCpuBackend::PageRange GSCpuBackend::TransferDestinationPages() const
{
    const GSBitBltBuf &buffer = m_transfer.bitbltbuf;
    const auto pages = bufferPages(buffer.dbp, buffer.dbw, buffer.dpsm,
                                   static_cast<uint32_t>(m_transfer.trxpos.dsay) + m_transfer.trxreg.rrh);
    if (m_referenceMode)
        return PageRange{pages.first, pages.second};
    // The upload writes rows DSAY onwards only. The GS address of row y is on page base + (y / page height) x (the
    // format's pages per row) + more (x, block spill), so the page rows above DSAY hold none of its pixels: a texture
    // cache's other slots in the same buffer no longer look touched.
    uint32_t pageWidth = 64u, pageHeight = 32u;
    pageSize(buffer.dpsm, pageWidth, pageHeight);
    const bool smallTexels = buffer.dpsm == GS_PSM_T8 || buffer.dpsm == GS_PSM_T4;
    const uint32_t pagesPerRow = std::max(1u, smallTexels ? (static_cast<uint32_t>(buffer.dbw) >> 1u) : static_cast<uint32_t>(buffer.dbw));
    const uint32_t firstPage = pages.first + (static_cast<uint32_t>(m_transfer.trxpos.dsay) / pageHeight) * pagesPerRow;
    return PageRange{std::min(firstPage, pages.second), pages.second};
}

void GSCpuBackend::FlushQueueUnlocked(FlushCause cause)
{
    if (m_queue.empty())
        return;
    ps2_host_profile::Scope profile(ps2_host_profile::GsDraw);
    ps2_host_profile::count(ps2_host_profile::GsFlushes);
    ps2_host_profile::count(cause);
    if (m_workers.empty() || m_queuedPixels < kParallelPixelThreshold)
    {
        DrawQueuedBand(RowBand{});
    }
    else
    {
        const uint32_t bandCount = static_cast<uint32_t>(m_workers.size()) + 1u;
        ps2_host_profile::count(ps2_host_profile::GsParallelFlushes);
        {
            std::lock_guard<std::mutex> lock(m_workMutex);
            m_workersPending = static_cast<uint32_t>(m_workers.size());
            m_workBandCount = bandCount;
            ++m_workGeneration;
        }
        m_workCv.notify_all();
        DrawQueuedBand(RowBand{0u, bandCount});
        std::unique_lock<std::mutex> lock(m_workMutex);
        m_doneCv.wait(lock, [&]() { return m_workersPending == 0u; });
    }
    m_queue.clear();
    m_queuedTargets.clear();
    m_queuedTextures.clear();
    m_queuedPixels = 0u;
    m_queuedClut.clear();
    m_clutVersions.clear();
    m_currentClutVersion = kNoClutVersion;
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    if (vram && vramSize < GSMem::MEMORY_SIZE)
        throw std::invalid_argument("GS CPU backend requires at least 4 MiB of VRAM");

    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushOther);
    m_vram = vram;
    m_vramSize = vramSize;
    ResetUnlocked();
}

void GSCpuBackend::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushOther);
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    m_clut.fill(0u);
    m_currentClutVersion = kNoClutVersion;
    m_clutCbp.fill(0u);
    m_texturePageCache.Invalidate();
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || batch.vertexCount == 0u)
        return;
    QueueBatchUnlocked(batch);
}

void GSCpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    // a queued primitive drawing onto the palette's own pages must land before it is read (a read: queued primitives
    // that only read those pages — the texture the palette sits next to — do not matter; the reference mode keeps the
    // old, wider check)
    const PageRange palettePages{tex0.cbp / 32u, tex0.cbp / 32u + 2u};
    if (m_referenceMode)
        FlushQueueIfTouchedUnlocked(palettePages, ps2_host_profile::GsFlushClut);
    else if (!m_queue.empty())
        for (const PageRange &target : m_queuedTargets)
            if (target.overlaps(palettePages))
            {
                FlushQueueUnlocked(ps2_host_profile::GsFlushClut);
                break;
            }
    if (!m_vram || (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm)))
        return;

    switch (tex0.cld)
    {
    case 0u:
    case 6u:
    case 7u:
        return;
    case 1u:
        break;
    case 2u:
        m_clutCbp[0] = tex0.cbp;
        break;
    case 3u:
        m_clutCbp[1] = tex0.cbp;
        break;
    case 4u:
        if (m_clutCbp[0] == tex0.cbp)
            return;
        m_clutCbp[0] = tex0.cbp;
        break;
    case 5u:
        if (m_clutCbp[1] == tex0.cbp)
            return;
        m_clutCbp[1] = tex0.cbp;
        break;
    default:
        return;
    }

    // Queued primitives must draw with the palette they were given (most loads repeat the palette already loaded).
    // A changed palette gets its own snapshot at the next submit (QueueBatchUnlocked); the reference mode (the verify
    // shadow) instead puts the old palette back, drains the queue, then applies the new one.
    const std::array<uint16_t, 512> previous = m_clut;
    LoadClutUnlocked(tex0, texclut);
    if (m_clut != previous)
    {
        if (!m_referenceMode)
            m_currentClutVersion = kNoClutVersion;
        else if (!m_queue.empty())
        {
            const std::array<uint16_t, 512> loaded = m_clut;
            m_clut = previous;
            FlushQueueUnlocked(ps2_host_profile::GsFlushClut);
            m_clut = loaded;
        }
    }
}

void GSCpuBackend::LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return;

    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;

    const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwoBit && !fourBit;
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;

    for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
    {
        uint32_t sourceX = 0u;
        uint32_t sourceY = 0u;
        uint32_t sourceWidth = 1u;

        if (tex0.csm == 0u && fourBit)
        {
            // CSM1, 16 entries: the CLUT is an 8 × 2 block (entries 0-7 on row 0, 8-15 on row 1) — reading it as a
            // row of 16 took entries 8-15 from outside the CLUT (alpha 0: Def Jam Vendetta's anti-aliased menu font
            // lost every solid pixel to its alpha test).
            sourceX = entry & 0x07u;
            sourceY = entry >> 3u;
        }
        else if (tex0.csm == 0u)
        {
            const uint32_t sourceIndex = swizzleClutIndexCSM1(entry);
            sourceX = sourceIndex & 0x0Fu;
            sourceY = sourceIndex >> 4u;
        }
        else
        {
            sourceWidth = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
            sourceX = (static_cast<uint32_t>(texclut.cou) << 4u) + entry;
            sourceY = static_cast<uint32_t>(texclut.cov);
        }

        const uint32_t raw = ReadTextureVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY); 
        const uint32_t destination = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteenBit ? 0x1FFu : 0x0FFu);
        if (sixteenBit)
        {
            m_clut[destination] = static_cast<uint16_t>(raw);
        }
        else
        {
            m_clut[destination] = static_cast<uint16_t>(raw & 0xFFFFu);
            m_clut[destination + 256u] = static_cast<uint16_t>(raw >> 16u);
        }
    }
}

void GSCpuBackend::Flush()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushExplicit);
}

void GSCpuBackend::TextureFlush()
{
    // TEXFLUSH needs no draining here: this renderer has no texture cache (reads come straight from GS memory) and a
    // primitive reading pages a queued one draws to already drains the queue (QueueBatchUnlocked).
    std::lock_guard<std::mutex> lock(m_mutex);
    m_texturePageCache.Invalidate();
}

void GSCpuBackend::Sync(GSSyncReason)
{
    // CPU backend is immediate. GPU backends may wait on fences/readbacks here.
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushReadback);
    return ReadVramUnlocked(psm, base, bw, x, y);
}

void GSCpuBackend::ReadVramRect(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                                uint32_t *out) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushReadback);
    if (!m_vram)
    {
        std::fill(out, out + static_cast<size_t>(w) * h, 0u);
        return;
    }
    const ReadVramFunc fn = m_readVramFuncs[psm & 0x3Fu];
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            *out++ = fn(m_vram, base, bw, x0 + x, y0 + y);
}

namespace
{
    // The reads / writes m_readVramFuncs / m_writeVramFuncs make for the frame and depth formats drawing uses most
    // (GSMem::ReadCT32 / CT24 / CT16 / Z32 / Z24 / Z16 and the writes), inlined; false = another format.
    GS_FORCEINLINE bool readVramDirect(uint32_t psm, const uint8_t *vram, u32 base, u32 bw, u32 x, u32 y, u32 &out)
    {
        using namespace GSMem;
        switch (psm)
        {
        case GS_PSM_CT32:
            out = PixelStorageTraits<C32>::Read(PageTableC32, vram, base, bw, x, y);
            return true;
        case GS_PSM_CT24:
            out = PixelStorageTraits<C24>::Read(PageTableC32, vram, base, bw, x, y);
            return true;
        case GS_PSM_CT16:
            out = PixelStorageTraits<C16>::Read(PageTableC16, vram, base, bw, x, y);
            return true;
        case GS_PSM_Z32:
            out = PixelStorageTraits<Z32>::Read(PageTableZ32, vram, base, bw, x, y);
            return true;
        case GS_PSM_Z24:
            out = PixelStorageTraits<Z24>::Read(PageTableZ32, vram, base, bw, x, y);
            return true;
        case GS_PSM_Z16:
            out = PixelStorageTraits<Z16>::Read(PageTableZ16, vram, base, bw, x, y);
            return true;
        default:
            return false;
        }
    }

    GS_FORCEINLINE bool writeVramDirect(uint32_t psm, uint8_t *vram, u32 base, u32 bw, u32 x, u32 y, u32 value)
    {
        using namespace GSMem;
        switch (psm)
        {
        case GS_PSM_CT32:
            PixelStorageTraits<C32>::Write(PageTableC32, vram, base, bw, x, y, value);
            return true;
        case GS_PSM_CT24:
            PixelStorageTraits<C24>::Write(PageTableC32, vram, base, bw, x, y, value);
            return true;
        case GS_PSM_CT16:
            PixelStorageTraits<C16>::Write(PageTableC16, vram, base, bw, x, y, static_cast<u16>(value));
            return true;
        case GS_PSM_Z32:
            PixelStorageTraits<Z32>::Write(PageTableZ32, vram, base, bw, x, y, value);
            return true;
        case GS_PSM_Z24:
            PixelStorageTraits<Z24>::Write(PageTableZ32, vram, base, bw, x, y, value);
            return true;
        case GS_PSM_Z16:
            PixelStorageTraits<Z16>::Write(PageTableZ16, vram, base, bw, x, y, static_cast<u16>(value));
            return true;
        default:
            return false;
        }
    }
}

GS_FORCEINLINE uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    // the common formats inline; the reference renderer (VENDETTA_GS_VERIFY's shadow) keeps the table, so the verify
    // mode compares the two
    u32 out = 0u;
    if (!m_referenceMode && readVramDirect(psm, m_vram, base, bw, x, y, out))
        return out;
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

GS_FORCEINLINE uint32_t GSCpuBackend::ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    if (!m_vram)
        return 0u;

    // Straight from GS memory, like every other read. The one-page copy cache (GSMem::TexturePageCache) copied
    // 8 KB each time a read crossed into another page — sprites crossing page edges on every row, bilinear
    // samples straddling an edge — and was the bulk of the rasteriser's time in Def Jam Vendetta's menus.
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushOther);
    WriteVramUnlocked(psm, base, bw, x, y, value);
}

GS_FORCEINLINE void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (!m_vram)
        return;
    if (!m_referenceMode && writeVramDirect(psm, m_vram, base, bw, x, y, value))
        return;
    m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value);
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushReadback);
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

namespace
{
    // VENDETTA_SPRITELOG's order stream: runs of S (sprites) / T (other primitives) per frame buffer, and P (a
    // presentation: the GS memory copied for the screen) — from the drawing and the presenting threads, in order
    struct SpriteOrderLog
    {
        std::mutex mutex;
        std::string text;
        char runKind = 0;
        uint32_t runFbp = 0xFFFFFFFFu;
        uint32_t runLength = 0u;

        void note(char kind, uint32_t fbp)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (kind == runKind && fbp == runFbp && kind != 'P')
            {
                ++runLength;
                return;
            }
            flushRun();
            runKind = kind;
            runFbp = fbp;
            runLength = 1u;
        }

        void flushRun()
        {
            if (runKind == 0)
                return;
            char item[48];
            if (runKind == 'P')
                std::snprintf(item, sizeof(item), " P(tick %u)", runFbp);
            else
                std::snprintf(item, sizeof(item), " %c%u@%x", runKind, runLength, runFbp);
            text += item;
            runKind = 0;
        }
    };

    SpriteOrderLog &spriteOrderLog()
    {
        static SpriteOrderLog log;
        return log;
    }

    double spriteLogAt()
    {
        static const double at = []
        {
            const char *value = std::getenv("VENDETTA_SPRITELOG");
            return value ? std::atof(value) : -1.0;
        }();
        return at;
    }

    const auto g_processStart = std::chrono::steady_clock::now(); // at program start
    double secondsSinceStart()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_processStart).count();
    }
}

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch, const RowBand &band, const uint16_t *clut)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    {
        // VENDETTA_SPRITELOG=<seconds>: for ~50 ms from that many seconds after start, every sprite drawn and a count
        // of the other primitives by type, with the drawing state (where a 2-D layer goes and why it may not show)
        const double logAt = spriteLogAt();
        if (logAt >= 0.0 && band.index == 0u)
        {
            const double spriteLogAt = logAt;
            const double now = secondsSinceStart();
            thread_local uint32_t otherCounts[8] = {};
            thread_local bool reported = false;
            const auto noteOrder = [&](char kind)
            { spriteOrderLog().note(kind, ctx.frame.fbp); };
            if (now >= spriteLogAt && now < spriteLogAt + 0.05)
            {
                noteOrder(state.prim.type == GS_PRIM_SPRITE ? 'S' : 'T');
                if (state.prim.type == GS_PRIM_SPRITE)
                {
                    uint32_t maxAlpha = 0u, nonZero = 0u;
                    if (state.prim.tme)
                    {
                        for (uint32_t ty = 0; ty < 32u; ty += 2u)
                            for (uint32_t tx = 0; tx < 32u; tx += 2u)
                            {
                                const uint32_t texel = SampleTexture(state, clut, 0.0f, 0.0f, 1.0f, static_cast<uint16_t>(tx * 16u + 8u), static_cast<uint16_t>(ty * 16u + 8u));
                                maxAlpha = std::max(maxAlpha, texel >> 24u);
                                nonZero += (texel != 0u) ? 1u : 0u;
                            }
                    }
                    std::fprintf(stderr, "[sprite]   texels: max alpha %u, %u of 256 non-zero\n", maxAlpha, nonZero);
                    const GSVertex &v0 = batch.vertices[0];
                    const GSVertex &v1 = batch.vertices[batch.vertexCount > 1 ? 1 : 0];
                    std::fprintf(stderr, "[sprite] %.3f tme=%u fst=%u abe=%u ctxt=%u fbp=0x%x fbw=%u fpsm=0x%x fbmsk=0x%x zbp=0x%x zpsm=0x%x zmsk=%u test=0x%llx alpha=0x%llx xy=(%.1f,%.1f)-(%.1f,%.1f) z=%.0f uv=(%u,%u)-(%u,%u) st=(%.3f,%.3f)-(%.3f,%.3f) rgba=%u,%u,%u,%u tex=tbp0x%x/tbw%u/psm0x%x/%ux%u/tcc%u/tfx%u/cbp0x%x sc=%u-%u,%u-%u off=%u,%u tex1=0x%llx\n",
                                 now, state.prim.tme, state.prim.fst, state.prim.abe, state.prim.ctxt, ctx.frame.fbp, ctx.frame.fbw, ctx.frame.psm, ctx.frame.fbmsk,
                                 ctx.zbuf.zbp, ctx.zbuf.psm, ctx.zbuf.zmask ? 1u : 0u,
                                 static_cast<unsigned long long>(ctx.test), static_cast<unsigned long long>(ctx.alpha),
                                 v0.x, v0.y, v1.x, v1.y, v1.z, v0.u, v0.v, v1.u, v1.v, v0.s, v0.t, v1.s, v1.t, v1.r, v1.g, v1.b, v1.a,
                                 ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 1u << ctx.tex0.tw, 1u << ctx.tex0.th, ctx.tex0.tcc, ctx.tex0.tfx, ctx.tex0.cbp,
                                 ctx.scissor.x0, ctx.scissor.x1, ctx.scissor.y0, ctx.scissor.y1, ctx.xyoffset.ofx, ctx.xyoffset.ofy,
                                 static_cast<unsigned long long>(ctx.tex1));
                }
                else
                {
                    ++otherCounts[state.prim.type & 7u];
                }
            }
            else if (now >= spriteLogAt + 0.05 && !reported)
            {
                reported = true;
                std::fprintf(stderr, "[sprite] other primitives in the window, by type 0-7: %u %u %u %u %u %u %u %u\n",
                             otherCounts[0], otherCounts[1], otherCounts[2], otherCounts[3], otherCounts[4], otherCounts[5], otherCounts[6], otherCounts[7]);
                SpriteOrderLog &orderLog = spriteOrderLog();
                std::lock_guard<std::mutex> lock(orderLog.mutex);
                orderLog.flushRun();
                std::fprintf(stderr, "[sprite] order (S = sprites, T = other primitives, @ frame buffer; P = a presentation):%s\n",
                             orderLog.text.c_str());
            }
        }
    }
    {
        // VENDETTA_PRIMLOG=<from s>,<to s>,x0,y0,x1,y1: every primitive drawn between those seconds after start whose
        // screen box (after the XY offset) touches that rectangle, with its drawing state — to diff two frames of a draw
        // that changes (a flicker). One line per primitive; "P" lines from the presentation mark the frames.
        struct PrimLogWindow
        {
            bool on = false;
            double from = 0.0, to = 0.0;
            float x0 = 0.0f, y0 = 0.0f, x1 = 0.0f, y1 = 0.0f;
        };
        static const PrimLogWindow window = []
        {
            PrimLogWindow w;
            const char *value = std::getenv("VENDETTA_PRIMLOG");
            if (value)
                w.on = std::sscanf(value, "%lf,%lf,%f,%f,%f,%f", &w.from, &w.to, &w.x0, &w.y0, &w.x1, &w.y1) == 6;
            return w;
        }();
        if (window.on && band.index == 0u)
        {
            const double now = secondsSinceStart();
            if (now >= window.from && now < window.to)
            {
                const float ox = static_cast<float>(ctx.xyoffset.ofx) / 16.0f, oy = static_cast<float>(ctx.xyoffset.ofy) / 16.0f;
                float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
                for (uint32_t i = 0; i < batch.vertexCount; ++i)
                {
                    minX = std::min(minX, batch.vertices[i].x - ox);
                    maxX = std::max(maxX, batch.vertices[i].x - ox);
                    minY = std::min(minY, batch.vertices[i].y - oy);
                    maxY = std::max(maxY, batch.vertices[i].y - oy);
                }
                if (batch.vertexCount && maxX >= window.x0 && minX <= window.x1 && maxY >= window.y0 && minY <= window.y1)
                {
                    const GSVertex &v0 = batch.vertices[0];
                    std::fprintf(stderr, "[prim] %.4f type=%u n=%u box=(%.1f,%.1f)-(%.1f,%.1f) z=%.0f..%.0f..%.0f tme=%u abe=%u fst=%u fge=%u iip=%u fbp=0x%x fbmsk=0x%x zbp=0x%x zmsk=%u test=0x%llx alpha=0x%llx fba=%u tex=0x%x/%u/0x%x/%ux%u/tcc%u/tfx%u/cbp0x%x rgba=%u,%u,%u,%u|%u,%u,%u,%u|%u,%u,%u,%u q=%.4f st=%.4f,%.4f\n",
                                 now, static_cast<unsigned>(state.prim.type), batch.vertexCount, minX, minY, maxX, maxY,
                                 batch.vertices[0].z, batch.vertices[batch.vertexCount > 1 ? 1 : 0].z, batch.vertices[batch.vertexCount - 1].z,
                                 state.prim.tme, state.prim.abe, state.prim.fst, state.prim.fge, state.prim.iip, ctx.frame.fbp, ctx.frame.fbmsk,
                                 ctx.zbuf.zbp, ctx.zbuf.zmask ? 1u : 0u, static_cast<unsigned long long>(ctx.test),
                                 static_cast<unsigned long long>(ctx.alpha), static_cast<unsigned>(ctx.fba),
                                 ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 1u << ctx.tex0.tw, 1u << ctx.tex0.th, ctx.tex0.tcc, ctx.tex0.tfx, ctx.tex0.cbp,
                                 v0.r, v0.g, v0.b, v0.a,
                                 batch.vertices[batch.vertexCount > 1 ? 1 : 0].r, batch.vertices[batch.vertexCount > 1 ? 1 : 0].g,
                                 batch.vertices[batch.vertexCount > 1 ? 1 : 0].b, batch.vertices[batch.vertexCount > 1 ? 1 : 0].a,
                                 batch.vertices[batch.vertexCount - 1].r, batch.vertices[batch.vertexCount - 1].g,
                                 batch.vertices[batch.vertexCount - 1].b, batch.vertices[batch.vertexCount - 1].a, v0.q, v0.s, v0.t);
                }
            }
        }
    }
    {
        // VENDETTA_GSLOG=1: draws that use the menu font's one-glyph texture cache (TBP 0x3FBD)
        static const bool drawLog = std::getenv("VENDETTA_GSLOG") != nullptr;
        thread_local uint32_t drawLogs = 0u;
        if (drawLog && band.index == 0u && drawLogs < 300u && state.prim.tme && ctx.tex0.tbp0 == 0x3FBDu)
        {
            ++drawLogs;
            const GSVertex &v0 = batch.vertices[0];
            const GSVertex &v1 = batch.vertices[batch.vertexCount > 1 ? 1 : 0];
            std::fprintf(stderr, "[gslog] glyph draw type=%u n=%u fst=%u abe=%u fbp=0x%x fbw=%u fpsm=0x%x fbmsk=0x%x xy=(%.1f,%.1f)-(%.1f,%.1f) uv=(%u,%u)-(%u,%u) st=(%.3f,%.3f) rgba=%u,%u,%u,%u test=0x%llx alpha=0x%llx sc=%u-%u,%u-%u xyoff=%u,%u ctxt=%u%c",
                         static_cast<unsigned>(state.prim.type), batch.vertexCount, state.prim.fst, state.prim.abe, ctx.frame.fbp, ctx.frame.fbw, ctx.frame.psm, ctx.frame.fbmsk,
                         v0.x, v0.y, v1.x, v1.y, v0.u, v0.v, v1.u, v1.v, v0.s, v0.t, v0.r, v0.g, v0.b, v0.a,
                         static_cast<unsigned long long>(ctx.test), static_cast<unsigned long long>(ctx.alpha),
                         ctx.scissor.x0, ctx.scissor.x1, ctx.scissor.y0, ctx.scissor.y1, ctx.xyoffset.ofx, ctx.xyoffset.ofy, state.prim.ctxt, 10);
            if (drawLogs <= 3u)
            {
                std::fprintf(stderr, "[gslog]   loaded clut (lo/hi):");
                for (uint32_t i = 0; i < 16u; ++i)
                    std::fprintf(stderr, " %04x%04x", clut[i + 256u], clut[i]);
                std::fprintf(stderr, "%c[gslog]   texels (u,v -> rgba):", 10);
                for (uint32_t pv = 0; pv < 24u; pv += 6u)
                    for (uint32_t pu = 0; pu < 12u; pu += 4u)
                        std::fprintf(stderr, " (%u,%u)%08x", pu, pv, SampleTexture(state, clut, 0.0f, 0.0f, 1.0f, static_cast<uint16_t>(pu * 16u + 8u), static_cast<uint16_t>(pv * 16u + 8u)));
                std::fprintf(stderr, " texW=%u texH=%u v1rgba=%u,%u,%u,%u%c", state.textureWidth, state.textureHeight, v1.r, v1.g, v1.b, v1.a, 10);
            }
        }
    }
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << batch.vertices[0].x << "," << batch.vertices[0].y << ")"
                      << " uv0=(" << (batch.vertices[0].u >> 4) << "," << (batch.vertices[0].v >> 4) << ")"
                      << " stq0=(" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q << ")"
                      << " v1=(" << batch.vertices[1].x << "," << batch.vertices[1].y << ")"
                      << " uv1=(" << (batch.vertices[1].u >> 4) << "," << (batch.vertices[1].v >> 4) << ")"
                      << " stq1=(" << batch.vertices[1].s << "," << batch.vertices[1].t << "," << batch.vertices[1].q << ")"
                      << " v2=(" << batch.vertices[2].x << "," << batch.vertices[2].y << ")"
                      << " uv2=(" << (batch.vertices[2].u >> 4) << "," << (batch.vertices[2].v >> 4) << ")"
                      << " stq2=(" << batch.vertices[2].s << "," << batch.vertices[2].t << "," << batch.vertices[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(batch.vertices[0].r) << ","
                      << static_cast<uint32_t>(batch.vertices[0].g) << ","
                      << static_cast<uint32_t>(batch.vertices[0].b) << ","
                      << static_cast<uint32_t>(batch.vertices[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(batch.vertices[1].r) << ","
                      << static_cast<uint32_t>(batch.vertices[1].g) << ","
                      << static_cast<uint32_t>(batch.vertices[1].b) << ","
                      << static_cast<uint32_t>(batch.vertices[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(batch.vertices[2].r) << ","
                      << static_cast<uint32_t>(batch.vertices[2].g) << ","
                      << static_cast<uint32_t>(batch.vertices[2].b) << ","
                      << static_cast<uint32_t>(batch.vertices[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((state.prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    switch (state.prim.type)
    {
    case GS_PRIM_SPRITE:
        DrawSprite(batch, band, clut);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        DrawTriangle(batch, band, clut);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        DrawLine(batch, band);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        const auto &ctx = state.context;
        int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
        if (band.owns(py))
            WritePixel(state, px, py, static_cast<u32>(v.z), v.r, v.g, v.b, v.a, v.fog);
        break;
    }
    default:
        break;
    }
}

void GSCpuBackend::WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    const auto &ctx = state.context;
    if (x < ctx.scissor.x0 || x > ctx.scissor.x1 || y < ctx.scissor.y0 || y > ctx.scissor.y1)
        return;

    if (state.prim.fge)
    {
        const uint32_t inverseFog = 255u - fog;
        auto applyFog = [&](uint8_t input, uint8_t fogColor) -> uint8_t
        {
            return static_cast<uint8_t>(((static_cast<uint32_t>(fog) * input) >> 8) + ((inverseFog * fogColor) >> 8));
        };

        r = applyFog(r, state.fogR);
        g = applyFog(g, state.fogG);
        b = applyFog(b, state.fogB);
    }

    const u32 fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const u32 fbw = std::max<u32>(ctx.frame.fbw, 1u);
    const u32 fpsm = ctx.frame.psm;
    const u32 zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const u32 zpsm = ctx.zbuf.psm;

    const PixelWriteMask writeMask = classifyAlphaTest(ctx.test, a, static_cast<uint8_t>(fpsm));
    if (!writeMask.writesAnything())
    {
        return;
    }

    // TEST.ZTE (bit 16) clear = no depth test: every pixel passes (the GS manual calls ZTE 0 "prohibited", but games
    // draw 2-D layers that way and the hardware / PCSX2 pass them all). Reading only ZTST, whose 0 is NEVER, dropped
    // every such pixel — Def Jam Vendetta's whole in-match HUD (life bars, timer, names, FIGHT!, the pause menu).
    const uint32_t ztestMethod = ((ctx.test >> 16) & 1u) != 0u ? static_cast<uint32_t>((ctx.test >> 17) & 3u) : 1u;
    const bool alphaBlendEnabled = state.prim.abe;
    const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && fpsm == GS_PSM_CT32;
    const bool destinationAlphaTestNeedsRead = ((ctx.test >> 14) & 0x1u) != 0u && (fpsm == GS_PSM_CT32 || fpsm == GS_PSM_CT16 || fpsm == GS_PSM_CT16S);

    // small optimization, avoid reading the framebuffer for simple draws
    // TODO: only one address lookup for rmw
    const bool frmw = destinationAlphaTestNeedsRead || (writeMask.writesFramebuffer() && ((ctx.frame.fbmsk != 0) || alphaBlendEnabled || preserveDestinationAlpha));

    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        rawFramebufferPixel = ReadVramUnlocked(fpsm, fbp, fbw, x, y);
        fbrgba = rawFramebufferPixel;

        if (bitsPerPixel(fpsm) == 16)
        {
            fbrgba = Rgba5551ToRgba8888(fbrgba);
        }
        else if (fpsm == GS_PSM_CT24)
        {
            // The GS supplies 0x80 as destination alpha for RGB24 blending.
            fbrgba |= 0x80000000u;
        }
    }

    if (!passesDestinationAlphaTest(ctx.test, static_cast<uint8_t>(fpsm), rawFramebufferPixel))
    {
        return;
    }

    bool zpass = false;
    uint32_t storedZ = 0u;
    switch (ztestMethod)
    {
    case 0:
        zpass = false;
        break;
    case 1:
        zpass = true;
        break;
    case 2:
        storedZ = ReadVramUnlocked(zpsm, zbp, fbw, x, y);
        zpass = static_cast<uint32_t>(z) >= storedZ;
        break;
    case 3:
        storedZ = ReadVramUnlocked(zpsm, zbp, fbw, x, y);
        zpass = static_cast<uint32_t>(z) > storedZ;
        break;
    }

    if (!zpass)
    {
        return;
    }

    if (writeMask.writesFramebuffer())
    {
        const u8 srcR = r;
        const u8 srcG = g;
        const u8 srcB = b;

        if (state.prim.abe)
        {
            uint8_t dr = fbrgba & 0xFF;
            uint8_t dg = (fbrgba >> 8) & 0xFF;
            uint8_t db = (fbrgba >> 16) & 0xFF;
            uint8_t da = (fbrgba >> 24) & 0xFF;

            // PABE disables alpha blending when the source alpha MSB is clear.
            if (!(state.pabe && (a & 0x80u) == 0u))
            {
                uint64_t alphaReg = ctx.alpha;
                uint8_t asel = alphaReg & 3;
                uint8_t bsel = (alphaReg >> 2) & 3;
                uint8_t csel = (alphaReg >> 4) & 3;
                uint8_t dsel = (alphaReg >> 6) & 3;
                uint8_t fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);

                auto pickRGB = [&](uint8_t sel, int cs, int cd) -> int
                {
                    if (sel == 0)
                        return cs;
                    if (sel == 1)
                        return cd;
                    return 0;
                };
                int cAlpha = (csel == 0) ? a : (csel == 1) ? da
                                                           : fix;

                r = clampU8(((pickRGB(asel, r, dr) - pickRGB(bsel, r, dr)) * cAlpha >> 7) + pickRGB(dsel, r, dr));
                g = clampU8(((pickRGB(asel, g, dg) - pickRGB(bsel, g, dg)) * cAlpha >> 7) + pickRGB(dsel, g, dg));
                b = clampU8(((pickRGB(asel, b, db) - pickRGB(bsel, b, db)) * cAlpha >> 7) + pickRGB(dsel, b, db));
            }
            else
            {
                r = srcR;
                g = srcG;
                b = srcB;
            }
        }

        if (writeMask.writeAlpha && (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(r, g, b, a);

        if (ctx.frame.fbmsk != 0)
        {
            pixel = (pixel & ~ctx.frame.fbmsk) | (fbrgba & ctx.frame.fbmsk);
        }

        if (preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        // format conversion
        if (bitsPerPixel(fpsm) == 16)
        {
            pixel = Rgba8888ToRgba5551(pixel);
        }

        WriteVramUnlocked(fpsm, fbp, fbw, x, y, pixel);
    }

    if (writeMask.writeDepth && !ctx.zbuf.zmask)
    {
        WriteVramUnlocked(zpsm, zbp, fbw, x, y, z);
    }
}

namespace
{
    // a palette entry as RGBA (GSCpuBackend::LookupCLUT; free, for the format-specific texel samplers)
    GS_FORCEINLINE uint32_t clutColor(const GSDrawState &state,
                                      const uint16_t *clut,
                                      uint8_t index,
                                      uint8_t cpsm,
                                      uint8_t csm,
                                      uint8_t csa,
                                      uint8_t sourcePsm)
{
    const bool sixteenBit = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;
    const uint32_t sourceIndex = isFourBitIndexedPsm(sourcePsm)
                                     ? (static_cast<uint32_t>(index) & 0x0Fu)
                                     : static_cast<uint32_t>(index);

    uint32_t clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
    if (!sixteenBit && csm == 0u && isEightBitIndexedPsm(sourcePsm))
    {
        const uint32_t block = std::min((sourceIndex & 0xF0u) + clutBase, 240u);
        clutIndex = block + (sourceIndex & 0x0Fu);
    }

    switch (cpsm)
    {
    case GS_PSM_CT32:
    {
        const uint32_t raw = static_cast<uint32_t>(clut[clutIndex]) | (static_cast<uint32_t>(clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw);
    }
    case GS_PSM_CT24:
    {
        const uint32_t raw = static_cast<uint32_t>(clut[clutIndex]) | (static_cast<uint32_t>(clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw & 0x00FFFFFFu);
    }
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(clut[clutIndex]));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

    // a texel of texture format Psm straight from GS memory: the read m_readVramFuncs[Psm] makes
    // (GSMem::ReadCT32 / ReadCT24 / ReadCT16 / ReadCT16S / ReadP8 / ReadP4), inlined
    template <uint8_t Psm>
    GS_FORCEINLINE uint32_t readTexel(const uint8_t *vram, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
    {
        using namespace GSMem;
        if constexpr (Psm == GS_PSM_CT32)
            return PixelStorageTraits<C32>::Read(PageTableC32, vram, base, bw, x, y);
        else if constexpr (Psm == GS_PSM_CT24)
            return PixelStorageTraits<C24>::Read(PageTableC32, vram, base, bw, x, y);
        else if constexpr (Psm == GS_PSM_CT16)
            return PixelStorageTraits<C16>::Read(PageTableC16, vram, base, bw, x, y);
        else if constexpr (Psm == GS_PSM_CT16S)
            return PixelStorageTraits<C16S>::Read(PageTableC16S, vram, base, bw, x, y);
        else if constexpr (Psm == GS_PSM_T8)
            return PixelStorageTraits<P8>::Read(PageTableP8, vram, base, bw, x, y);
        else
        {
            static_assert(Psm == GS_PSM_T4);
            return PixelStorageTraits<P4>::Read(PageTableP4, vram, base, bw, x, y);
        }
    }

    // one texel of a texture whose format is known ahead (the generic sampler in SampleTexture, with its format
    // switch and the read through m_readVramFuncs resolved at compile time)
    template <uint8_t Psm>
    struct FormatTexelSampler
    {
        const GSDrawState &state;
        const uint16_t *clut;
        const GSTex0Reg &tex;
        const uint8_t *vram;
        int texW, texH;
        uint8_t wrapU, wrapV;
        uint16_t minU, maxU, minV, maxV;

        GS_FORCEINLINE uint32_t operator()(int sampleU, int sampleV) const
        {
            sampleU = wrapTextureCoordinate(sampleU, texW, wrapU, minU, maxU);
            sampleV = wrapTextureCoordinate(sampleV, texH, wrapV, minV, maxV);
            const u32 out = readTexel<Psm>(vram, tex.tbp0, tex.tbw, sampleU, sampleV);
            if constexpr (Psm == GS_PSM_CT32 || Psm == GS_PSM_CT24)
                return applyTexa(state.texa, Psm, out);
            else if constexpr (Psm == GS_PSM_CT16 || Psm == GS_PSM_CT16S)
                return applyTexa(state.texa, Psm, Rgba5551ToRgba8888(out));
            else
                return clutColor(state, clut, static_cast<u8>(out), tex.cpsm, tex.csm, tex.csa, Psm);
        }
    };

    // point or bilinear sampling with a texel sampler (shared by the generic and the format-specific paths)
    template <typename Sampler>
    GS_FORCEINLINE uint32_t filterTexture(const Sampler &samplePoint, bool linear, float texUf, float texVf)
    {
        if (!linear)
            return samplePoint(static_cast<int>(texUf), static_cast<int>(texVf));

        const float sampleU = texUf - 0.5f;
        const float sampleV = texVf - 0.5f;
        const int u0 = static_cast<int>(std::floor(sampleU));
        const int v0 = static_cast<int>(std::floor(sampleV));
        const int u1 = u0 + 1;
        const int v1 = v0 + 1;
        const float fx = sampleU - static_cast<float>(u0);
        const float fy = sampleV - static_cast<float>(v0);

        const uint32_t c00 = samplePoint(u0, v0);
        const uint32_t c10 = samplePoint(u1, v0);
        const uint32_t c01 = samplePoint(u0, v1);
        const uint32_t c11 = samplePoint(u1, v1);

        return bilinearBlend(c00, c10, c01, c11, fx, fy);
    }
}

GS_FORCEINLINE uint32_t GSCpuBackend::LookupCLUT(const GSDrawState &state,
                                                 const uint16_t *clut,
                                                 uint8_t index,
                                                 uint8_t cpsm,
                                                 uint8_t csm,
                                                 uint8_t csa,
                                                 uint8_t sourcePsm)
{
    return clutColor(state, clut, index, cpsm, csm, csa, sourcePsm);
}

uint32_t GSCpuBackend::SampleTexture(const GSDrawState &state, const uint16_t *clut, float s, float t, float q, uint16_t u, uint16_t v)
{
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;

    const int texW = state.textureWidth;
    const int texH = state.textureHeight;
    const uint64_t clamp = ctx.clamp;
    const uint8_t wrapU = static_cast<uint8_t>(clamp & 0x3u);
    const uint8_t wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    const uint16_t minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
    const uint16_t maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
    const uint16_t minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
    const uint16_t maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);

    float texUf, texVf;
    if (state.prim.fst)
    {
        texUf = static_cast<float>(u) / 16.0f;
        texVf = static_cast<float>(v) / 16.0f;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        texUf = s * invQ * static_cast<float>(texW);
        texVf = t * invQ * static_cast<float>(texH);
    }

    // one texel (a local struct rather than a lambda, so its call can be forced inline)
    struct TexelSampler
    {
        GSCpuBackend &self;
        const GSDrawState &state;
        const uint16_t *clut;
        const GSTex0Reg &tex;
        int texW, texH;
        uint8_t wrapU, wrapV;
        uint16_t minU, maxU, minV, maxV;

        GS_FORCEINLINE uint32_t operator()(int sampleU, int sampleV) const
        {
            sampleU = wrapTextureCoordinate(sampleU, texW, wrapU, minU, maxU);
            sampleV = wrapTextureCoordinate(sampleV, texH, wrapV, minV, maxV);

            u32 out = self.ReadTextureVramUnlocked(tex.psm, tex.tbp0, tex.tbw, sampleU, sampleV);

            switch (tex.psm)
            {
            case GS_PSM_CT32:
            case GS_PSM_Z32:
            case GS_PSM_CT24:
            case GS_PSM_Z24:
                return applyTexa(state.texa, tex.psm, out);
            case GS_PSM_CT16:
            case GS_PSM_CT16S:
            case GS_PSM_Z16:
            case GS_PSM_Z16S:
                return applyTexa(state.texa, tex.psm, Rgba5551ToRgba8888(out));
            case GS_PSM_T8:
            case GS_PSM_T8H:
            case GS_PSM_T4:
            case GS_PSM_T4HL:
            case GS_PSM_T4HH:
                return self.LookupCLUT(state, clut, static_cast<u8>(out), tex.cpsm, tex.csm, tex.csa, tex.psm);
            }

            return 0xFFFF00FFu;
        }
    };
    // The common texture formats sample through a sampler with the format fixed: no indirect read call and no format
    // switch per texel. The reference renderer (VENDETTA_GS_VERIFY's shadow) keeps the generic sampler, so the verify
    // mode compares the two.
    if (!m_referenceMode && m_vram)
    {
        switch (tex.psm)
        {
        case GS_PSM_CT32:
            return filterTexture(FormatTexelSampler<GS_PSM_CT32>{state, clut, tex, m_vram, texW, texH, wrapU, wrapV, minU, maxU, minV, maxV},
                                 state.linearFilter, texUf, texVf);
        case GS_PSM_CT24:
            return filterTexture(FormatTexelSampler<GS_PSM_CT24>{state, clut, tex, m_vram, texW, texH, wrapU, wrapV, minU, maxU, minV, maxV},
                                 state.linearFilter, texUf, texVf);
        case GS_PSM_CT16:
            return filterTexture(FormatTexelSampler<GS_PSM_CT16>{state, clut, tex, m_vram, texW, texH, wrapU, wrapV, minU, maxU, minV, maxV},
                                 state.linearFilter, texUf, texVf);
        case GS_PSM_CT16S:
            return filterTexture(FormatTexelSampler<GS_PSM_CT16S>{state, clut, tex, m_vram, texW, texH, wrapU, wrapV, minU, maxU, minV, maxV},
                                 state.linearFilter, texUf, texVf);
        case GS_PSM_T8:
            return filterTexture(FormatTexelSampler<GS_PSM_T8>{state, clut, tex, m_vram, texW, texH, wrapU, wrapV, minU, maxU, minV, maxV},
                                 state.linearFilter, texUf, texVf);
        case GS_PSM_T4:
            return filterTexture(FormatTexelSampler<GS_PSM_T4>{state, clut, tex, m_vram, texW, texH, wrapU, wrapV, minU, maxU, minV, maxV},
                                 state.linearFilter, texUf, texVf);
        default:
            break;
        }
    }

    const TexelSampler samplePoint{*this, state, clut, tex, texW, texH, wrapU, wrapV, minU, maxU, minV, maxV};
    return filterTexture(samplePoint, state.linearFilter, texUf, texVf);
}

void GSCpuBackend::DrawSprite(const GSPrimitiveBatch &batch, const RowBand &band, const uint16_t *clut)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;
    u32 z1 = static_cast<u32>(v1.z);

    if (x0 > x1)
        std::swap(x0, x1);
    if (y0 > y1)
        std::swap(y0, y1);

    const int unclippedX0 = x0;
    const int unclippedY0 = y0;
    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int unclippedX1 = unclippedX0 + spanX - 1;
    const int unclippedY1 = unclippedY0 + spanY - 1;

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return;

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);
    if (band.index == 0u)
    {
        ++t_drawCounters.sprites;
        t_drawCounters.spritePixels += static_cast<uint64_t>(drawX1 - drawX0 + 1) * static_cast<uint64_t>(drawY1 - drawY0 + 1);
    }

    const uint64_t alphaReg = ctx.alpha;
    const uint8_t alphaMode = static_cast<uint8_t>(alphaReg & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((alphaReg >> 32) & 0xFFu);

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;

    if (state.prim.tme)
    {
        const auto &tex = ctx.tex0;
        const int texW = state.textureWidth;
        const int texH = state.textureHeight;

        float u0f, v0f, u1f, v1f;
        if (state.prim.fst)
        {
            u0f = static_cast<float>(v0.u >> 4);
            v0f = static_cast<float>(v0.v >> 4);
            u1f = static_cast<float>(v1.u >> 4);
            v1f = static_cast<float>(v1.v >> 4);
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }

        static const bool glyphLog = std::getenv("VENDETTA_GSLOG") != nullptr;
        thread_local uint32_t glyphProbeDraws = 0u;
        const bool glyphProbe = glyphLog && band.index == 0u && tex.tbp0 == 0x3FBDu && glyphProbeDraws < 6u;
        if (glyphProbe)
            ++glyphProbeDraws;
        uint32_t glyphProbeCount = 0u;
        float spriteW = static_cast<float>(spanX);
        float spriteH = static_cast<float>(spanY);
        if (spriteW < 1.0f)
            spriteW = 1.0f;
        if (spriteH < 1.0f)
            spriteH = 1.0f;

        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!band.owns(y))
                continue;
            float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / spriteH;
            float texVf = v0f + (v1f - v0f) * ty;

            for (int x = drawX0; x <= drawX1; ++x)
            {
                float tx = (static_cast<float>(x - unclippedX0) + 0.5f) / spriteW;
                float texUf = u0f + (u1f - u0f) * tx;
                uint32_t texel = 0xFFFF00FFu;
                if (state.prim.fst)
                {
                    const int fixedU = static_cast<int>((texUf * 16.0f) + 0.5f);
                    const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                    const uint16_t sampleU = static_cast<uint16_t>(clampInt(fixedU, 0, 0xFFFF));
                    const uint16_t sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
                    texel = SampleTexture(state, clut, 0.0f, 0.0f, 1.0f, sampleU, sampleV);
                }
                else
                {
                    texel = SampleTexture(state, clut, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH), 1.0f, 0u, 0u);
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                WritePixel(state, x, y, z1, color.r, color.g, color.b, color.a, v1.fog);
                if (glyphProbe && color.a > 0x40u && glyphProbeCount < 4u)
                {
                    ++glyphProbeCount;
                    const uint32_t after = ReadVramUnlocked(ctx.frame.psm, GSInternal::framePageBaseToBlock(ctx.frame.fbp),
                                                            std::max<uint32_t>(ctx.frame.fbw, 1u), x, y);
                    std::fprintf(stderr, "[gslog]   glyph pixel (%d,%d) src rgba=%u,%u,%u,%u -> fb=%08x%c", x, y,
                                 color.r, color.g, color.b, color.a, after, 10);
                }
            }
        }
    }
    else
    {
        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!band.owns(y))
                continue;
            for (int x = drawX0; x <= drawX1; ++x)
                WritePixel(state, x, y, z1, r, g, b, a, v1.fog);
        }
    }
}

void GSCpuBackend::DrawTriangle(const GSPrimitiveBatch &batch, const RowBand &band, const uint16_t *clut)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    float fx2 = v2.x - static_cast<float>(ofx);
    float fy2 = v2.y - static_cast<float>(ofy);

    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));

    minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
    maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
    minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
    maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);

    float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return;

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;
    if (band.index == 0u)
    {
        ++t_drawCounters.triangles;
        t_drawCounters.triangleBoxPixels += static_cast<uint64_t>(maxX - minX + 1) * static_cast<uint64_t>(maxY - minY + 1);
    }
    uint64_t visitedPixels = 0u;

    // The barycentric weights are linear in x along a row: w = a*px + b(py). Each row visits only the x range where
    // all three can be >= -epsilon (plus a pixel of margin; the exact test below still decides), not the whole
    // bounding box - about half the pixels visited before.
    const float scale = winding * invAbsDenom;
    const float a0 = (fy1 - fy2) * scale;
    const float a1 = (fy2 - fy0) * scale;
    const float a2 = -a0 - a1;

    for (int y = minY; y <= maxY; ++y)
    {
        if (!band.owns(y))
            continue;
        float py = static_cast<float>(y) + 0.5f;
        const float b0 = ((fx2 - fx1) * (py - fy2)) * scale - a0 * fx2;
        const float b1 = ((fx0 - fx2) * (py - fy2)) * scale - a1 * fx2;
        const float b2 = 1.0f - b0 - b1;
        float lo = static_cast<float>(minX);
        float hi = static_cast<float>(maxX) + 1.0f;
        bool empty = false;
        auto limit = [&](float a, float b)
        {
            // a * px + b >= -epsilon
            if (a > 0.0f)
                lo = std::max(lo, (-kEdgeEpsilon - b) / a);
            else if (a < 0.0f)
                hi = std::min(hi, (-kEdgeEpsilon - b) / a);
            else if (b < -kEdgeEpsilon)
                empty = true;
        };
        limit(a0, b0);
        limit(a1, b1);
        limit(a2, b2);
        if (empty || !(lo <= hi))
            continue;
        const int rowX0 = std::max(minX, static_cast<int>(std::floor(lo - 0.5f)) - 1);
        const int rowX1 = std::min(maxX, static_cast<int>(std::ceil(hi - 0.5f)) + 1);
        for (int x = rowX0; x <= rowX1; ++x)
        {
            float px = static_cast<float>(x) + 0.5f;

            float w0 = (((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
            float w1 = (((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                continue;

            ++visitedPixels;
            double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

            uint8_t r, g, b, a;
            if (state.prim.iip)
            {
                r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (state.prim.tme)
            {
                float is, it, iq;
                uint16_t iu, iv;
                if (state.prim.fst)
                {
                    iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                    iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    // The GS DDA interpolates the homogeneous S, T and Q
                    // values. Texel coordinates are calculated from S/Q and
                    // T/Q only after interpolation.
                    is = v0.s * w0 + v1.s * w1 + v2.s * w2;
                    it = v0.t * w0 + v1.t * w1 + v2.t * w2;
                    iq = v0.q * w0 + v1.q * w1 + v2.q * w2;
                    iu = 0;
                    iv = 0;
                }

                uint32_t texel = SampleTexture(state, clut, is, it, iq, iu, iv);

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const auto &tex = ctx.tex0;
                const uint8_t shadeR = r;
                const uint8_t shadeG = g;
                const uint8_t shadeB = b;
                const uint8_t shadeA = a;
                const TextureCombineResult color = combineTexture(tex, shadeR, shadeG, shadeB, shadeA, tr, tg, tb, ta);

                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            const uint8_t fog = clampU8(static_cast<int>(v0.fog * w0 + v1.fog * w1 + v2.fog * w2));
            WritePixel(state, x, y, static_cast<u32>(z + 0.5), r, g, b, a, fog);
        }
    }
    t_drawCounters.trianglePixels += visitedPixels;
}

void GSCpuBackend::DrawLine(const GSPrimitiveBatch &batch, const RowBand &band)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r, g, b, a;
        if (state.prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        double z = (v0.z + (v1.z - v0.z) * t);
        const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
        if (band.owns(y0))
            WritePixel(state, x0, y0, static_cast<u32>(z), r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
        ++step;
    }
}

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (command.direction != 0u)
        FlushQueueUnlocked(ps2_host_profile::GsFlushTransfer); // local -> local / host read GS memory: everything queued must be drawn first
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;

    // VENDETTA_GSLOG=1: each image transfer the GS is given
    static const bool gsLog = std::getenv("VENDETTA_GSLOG") != nullptr;
    thread_local uint32_t gsLogCount = 0u;
    if (gsLog && gsLogCount < 200000u)
    {
        ++gsLogCount;
        std::fprintf(stderr, "[gslog] transfer dir=%u dbp=0x%x dbw=%u dpsm=0x%x dsax=%u dsay=%u rrw=%u rrh=%u sbp=0x%x sbw=%u\n",
                     static_cast<unsigned>(command.direction), static_cast<unsigned>(command.bitbltbuf.dbp),
                     static_cast<unsigned>(command.bitbltbuf.dbw), static_cast<unsigned>(command.bitbltbuf.dpsm),
                     static_cast<unsigned>(command.trxpos.dsax), static_cast<unsigned>(command.trxpos.dsay),
                     static_cast<unsigned>(command.trxreg.rrw), static_cast<unsigned>(command.trxreg.rrh),
                     static_cast<unsigned>(command.bitbltbuf.sbp), static_cast<unsigned>(command.bitbltbuf.sbw));
    }

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    ps2_host_profile::Scope profile(ps2_host_profile::GsUpload);
    // only when the new pixels land on pages a queued primitive reads (a texture it uses) or draws to
    FlushQueueIfTouchedUnlocked(TransferDestinationPages(), ps2_host_profile::GsFlushUpload);
    {
        static const bool uploadLog = std::getenv("VENDETTA_GSLOG") != nullptr;
        thread_local uint32_t uploadLogs = 0u;
        if (uploadLog && uploadLogs < 4000u && m_transfer.bitbltbuf.dpsm == 0x14u)
        {
            ++uploadLogs;
            std::fprintf(stderr, "[gslog] upload psmt4 bytes=%u dir=%u copied=%u total=%u%c", sizeBytes,
                         static_cast<unsigned>(m_transferState.direction), m_transferState.copiedPixels, m_transferState.totalPixels, 10);
        }
    }
    if (std::getenv("VENDETTA_GSCAPTURE") && m_transfer.bitbltbuf.dbw == 10u && m_transfer.trxreg.rrw == 16u)
    {
        thread_local uint32_t entries = 0u;
        if (entries++ < 12u)
            std::fprintf(stderr, "[gscapture] UploadImage bytes=%u dir=%u total=%u%c", sizeBytes, static_cast<unsigned>(m_transferState.direction), m_transferState.totalPixels, 10);
    }
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    // The texture page cache is a snapshot of one page of local memory: new pixels may land in it (a game that
    // uploads glyph after glyph into one cache slot and draws each — Def Jam Vendetta's menu font).
    m_texturePageCache.Invalidate();
    if (m_transfer.trxreg.rrw == 0u || m_transfer.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
        return;

    const uint32_t dbp = m_transfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = m_transfer.bitbltbuf.dpsm;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t dsax = m_transfer.trxpos.dsax;
    uint32_t offset = 0u;

    // VENDETTA_GSCAPTURE=<file.ppm>: what a full-screen upload in 16-pixel columns (a movie frame: 640-wide CT32,
    // RRW 16) delivers, assembled as sent — saved once the last column (x = 624) is complete.
    static const char *capturePath = std::getenv("VENDETTA_GSCAPTURE");
    thread_local std::vector<uint32_t> capture;
    thread_local uint32_t capturesSaved = 0u;
    if (capturePath && dpsm == GS_PSM_CT32 && dbw == 10u && rrw == 16u && capturesSaved < 3u)
    {
        if (capture.empty())
            capture.assign(640u * 448u, 0u);
        thread_local uint32_t calls = 0u;
        if (calls++ < 12u)
            std::fprintf(stderr, "[gscapture] upload dsax=%u bytes=%u copied=%u total=%u%c", dsax, sizeBytes, m_transferState.copiedPixels, m_transferState.totalPixels, 10);
        for (uint32_t i = 0; i + 3u < sizeBytes; i += 4u)
        {
            const uint32_t pixel = m_transferState.copiedPixels + i / 4u;
            const uint32_t px = dsax + pixel % rrw;
            const uint32_t py = m_transfer.trxpos.dsay + pixel / rrw;
            if (px < 640u && py < 448u)
                std::memcpy(&capture[py * 640u + px], data + i, 4u);
        }
        if (dsax == 624u && m_transferState.copiedPixels + sizeBytes / 4u >= m_transferState.totalPixels)
        {
            const std::string path = std::string(capturePath) + std::to_string(capturesSaved++) + ".ppm";
            if (std::FILE *f = std::fopen(path.c_str(), "wb"))
            {
                std::fprintf(f, "P6%c640 448%c255%c", 10, 10, 10);
                for (const uint32_t c : capture)
                {
                    const uint8_t rgb[3] = {static_cast<uint8_t>(c), static_cast<uint8_t>(c >> 8), static_cast<uint8_t>(c >> 16)};
                    std::fwrite(rgb, 1, 3, f);
                }
                std::fclose(f);
            }
        }
    }

    auto advancePixel = [&](uint32_t count)
    {
        const uint32_t totalPixels = m_transferState.totalPixels;
        m_transferState.copiedPixels =
            std::min<uint32_t>(totalPixels, m_transferState.copiedPixels + count);

        if (m_transferState.copiedPixels >= totalPixels)
        {
            m_transferState.direction = 3u;
            m_transferState.totalPixels = 0u;
            return;
        }

        m_transferState.x = dsax + (m_transferState.copiedPixels % rrw);
        m_transferState.y = m_transfer.trxpos.dsay + (m_transferState.copiedPixels / rrw);
    };

    // the format's write, looked up once (m_vram is set: checked above); WriteVramUnlocked's inlined format switch in
    // this per-pixel loop made uploads slower
    const auto writeUpload = m_writeVramFuncs[dpsm & 0x3Fu];
    while (offset < sizeBytes && m_transferState.direction == 0u)
    {
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            if (sizeBytes - offset < 4u)
                return;
            uint32_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            writeUpload(m_vram, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 4u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            if (sizeBytes - offset < 3u)
                return;
            const uint32_t value = static_cast<uint32_t>(data[offset]) |
                                   (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
                                   (static_cast<uint32_t>(data[offset + 2u]) << 16u);
            writeUpload(m_vram, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 3u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            if (sizeBytes - offset < 2u)
                return;
            uint16_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            writeUpload(m_vram, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 2u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_T8:
        case GS_PSM_T8H:
            writeUpload(m_vram, dbp, dbw, m_transferState.x, m_transferState.y, data[offset++]);
            advancePixel(1u);
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint8_t packed = data[offset++];
            const uint32_t firstPixel = m_transferState.copiedPixels;
            writeUpload(m_vram, dbp, dbw,
                              dsax + (firstPixel % rrw),
                              m_transfer.trxpos.dsay + (firstPixel / rrw),
                              packed & 0x0Fu);
            if (firstPixel + 1u < m_transferState.totalPixels)
            {
                const uint32_t secondPixel = firstPixel + 1u;
                writeUpload(m_vram, dbp, dbw,
                                  dsax + (secondPixel % rrw),
                                  m_transfer.trxpos.dsay + (secondPixel / rrw),
                                  (packed >> 4u) & 0x0Fu);
            }
            advancePixel(std::min<uint32_t>(2u, m_transferState.totalPixels - firstPixel));
            break;
        }
        default:
            return;
        }
    }
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    m_texturePageCache.Invalidate(); // the copy may overwrite the cached texture page
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }

    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushReadback);
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ClearFramebufferWritesPixels(const GSContext &context, bool hasVram)
{
    if (!hasVram || context.frame.fbw == 0u)
        return false;
    return context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24 ||
           context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S;
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushOther);
    if (!m_vram || context.frame.fbw == 0u)
        return false;

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(context.frame.fbw, 1u);
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }
    return false;
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    // Snapshot local memory under the backend lock, then perform the expensive
    // display conversion without holding the producer-side raster lock.
    thread_local std::vector<uint8_t> snapshot;
    SnapshotVram(snapshot);
    if (snapshot.empty())
        return {};

    thread_local GSCpuBackend snapshotBackend{1u};
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

void GSCpuBackend::PresentToMailbox(const GSPresentationRequest &request, GSPresentTicket ticket, GSPresentMailbox &box)
{
    GSPresentSlot &slot = box.writeSlot();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        FlushQueueUnlocked(ps2_host_profile::GsFlushPresent);
        if (spriteLogAt() >= 0.0)
        {
            const double now = secondsSinceStart();
            if (now >= spriteLogAt() && now < spriteLogAt() + 0.05)
                spriteOrderLog().note('P', static_cast<uint32_t>(request.vsyncTick));
        }
        ps2_host_profile::Scope profile(ps2_host_profile::GsPresent);
        if (m_vram && m_vramSize != 0u)
        {
            slot.vram.resize(m_vramSize);
            std::memcpy(slot.vram.data(), m_vram, m_vramSize);
        }
        else
            slot.vram.clear();
    }
    slot.request = request;
    slot.seq = ticket.seq;
    slot.valid = !slot.vram.empty();
    // VENDETTA_GS_HASH=1: a hash of GS memory at every presentation (to compare runs / drawing modes)
    static const bool hashLog = std::getenv("VENDETTA_GS_HASH") != nullptr;
    if (hashLog && slot.valid)
        std::fprintf(stderr, "[gs-hash] seq=%llu tick=%llu flip=%u vram=%016llx%c", static_cast<unsigned long long>(ticket.seq),
                     static_cast<unsigned long long>(request.vsyncTick), ticket.flip ? 1u : 0u,
                     static_cast<unsigned long long>(hashBlock(slot.vram.data(), slot.vram.size())), 10);
    box.publish();
}

uint64_t GSCpuBackend::HashVram(std::array<uint64_t, 512> *pages)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    FlushQueueUnlocked(ps2_host_profile::GsFlushOther);
    if (!m_vram || m_vramSize < GSMem::MEMORY_SIZE)
        return 0u;
    constexpr size_t kPageBytes = 8192u;
    uint64_t hash = 0u;
    for (size_t page = 0; page < 512u; ++page)
    {
        const uint64_t pageHash = hashBlock(m_vram + page * kPageBytes, kPageBytes);
        if (pages)
            (*pages)[page] = pageHash;
        hash = (hash ^ pageHash) * 0x9E3779B97F4A7C15ull + page;
    }
    return hash;
}

void GSCpuBackend::VerifyPages(const std::array<uint64_t, 512> &expected, uint64_t check, uint64_t streamHash)
{
    (void)streamHash;
    std::array<uint64_t, 512> pages{};
    HashVram(&pages);
    ps2_host_profile::count(ps2_host_profile::GsVerifyChecks);
    uint32_t differing = 0u, first = 0u;
    for (uint32_t page = 0; page < 512u; ++page)
        if (pages[page] != expected[page] && differing++ == 0u)
            first = page;
    if (differing == 0u)
        return;
    ps2_host_profile::count(ps2_host_profile::GsVerifyMismatch);
    thread_local uint32_t reports = 0u;
    if (reports++ < 20u)
        std::fprintf(stderr, "[gs-verify] check %llu: %u of 512 GS memory pages differ from the 1-band shadow renderer; first at 0x%06x (page %u)%c",
                     static_cast<unsigned long long>(check), differing, first * 8192u, first, 10);
}

PresentationFrame GSCpuBackend::ConvertSnapshot(const uint8_t *vram, uint32_t vramSize, const GSPresentationRequest &request)
{
    if (!vram || vramSize < GSMem::MEMORY_SIZE)
        return {};
    thread_local GSCpuBackend converter{1u};
    converter.Initialize(const_cast<uint8_t *>(vram), vramSize);
    return converter.PresentFromLocalMemory(request);
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    // FIELD mode (INT = 1, FFMD = 0): the buffer holds the whole frame and a TV shows its even / odd lines on
    // alternate fields. A PC screen shows every line at once (both fields woven) — it used to show one field
    // line-doubled per vblank, which halved the detail and made still pictures shimmer.
    // VENDETTA_GSLOG=1: the display registers whenever they change (what the picture is built from)
    static const bool displayLog = std::getenv("VENDETTA_GSLOG") != nullptr;
    thread_local uint64_t lastDisplayRegs[6] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
    const uint64_t displayRegs[6] = {request.pmode, request.smode2, request.dispfb1, request.display1, request.dispfb2, request.display2};
    if (displayLog && std::memcmp(displayRegs, lastDisplayRegs, sizeof(displayRegs)) != 0)
    {
        std::memcpy(lastDisplayRegs, displayRegs, sizeof(displayRegs));
        std::fprintf(stderr, "[gslog] display pmode=0x%llx smode2=0x%llx dispfb1=0x%llx display1=0x%llx dispfb2=0x%llx display2=0x%llx%c",
                     static_cast<unsigned long long>(request.pmode), static_cast<unsigned long long>(request.smode2),
                     static_cast<unsigned long long>(request.dispfb1), static_cast<unsigned long long>(request.display1),
                     static_cast<unsigned long long>(request.dispfb2), static_cast<unsigned long long>(request.display2), 10);
    }
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    // An interlaced TV shows 448 lines (NTSC) / 512 (PAL) of the display area, whatever DH says (as PCSX2: the
    // video mode's 224 / 256 lines per field × 2). Def Jam Vendetta's movies set DH = 960 for a 640 × 480 buffer;
    // reading 512 lines took 32 lines of the next buffer and squashed the picture.
    if (smode2.interlaced)
    {
        const uint32_t visibleLines = (((request.smode1 >> 13) & 0x3ull) == 3ull) ? 512u : 448u;
        height1 = std::min(height1, visibleLines);
        height2 = std::min(height2, visibleLines);
    }
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    if (!valid1 && !valid2)
        return result;

    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t height,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        selected = displayFrame;
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(displayFrame, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        return true;
    };

    // SMODE2 INT = 1 with FFMD = 1: every field is a frame buffer of DH/2 lines shown on all DH output lines (games
    // that draw 224-line fields into alternating buffers — Def Jam Vendetta's menus). Reading DH lines ran into the
    // next buffer (a second copy of the picture below the first) and squashed the image.
    const bool halfHeightFields = smode2.interlaced && smode2.frameMode;
    // the field buffers shown so far (for the weave below): rotate when the displayed buffer changes
    thread_local GSFrameReg s_curField{}, s_prevField{};
    thread_local bool s_haveCurField = false, s_havePrevField = false;
    if (halfHeightFields)
    {
        const GSFrameReg &shown = valid1 ? displayFrame1 : displayFrame2;
        if (!s_haveCurField || s_curField.fbp != shown.fbp)
        {
            s_prevField = s_curField;
            s_havePrevField = s_haveCurField;
            s_curField = shown;
            s_haveCurField = true;
        }
    }
    auto copyDisplay = [&](const GSFrameReg &displayFrame,
                           const GSDisplayReadOrigin &origin,
                           uint32_t width,
                           uint32_t height,
                           bool allowPreferred,
                           bool preserveAlpha,
                           GSFrameReg &selected,
                           std::vector<uint8_t> &pixels,
                           bool &usedPreferred) -> bool
    {
        const uint32_t readHeight = halfHeightFields ? (height + 1u) / 2u : height;
        if (!copySource(displayFrame, origin, width, readHeight, allowPreferred, preserveAlpha, selected, pixels, usedPreferred))
            return false;
        if (halfHeightFields)
        {
            // Weave (as a TV / PCSX2): this field's buffer on one line parity, the previous field's buffer (the other
            // one of the pair, drawn half a line apart - HCTP) on the other. Line-doubling each field alone made the
            // picture bounce a line up and down every vblank.
            const size_t rowBytes = static_cast<size_t>(kHostFrameWidth) * 4u;
            std::vector<uint8_t> previous;
            if (s_havePrevField && s_prevField.fbp != displayFrame.fbp && s_prevField.fbw == displayFrame.fbw &&
                s_prevField.psm == displayFrame.psm)
            {
                GSFrameReg prevSelected{};
                bool prevPreferred = false;
                if (!copySource(s_prevField, origin, width, readHeight, false, preserveAlpha, prevSelected, previous, prevPreferred))
                    previous.clear();
            }
            // The host presents a complete woven frame, not individual CRT fields.  Do not use
            // the host/vsync counter here: it advances independently of the GS field buffer and
            // made an unchanged movie/menu frame alternate its vertical phase every refresh.
            // The displayed field buffer is already tracked above, so keep a stable weave phase.
            constexpr uint32_t parity = 0u;
            for (uint32_t y = height; y-- > 0u;)
            {
                if ((y + 1u) * rowBytes > pixels.size())
                    continue;
                const bool fromPrevious = !previous.empty() && (y & 1u) != parity && (y / 2u + 1u) * rowBytes <= previous.size();
                if (fromPrevious)
                    std::memcpy(pixels.data() + y * rowBytes, previous.data() + (y / 2u) * rowBytes, rowBytes);
                else if (y / 2u != y)
                    std::memcpy(pixels.data() + y * rowBytes, pixels.data() + (y / 2u) * rowBytes, rowBytes);
            }
        }
        return true;
    };

    if (valid1 && valid2)
    {
        GSFrameReg selected1{}, selected2{};
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copyDisplay(displayFrame1, origin1, width1, height1, false, true, selected1, crt1, preferred1) &&
            copyDisplay(displayFrame2, origin2, width2, height2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (!pmode.slbg)
                for (uint32_t y = 0; y < height2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            for (uint32_t y = 0; y < height1; ++y)
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    const uint32_t factor = pmode.mmod ? pmode.alp : std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u);
                    dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                    dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                    dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    dst[3] = pmode.amod ? dst[3] : src[3];
                }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            return result;
        }
    }

    const GSFrameReg &displayFrame = valid1 ? displayFrame1 : displayFrame2;
    const GSDisplayReadOrigin &origin = valid1 ? origin1 : origin2;
    result.width = valid1 ? width1 : width2;
    result.height = valid1 ? height1 : height2;
    GSFrameReg selected = displayFrame;
    if (!copyDisplay(displayFrame, origin, result.width, result.height, true, false, selected, result.pixels, result.usedPreferred))
        return {};
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return result;
}
