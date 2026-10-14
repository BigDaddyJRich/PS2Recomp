// HCTP native model draws: the p08 shader spec (notes\VU1_PROGRAMS.md "SHADER SPEC") as plain C++.
// HCTP_NATIVE_VERIFY=1: before each later p08 chunk (MSCNT at the loop head) the corners are computed from VU1 data
// memory with the spec's formulas; after the VU1 ran, they are compared with the packet p08 wrote at TOP+33 (per
// corner: STQ, RGBA, XYZF2 incl. the backface ADC bit). Every 5 s: corners compared / matching per field.
// HCTP_NATIVE_P08=1: chunks that need no clipping are drawn from the spec (the packet goes straight to the GS) and
// the VU1 does not run for them; chunks with a CLIP flag stay on the VU1 (p05/p06 near-plane clipper).
#include "hctp_native_vu1.h"

#include "runtime/gs/gs_frontend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
    constexpr uint64_t kP08Hash = 0xcaf9bd881ee1d7ccull; // FNV-1a of p08's first 512 code bytes at VU 0x2a00
    constexpr uint32_t kP08LoopHead = 0x3010u;           // MSCNT for a later chunk resumes here

    struct Corner
    {
        float stq[3];
        int32_t rgba[4];
        int32_t xyz[4]; // XYZF2 as kicked: x, y, z, fog<<4 | ADC
    };

    bool g_on = false;     // HCTP_NATIVE_VERIFY=1
    bool g_native = false; // HCTP_NATIVE_P08=1
    bool g_pending = false;
    bool g_chunkClip = false;
    uint32_t g_top = 0, g_count = 0;
    std::vector<Corner> g_expected;
    std::vector<uint8_t> g_packet;
    uint64_t g_chunks = 0, g_corners = 0, g_okStq = 0, g_okRgba = 0, g_okXyz = 0, g_okAll = 0;
    uint64_t g_clipChunks = 0, g_nativeChunks = 0, g_vuChunks = 0;
    std::chrono::steady_clock::time_point g_last;
    int g_dumped = 0;

    const float *q(const uint8_t *data, uint32_t qw) { return reinterpret_cast<const float *>(data + (qw & 0x3FFu) * 16u); }

    uint32_t cornerIndex(const uint8_t *data, uint32_t top, uint32_t k)
    {
        uint32_t bits;
        std::memcpy(&bits, &q(data, top + 1u + 2u * k)[3], 4);
        return bits & 0xFFFFu;
    }

    uint64_t codeHash(const uint8_t *code)
    {
        uint64_t h = 1469598103934665603ull;
        for (uint32_t i = 0; i < 0x200u; ++i)
            h = (h ^ code[0x2a00u + i]) * 1099511628211ull;
        return h;
    }

    int32_t ftoi(float v, float scale)
    {
        const double s = static_cast<double>(v) * scale;
        if (s >= 2147483647.0)
            return 2147483647;
        if (s <= -2147483648.0)
            return static_cast<int32_t>(0x80000000u);
        return static_cast<int32_t>(s); // truncation, as FTOIn
    }

    // the spec for one p08 chunk -> g_expected, g_chunkClip; false when the overlay is not p08 / not a chunk
    bool computeChunk(const uint8_t *code, const uint8_t *data, const VU1State &st, uint32_t top)
    {
        if (!code || !data || st.pc != kP08LoopHead)
            return false;
        static uint64_t s_hashCode = 0, s_hash = 0;
        // the overlay at 0x2a00 changes rarely compared with the chunk rate: hash only when its first words change
        uint64_t sig;
        std::memcpy(&sig, code + 0x2a10u, 8);
        if (sig != s_hashCode)
        {
            s_hashCode = sig;
            s_hash = codeHash(code);
        }
        if (s_hash != kP08Hash)
            return false;

        uint32_t header;
        std::memcpy(&header, data + (top & 0x3FFu) * 16u, 4);
        const uint32_t count = header & 0x7FFFu;
        if (count == 0u || count > 48u)
            return false;
        const float *v1 = q(data, 1006), *v1007 = q(data, 1007), *mat = q(data, 1019), *scroll = q(data, 1009);
        const float s1[4] = {v1[0] * 255.0f, v1[1] * 255.0f, v1[2] * 255.0f, v1[3] * 128.0f};
        const float s2[4] = {v1007[0] * mat[0], v1007[1] * mat[1], v1007[2] * mat[2], v1007[3] * mat[3]};
        // clip constants as p08's setup left them: vf15.z = -2048, vf15.x = [1008].x - 2047, vf15.w = 2047,
        // vf11.w = 1024; vi8 = backface mode
        const float offXY = st.vf[15][2], offZ = st.vf[15][0];
        const float guard = std::fabs(st.vf[15][3]), band = std::fabs(st.vf[11][3]);
        const int32_t cullMask = st.vi[8];
        g_chunkClip = false;
        g_expected.resize(count);
        for (uint32_t k = 0; k < count; ++k)
        {
            const float *c0 = q(data, top + 1u + 2u * k);
            const float *c1 = q(data, top + 2u + 2u * k);
            const uint32_t idx = cornerIndex(data, top, k);
            const float *P = q(data, idx + 320u), *N = q(data, idx + 160u), *Lc = q(data, idx + 480u);
            const float light[4] = {Lc[0], Lc[1], Lc[2], 1.0f};
            Corner &o = g_expected[k];
            for (int c = 0; c < 4; ++c)
            {
                float col = c1[c] * s1[c] * light[c] + c1[c] * s2[c];
                col = std::fmin(col, 255.0f);
                o.rgba[c] = ftoi(col, 1.0f);
            }
            const float st3[3] = {c0[0] + scroll[0], c0[1] + scroll[1], c0[2]};
            for (int c = 0; c < 3; ++c)
                o.stq[c] = st3[c] * N[3];
            for (int c = 0; c < 4; ++c)
                o.xyz[c] = ftoi(P[c], 16.0f);
            // CLIP vs 1024 (x/y: needs the clipper) and vs 2047 (z-): any flag -> the chunk stays on the VU1
            const float cx = P[0] + offXY, cy = P[1] + offXY, cz = q(data, idx)[2] + offZ;
            if (std::fabs(cx) > band || std::fabs(cy) > band || cz < -guard)
                g_chunkClip = true;
            // backface (3rd corner on): cross z of (P[k-2]-P[k-1]) x (P[k-1]-P[k]), sign + strip parity, & vi8 -> ADC
            if (k >= 2u)
            {
                const float *P0 = q(data, cornerIndex(data, top, k - 2u) + 320u);
                const float *P1 = q(data, cornerIndex(data, top, k - 1u) + 320u);
                const float ax = P0[0] - P1[0], ay = P0[1] - P1[1], bx = P1[0] - P[0], by = P1[1] - P[1];
                const float crossZ = ax * by - bx * ay;
                const int32_t v = (crossZ < 0.0f ? 32 : 0) + (((k - 2u) & 1u) ? 32 : 0);
                if (v & cullMask)
                    o.xyz[3] = (o.xyz[3] & 0xFFFF) | 0x8000;
            }
        }
        g_top = top;
        g_count = count;
        return true;
    }

    void report()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - g_last <= std::chrono::seconds(5))
            return;
        g_last = now;
        if (g_corners)
            std::fprintf(stderr, "[nvu1] p08 clipChunks=%llu chunks=%llu corners=%llu match: all %.2f%% stq %.2f%% rgba %.2f%% xyzf %.2f%%%c",
                         static_cast<unsigned long long>(g_clipChunks), static_cast<unsigned long long>(g_chunks),
                         static_cast<unsigned long long>(g_corners), 100.0 * g_okAll / g_corners,
                         100.0 * g_okStq / g_corners, 100.0 * g_okRgba / g_corners, 100.0 * g_okXyz / g_corners, 10);
        if (g_nativeChunks)
            std::fprintf(stderr, "[nvu1] p08 native chunks=%llu vu (clip) chunks=%llu%c",
                         static_cast<unsigned long long>(g_nativeChunks), static_cast<unsigned long long>(g_vuChunks), 10);
    }
}

void hctpNativeVu1Init()
{
    const char *v = std::getenv("HCTP_NATIVE_VERIFY");
    g_on = v && v[0] == '1';
    const char *n = std::getenv("HCTP_NATIVE_P08");
    g_native = n && n[0] == '1' && !g_on;
    if (g_on || g_native)
        std::fprintf(stderr, "[nvu1] p08 spec verify %d native %d%c", g_on ? 1 : 0, g_native ? 1 : 0, 10);
}

bool hctpNativeVu1TryDraw(const uint8_t *code, uint8_t *data, const VU1State &st, uint32_t top, GS &gs)
{
    if (!g_native || !computeChunk(code, data, st, top))
        return false;
    if (g_chunkClip)
    {
        ++g_vuChunks;
        return false;
    }
    // the packet p08 would kick from TOP+33: the chunk's own GIF tag, then STQ / RGBA / XYZF2 per corner
    const uint32_t qwords = 1u + 3u * g_count;
    g_packet.resize(qwords * 16u);
    uint8_t *o = g_packet.data();
    std::memcpy(o, data + (top & 0x3FFu) * 16u, 16);
    for (uint32_t k = 0; k < g_count; ++k)
    {
        const Corner &e = g_expected[k];
        uint8_t *c = o + (1u + 3u * k) * 16u;
        const float stq[4] = {e.stq[0], e.stq[1], e.stq[2], 0.0f};
        std::memcpy(c, stq, 16);
        std::memcpy(c + 16, e.rgba, 16);
        std::memcpy(c + 32, e.xyz, 16);
    }
    // VU memory as the VU would leave it
    for (uint32_t i = 0; i < qwords; ++i)
        std::memcpy(data + ((top + 33u + i) & 0x3FFu) * 16u, o + i * 16u, 16);
    gs.processGIFPacket(o, qwords * 16u, 1u);
    ++g_nativeChunks;
    report();
    return true;
}

void hctpNativeVu1BeforeResume(const uint8_t *code, const uint8_t *data, const VU1State &st, uint32_t top)
{
    g_pending = false;
    if (!g_on)
        return;
    g_pending = computeChunk(code, data, st, top);
    if (g_pending && g_chunkClip)
    {
        ++g_clipChunks;
        g_pending = false; // clipped chunks stay on the VU1; only native-eligible ones are scored
    }
}

void hctpNativeVu1AfterResume(const uint8_t *data)
{
    if (!g_pending || !data)
        return;
    g_pending = false;
    ++g_chunks;
    for (uint32_t k = 0; k < g_count; ++k)
    {
        const uint8_t *out = data + ((g_top + 34u + 3u * k) & 0x3FFu) * 16u;
        float stq[3];
        int32_t rgba[4], xyz[4];
        std::memcpy(stq, out, 12);
        std::memcpy(rgba, out + 16, 16);
        std::memcpy(xyz, out + 32, 16);
        const Corner &e = g_expected[k];
        bool okStq = true, okRgba = true, okXyz = true;
        for (int c = 0; c < 3; ++c)
            okStq = okStq && std::fabs(stq[c] - e.stq[c]) <= 1e-4f * std::fmax(1.0f, std::fabs(e.stq[c]));
        for (int c = 0; c < 4; ++c)
            okRgba = okRgba && std::abs(rgba[c] - e.rgba[c]) <= 1;
        for (int c = 0; c < 3; ++c)
            okXyz = okXyz && std::abs(xyz[c] - e.xyz[c]) <= 1;
        okXyz = okXyz && (xyz[3] & 0x8000) == (e.xyz[3] & 0x8000) && std::abs((xyz[3] & 0x7FFF) - (e.xyz[3] & 0x7FFF)) <= 16;
        ++g_corners;
        g_okStq += okStq;
        g_okRgba += okRgba;
        g_okXyz += okXyz;
        g_okAll += okStq && okRgba && okXyz;
        if (!(okStq && okRgba && okXyz) && g_dumped < 12)
        {
            ++g_dumped;
            std::fprintf(stderr, "[nvu1] mismatch top=0x%x k=%u: stq vu %g %g %g / spec %g %g %g | rgba vu %d %d %d %d / spec %d %d %d %d"
                                 " | xyzf vu %d %d %d %x / spec %d %d %d %x%c",
                         g_top, k, stq[0], stq[1], stq[2], e.stq[0], e.stq[1], e.stq[2], rgba[0], rgba[1], rgba[2], rgba[3],
                         e.rgba[0], e.rgba[1], e.rgba[2], e.rgba[3], xyz[0], xyz[1], xyz[2], xyz[3], e.xyz[0], e.xyz[1],
                         e.xyz[2], e.xyz[3], 10);
        }
    }
    report();
}
