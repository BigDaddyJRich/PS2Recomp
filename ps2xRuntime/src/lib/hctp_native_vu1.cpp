// HCTP native model draws - step 1: the p08 shader spec (notes\VU1_PROGRAMS.md "SHADER SPEC") as plain C++, checked
// against the emulated VU1 (HCTP_NATIVE_VERIFY=1). Before each later p08 chunk (MSCNT at the loop head) the corners
// are computed from VU1 data memory with the spec's formulas; after the VU1 ran, they are compared with the packet
// p08 wrote at TOP+33 (per corner: STQ, RGBA, XYZF). Every 5 s: corners compared / matching per field. When this
// reports (near) 100%, the same math becomes the Direct3D 11 vertex shader.
#include "hctp_native_vu1.h"

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
        int32_t xyz[3];
    };

    bool g_on = false;
    bool g_pending = false;
    uint32_t g_top = 0, g_count = 0;
    std::vector<Corner> g_expected;
    uint64_t g_chunks = 0, g_corners = 0, g_okStq = 0, g_okRgba = 0, g_okXyz = 0, g_okAll = 0;
    std::chrono::steady_clock::time_point g_last;
    int g_dumped = 0;

    const float *q(const uint8_t *data, uint32_t qw) { return reinterpret_cast<const float *>(data + (qw & 0x3FFu) * 16u); }

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
}

void hctpNativeVu1Init()
{
    const char *v = std::getenv("HCTP_NATIVE_VERIFY");
    g_on = v && v[0] == '1';
    if (g_on)
        std::fprintf(stderr, "[nvu1] p08 spec verify on%c", 10);
}

void hctpNativeVu1BeforeResume(const uint8_t *code, const uint8_t *data, uint32_t pc, uint32_t top)
{
    g_pending = false;
    if (!g_on || !code || !data || pc != kP08LoopHead)
        return;
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
        return;

    uint32_t header;
    std::memcpy(&header, data + (top & 0x3FFu) * 16u, 4);
    const uint32_t count = header & 0x7FFFu;
    if (count == 0u || count > 48u)
        return;
    const float *v1 = q(data, 1006), *v1007 = q(data, 1007), *mat = q(data, 1019), *scroll = q(data, 1009);
    const float s1[4] = {v1[0] * 255.0f, v1[1] * 255.0f, v1[2] * 255.0f, v1[3] * 128.0f};
    const float s2[4] = {v1007[0] * mat[0], v1007[1] * mat[1], v1007[2] * mat[2], v1007[3] * mat[3]};
    g_expected.resize(count);
    for (uint32_t k = 0; k < count; ++k)
    {
        const float *c0 = q(data, top + 1u + 2u * k);
        const float *c1 = q(data, top + 2u + 2u * k);
        uint32_t idxBits;
        std::memcpy(&idxBits, &c0[3], 4);
        const uint32_t idx = idxBits & 0xFFFFu;
        const float *P = q(data, idx + 320u), *N = q(data, idx + 160u), *Lc = q(data, idx + 480u);
        const float light[4] = {Lc[0], Lc[1], Lc[2], 1.0f};
        Corner &o = g_expected[k];
        for (int c = 0; c < 4; ++c)
        {
            float col = c1[c] * s1[c] * light[c] + c1[c] * s2[c];
            col = std::fmin(col, 255.0f);
            o.rgba[c] = ftoi(col, 1.0f);
        }
        const float st[3] = {c0[0] + scroll[0], c0[1] + scroll[1], c0[2]};
        for (int c = 0; c < 3; ++c)
            o.stq[c] = st[c] * N[3];
        for (int c = 0; c < 3; ++c)
            o.xyz[c] = ftoi(P[c], 16.0f);
    }
    g_top = top;
    g_count = count;
    g_pending = true;
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
        int32_t rgba[4], xyz[3];
        std::memcpy(stq, out, 12);
        std::memcpy(rgba, out + 16, 16);
        std::memcpy(xyz, out + 32, 12);
        const Corner &e = g_expected[k];
        bool okStq = true, okRgba = true, okXyz = true;
        for (int c = 0; c < 3; ++c)
            okStq = okStq && std::fabs(stq[c] - e.stq[c]) <= 1e-4f * std::fmax(1.0f, std::fabs(e.stq[c]));
        for (int c = 0; c < 4; ++c)
            okRgba = okRgba && std::abs(rgba[c] - e.rgba[c]) <= 1;
        for (int c = 0; c < 3; ++c)
            okXyz = okXyz && std::abs(xyz[c] - e.xyz[c]) <= 1;
        ++g_corners;
        g_okStq += okStq;
        g_okRgba += okRgba;
        g_okXyz += okXyz;
        g_okAll += okStq && okRgba && okXyz;
        if (!(okStq && okRgba && okXyz) && g_dumped < 12)
        {
            ++g_dumped;
            std::fprintf(stderr, "[nvu1] mismatch top=0x%x k=%u: stq vu %g %g %g / spec %g %g %g | rgba vu %d %d %d %d / spec %d %d %d %d"
                                 " | xyz vu %d %d %d / spec %d %d %d%c",
                         g_top, k, stq[0], stq[1], stq[2], e.stq[0], e.stq[1], e.stq[2], rgba[0], rgba[1], rgba[2], rgba[3],
                         e.rgba[0], e.rgba[1], e.rgba[2], e.rgba[3], xyz[0], xyz[1], xyz[2], e.xyz[0], e.xyz[1], e.xyz[2], 10);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - g_last > std::chrono::seconds(5) && g_corners)
    {
        g_last = now;
        std::fprintf(stderr, "[nvu1] p08 chunks=%llu corners=%llu match: all %.2f%% stq %.2f%% rgba %.2f%% xyz %.2f%%%c",
                     static_cast<unsigned long long>(g_chunks), static_cast<unsigned long long>(g_corners),
                     100.0 * g_okAll / g_corners, 100.0 * g_okStq / g_corners, 100.0 * g_okRgba / g_corners,
                     100.0 * g_okXyz / g_corners, 10);
    }
}
