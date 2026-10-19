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
    // p18 (same overlay slot): p08 without the colour-constant scaling, alpha from the vertex light .w, ST not
    // multiplied by q (affine: STQ = (S + scroll, T + scroll, corner Q)); chunks that need clipping stay on the VU1
    constexpr uint64_t kP18Hash = 0x0aa9d7fc6939ef27ull;
    constexpr uint32_t kP18LoopHead = 0x2e30u;
    // p10 (resident at VU 0x3740, next to p08): p08 with reflection-mapped ST from the vertex normal:
    // S = [1009].x + n.x * [1009].z, T = [1009].y + n.y * [1009].w, STQ = (S, T, 1) * q
    constexpr uint64_t kP10Hash = 0x76528d9e4a4a138full;
    constexpr uint32_t kP10LoopHead = 0x3d80u;
    int g_variant = 0; // 0 = p08, 1 = p18, 2 = p10 (the chunk computeChunk last accepted)
    bool g_p18 = false; // HCTP_NATIVE_P18=1
    bool g_p10 = false; // HCTP_NATIVE_P10=1
    uint64_t g_p10Seen = 0;
    int g_p18PcLog = 0;
    uint64_t g_p18Seen = 0, g_p18Native = 0; // p18 chunks accepted by computeChunk / drawn natively

    struct Corner
    {
        float stq[3];
        int32_t rgba[4];
        int32_t xyz[4]; // XYZF2 as kicked: x, y, z, fog<<4 | ADC
        // for the native clipper: clip-space position, view z + near offset, unclamped inputs of the outputs
        float clip[4];
        float nearD;
        float col[4];
        float st[3];
        float fog;
        bool flagged; // any CLIP flag (p08 would send its triangles to p05/p06)
    };

    // a clipped polygon vertex
    struct CV
    {
        float c[4], nearD, col[4], st[3], fog;
    };

    bool g_on = false;     // HCTP_NATIVE_VERIFY=1
    bool g_native = false; // HCTP_NATIVE_P08=1
    bool g_pending = false;
    bool g_chunkClip = false;
    uint32_t g_top = 0, g_count = 0;
    std::vector<Corner> g_expected;
    std::vector<uint8_t> g_packet;
    uint64_t g_chunks = 0, g_corners = 0, g_okStq = 0, g_okRgba = 0, g_okXyz = 0, g_okAll = 0;
    uint64_t g_clipChunks = 0, g_nativeChunks = 0, g_vuChunks = 0, g_clipTris = 0;
    uint64_t g_projN = 0, g_projOk = 0;
    std::vector<uint8_t> g_clipPacket;
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
        if (!code || !data)
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
        if (s_hash == kP18Hash && g_p18 && g_p18PcLog < 12)
        {
            ++g_p18PcLog;
            std::fprintf(stderr, "[nvu1] p18 candidate pc=0x%x top=0x%x%c", st.pc, top, 10);
        }
        if (st.pc == kP10LoopHead)
        {
            static uint64_t s_sig10 = 0, s_hash10 = 0;
            uint64_t sig10;
            std::memcpy(&sig10, code + 0x3750u, 8);
            if (sig10 != s_sig10)
            {
                s_sig10 = sig10;
                uint64_t h = 1469598103934665603ull;
                for (uint32_t i = 0; i < 0x200u; ++i)
                    h = (h ^ code[0x3740u + i]) * 1099511628211ull;
                s_hash10 = h;
            }
            if (s_hash10 != kP10Hash || !(g_p10 || g_on))
                return false;
            g_variant = 2;
            ++g_p10Seen;
        }
        else if (st.pc != kP08LoopHead && st.pc != kP18LoopHead)
            return false;
        else if (s_hash == kP08Hash && st.pc == kP08LoopHead)
            g_variant = 0;
        else if (s_hash == kP18Hash && st.pc == kP18LoopHead && (g_p18 || g_on))
        {
            g_variant = 1;
            ++g_p18Seen;
        }
        else
            return false;
        const bool p18 = g_variant == 1;

        uint32_t header;
        std::memcpy(&header, data + (top & 0x3FFu) * 16u, 4);
        const uint32_t count = header & 0x7FFFu;
        if (count == 0u || count > 48u)
            return false;
        const float *v1 = q(data, 1006), *v1007 = q(data, 1007), *mat = q(data, 1019), *scroll = q(data, 1009);
        const float s1[4] = {p18 ? v1[0] : v1[0] * 255.0f, p18 ? v1[1] : v1[1] * 255.0f, p18 ? v1[2] : v1[2] * 255.0f,
                             p18 ? v1[3] : v1[3] * 128.0f};
        const float s2[4] = {v1007[0] * mat[0], v1007[1] * mat[1], v1007[2] * mat[2], p18 ? 0.0f : v1007[3] * mat[3]};
        // clip constants as p08's setup left them: vf15.z = -2048, vf15.x = [1008].x - 2047, vf15.w = 2047,
        // vf11.w = 1024; vi8 = backface mode
        const float offXY = st.vf[15][2], offZ = st.vf[15][0];
        const float guard = std::fabs(st.vf[15][3]), band = std::fabs(st.vf[11][3]);
        const int32_t cullMask = st.vi[8];
        const float *pm[4] = {q(data, 1020), q(data, 1021), q(data, 1022), q(data, 1023)}; // vf26..vf29
        const float nearOff = offZ + guard; // [1008].x: keep view z + [1008].x >= 0
        g_chunkClip = false;
        g_expected.resize(count);
        for (uint32_t k = 0; k < count; ++k)
        {
            const float *c0 = q(data, top + 1u + 2u * k);
            const float *c1 = q(data, top + 2u + 2u * k);
            const uint32_t idx = cornerIndex(data, top, k);
            const float *P = q(data, idx + 320u), *N = q(data, idx + 160u), *Lc = q(data, idx + 480u);
            const float light[4] = {Lc[0], Lc[1], Lc[2], p18 ? Lc[3] : 1.0f};
            Corner &o = g_expected[k];
            for (int c = 0; c < 4; ++c)
            {
                float col = c1[c] * s1[c] * light[c] + c1[c] * s2[c];
                col = std::fmin(col, 255.0f);
                o.rgba[c] = ftoi(col, 1.0f);
                o.col[c] = col;
            }
            const float *Nn = q(data, cornerIndex(data, top, k) + 160u);
            const float st3[3] = {g_variant == 2 ? scroll[0] + Nn[0] * scroll[2] : c0[0] + scroll[0],
                                  g_variant == 2 ? scroll[1] + Nn[1] * scroll[3] : c0[1] + scroll[1], g_variant == 2 ? 1.0f : c0[2]};
            for (int c = 0; c < 3; ++c)
            {
                o.stq[c] = p18 ? st3[c] : st3[c] * N[3];
                o.st[c] = st3[c];
            }
            const float *V = q(data, idx);
            for (int c = 0; c < 4; ++c)
                o.clip[c] = pm[0][c] * V[0] + pm[1][c] * V[1] + pm[2][c] * V[2] + pm[3][c] * V[3];
            o.nearD = V[2] + nearOff;
            o.fog = P[3];
            for (int c = 0; c < 4; ++c)
                o.xyz[c] = ftoi(P[c], 16.0f);
            // CLIP vs 1024 (x/y: needs the clipper) and vs 2047 (z-): any flag -> the chunk stays on the VU1
            const float cx = P[0] + offXY, cy = P[1] + offXY, cz = q(data, idx)[2] + offZ;
            o.flagged = std::fabs(cx) > band || std::fabs(cy) > band || cz < -guard;
            if (o.flagged)
                g_chunkClip = true;
            else if (g_on && o.clip[3] != 0.0f)
            {
                ++g_projN;
                g_projOk += std::fabs(o.clip[0] / o.clip[3] - P[0]) < 0.05f && std::fabs(o.clip[1] / o.clip[3] - P[1]) < 0.05f;
            }
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

    // Sutherland-Hodgman against: view z + [1008].x >= 0, w > 0, 1024 <= x/w <= 3072, 1024 <= y/w <= 3072
    float planeDist(const CV &v, int p)
    {
        switch (p)
        {
        case 0: return v.nearD;
        case 1: return v.c[3] - 1e-5f;
        case 2: return v.c[0] - 1024.0f * v.c[3];
        case 3: return 3072.0f * v.c[3] - v.c[0];
        case 4: return v.c[1] - 1024.0f * v.c[3];
        default: return 3072.0f * v.c[3] - v.c[1];
        }
    }

    CV lerp(const CV &a, const CV &b, float t)
    {
        CV r;
        for (int i = 0; i < 4; ++i)
        {
            r.c[i] = a.c[i] + (b.c[i] - a.c[i]) * t;
            r.col[i] = a.col[i] + (b.col[i] - a.col[i]) * t;
        }
        for (int i = 0; i < 3; ++i)
            r.st[i] = a.st[i] + (b.st[i] - a.st[i]) * t;
        r.nearD = a.nearD + (b.nearD - a.nearD) * t;
        r.fog = a.fog + (b.fog - a.fog) * t;
        return r;
    }

    void putQ(std::vector<uint8_t> &out, const void *p)
    {
        const uint8_t *b = static_cast<const uint8_t *>(p);
        out.insert(out.end(), b, b + 16);
    }

    // the flagged triangles of the chunk (ADC set in the main packet), clipped and appended to g_clipPacket as
    // triangle fans with their own tags; then PRIM restored. False when the chunk's tag does not set PRIM (PRE).
    bool buildClipPacket(const uint8_t *data, const VU1State &st, uint32_t top)
    {
        uint64_t tag[2];
        std::memcpy(tag, data + (top & 0x3FFu) * 16u, 16);
        if (!((tag[0] >> 46) & 1u))
            return false;
        const uint64_t prim = (tag[0] >> 47) & 0x7FFu;
        const uint64_t fanPrim = (prim & ~7ull) | 5ull;
        const int32_t cullMask = st.vi[8];
        g_clipPacket.clear();
        std::vector<CV> poly, next;
        for (uint32_t k = 2; k < g_count; ++k)
        {
            Corner *tri[3] = {&g_expected[k - 2u], &g_expected[k - 1u], &g_expected[k]};
            if (!tri[0]->flagged && !tri[1]->flagged && !tri[2]->flagged)
                continue;
            g_expected[k].xyz[3] |= 0x8000; // the strip skips it; the clipped polygon replaces it
            poly.clear();
            for (Corner *c : tri)
            {
                CV v;
                std::memcpy(v.c, c->clip, sizeof(v.c));
                std::memcpy(v.col, c->col, sizeof(v.col));
                std::memcpy(v.st, c->st, sizeof(v.st));
                v.nearD = c->nearD;
                v.fog = c->fog;
                poly.push_back(v);
            }
            for (int p = 0; p < 6 && poly.size() >= 3u; ++p)
            {
                next.clear();
                for (size_t i = 0; i < poly.size(); ++i)
                {
                    const CV &a = poly[i], &b = poly[(i + 1u) % poly.size()];
                    const float da = planeDist(a, p), db = planeDist(b, p);
                    if (da >= 0.0f)
                        next.push_back(a);
                    if ((da >= 0.0f) != (db >= 0.0f))
                        next.push_back(lerp(a, b, da / (da - db)));
                }
                poly.swap(next);
            }
            if (poly.size() < 3u)
                continue;
            // backface as p08 decides it, on the clipped polygon's screen positions (clipping keeps the winding)
            float sx[3], sy[3];
            for (int i = 0; i < 3; ++i)
            {
                sx[i] = poly[i].c[0] / poly[i].c[3];
                sy[i] = poly[i].c[1] / poly[i].c[3];
            }
            const float crossZ = (sx[0] - sx[1]) * (sy[1] - sy[2]) - (sx[1] - sx[2]) * (sy[0] - sy[1]);
            const int32_t v = (crossZ < 0.0f ? 32 : 0) + (((k - 2u) & 1u) ? 32 : 0);
            if (v & cullMask)
                continue;
            const uint64_t t[2] = {(tag[0] & ~(0x7FFFull | (1ull << 15) | (0x7FFull << 47))) | poly.size() | (1ull << 46) |
                                       (fanPrim << 47),
                                   tag[1]};
            putQ(g_clipPacket, t);
            for (const CV &cv : poly)
            {
                const float qv = 1.0f / cv.c[3];
                const float stq[4] = {cv.st[0] * qv, cv.st[1] * qv, cv.st[2] * qv, 0.0f};
                int32_t rgba[4], xyzf[4];
                for (int c = 0; c < 4; ++c)
                    rgba[c] = ftoi(std::fmin(cv.col[c], 255.0f), 1.0f);
                for (int c = 0; c < 3; ++c)
                    xyzf[c] = ftoi(cv.c[c] * qv, 16.0f);
                xyzf[3] = ftoi(std::fmin(std::fmax(cv.fog, 0.0f), 255.0f), 16.0f);
                putQ(g_clipPacket, stq);
                putQ(g_clipPacket, rgba);
                putQ(g_clipPacket, xyzf);
            }
            ++g_clipTris;
        }
        // PRIM back to the chunk's own (A+D), EOP
        const uint64_t ad[2] = {1ull | (1ull << 15) | (1ull << 60), 0xEull};
        const uint64_t primReg[2] = {prim, 0x00ull};
        putQ(g_clipPacket, ad);
        putQ(g_clipPacket, primReg);
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
        if (g_projN)
            std::fprintf(stderr, "[nvu1] p08 clip-space check: %.2f%% of %llu corners%c", 100.0 * g_projOk / g_projN,
                         static_cast<unsigned long long>(g_projN), 10);
        if (g_p10Seen)
            std::fprintf(stderr, "[nvu1] p10 chunks seen=%llu%c", static_cast<unsigned long long>(g_p10Seen), 10);
        if (g_p18Seen)
            std::fprintf(stderr, "[nvu1] p18 chunks seen=%llu native=%llu%c", static_cast<unsigned long long>(g_p18Seen),
                         static_cast<unsigned long long>(g_p18Native), 10);
        if (g_nativeChunks)
            std::fprintf(stderr, "[nvu1] p08 native chunks=%llu (clipped tris %llu) vu chunks=%llu%c",
                         static_cast<unsigned long long>(g_nativeChunks), static_cast<unsigned long long>(g_clipTris),
                         static_cast<unsigned long long>(g_vuChunks), 10);
    }
}

void hctpNativeVu1Init()
{
    const char *v = std::getenv("HCTP_NATIVE_VERIFY");
    g_on = v && v[0] == '1';
    const char *n = std::getenv("HCTP_NATIVE_P08");
    g_native = n && n[0] == '1' && !g_on;
    const char *n18 = std::getenv("HCTP_NATIVE_P18");
    g_p18 = n18 && n18[0] == '1';
    const char *n10 = std::getenv("HCTP_NATIVE_P10");
    g_p10 = n10 && n10[0] == '1';
    if (g_on || g_native)
        std::fprintf(stderr, "[nvu1] p08 spec verify %d native %d%c", g_on ? 1 : 0, g_native ? 1 : 0, 10);
}

bool hctpNativeVu1TryDraw(const uint8_t *code, uint8_t *data, const VU1State &st, uint32_t top, GS &gs)
{
    if (!g_native || !computeChunk(code, data, st, top))
        return false;
    // HCTP_NATIVE_P08_CLIP=0: clipped chunks stay on the VU1 (p05/p06)
    static const char *s_clipEnv = std::getenv("HCTP_NATIVE_P08_CLIP");
    static const bool s_clip = !(s_clipEnv && s_clipEnv[0] == '0');
    if (g_chunkClip && (g_variant != 0 || !s_clip || !buildClipPacket(data, st, top)))
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
    if (g_chunkClip)
        gs.processGIFPacket(g_clipPacket.data(), static_cast<uint32_t>(g_clipPacket.size()), 1u);
    ++g_nativeChunks;
    if (g_variant == 1)
        ++g_p18Native;
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
            std::fprintf(stderr, "[nvu1] mismatch p%d top=0x%x k=%u: stq vu %g %g %g / spec %g %g %g | rgba vu %d %d %d %d / spec %d %d %d %d"
                                 " | xyzf vu %d %d %d %x / spec %d %d %d %x%c",
                         g_variant == 1 ? 18 : g_variant == 2 ? 10 : 8, g_top, k, stq[0], stq[1], stq[2], e.stq[0], e.stq[1], e.stq[2], rgba[0], rgba[1], rgba[2], rgba[3],
                         e.rgba[0], e.rgba[1], e.rgba[2], e.rgba[3], xyz[0], xyz[1], xyz[2], xyz[3], e.xyz[0], e.xyz[1],
                         e.xyz[2], e.xyz[3], 10);
        }
    }
    report();
}

// ---------------------------------------------------------------------------------------------------------------
// p01 (VU 0x00a0): 4-bone skinned transform + 3 directional lights, as C++ (notes\VU1_PROGRAMS.md, p01 corrections:
// the normal is normalised when [1010].x == 0, and the light matrix is applied by columns).
// HCTP_NATIVE_VERIFY=1 compares it with the VU1's results; HCTP_NATIVE_P01=1 runs it instead of the VU1.
namespace
{
    struct Stage1Prog
    {
        uint32_t start, end;
        int bones;      // matrices at 801 + 4k; 1 = rigid (weight 1)
        bool weights;   // blend weights at b+640
        bool normalise; // honours [1010].x == 0
    };
    constexpr Stage1Prog kStage1[4] = {
        {0x00a0u, 0x04c0u, 4, true, true},
        {0x05a0u, 0x0a40u, 3, true, true},
        {0x0aa0u, 0x0ea8u, 2, true, true},
        {0x0fa0u, 0x1280u, 1, false, false},
    };
    int g_s1 = -1; // which program the current batch is
    bool g_p01Native = false;
    uint64_t g_p01Hash = 0;
    uint64_t g_s1Verts[4] = {0, 0, 0, 0}, g_s1Ok[4] = {0, 0, 0, 0};
    uint64_t g_p01Batches = 0, g_p01Verts = 0, g_p01Ok[4] = {0, 0, 0, 0}, g_p01NativeBatches = 0;
    int g_p01Dumped = 0;
    std::vector<float> g_p01Out; // per vertex: pos4, normal4, screen4, light4
    uint32_t g_p01Base = 0, g_p01Count = 0;
    bool g_p01Pending = false;
    std::chrono::steady_clock::time_point g_p01Last;

    uint64_t p01Hash(const uint8_t *code)
    {
        uint64_t h = 1469598103934665603ull;
        for (uint32_t i = kStage1[g_s1].start; i < kStage1[g_s1].end; ++i)
            h = (h ^ code[i]) * 1099511628211ull;
        return h;
    }

    // computes the batch into g_p01Out; false = not a p01 batch we handle
    bool p01Compute(const uint8_t *data)
    {
        uint32_t w800[4];
        std::memcpy(w800, data + 800u * 16u, 16);
        const uint32_t count = w800[0] & 0xFFFFu, b = w800[1] & 0xFFFFu;
        if (count == 0u || count > 160u || b + 640u + count > 1024u)
            return false;
        const float *hdr = q(data, 800);
        const float fogA = hdr[2], fogB = hdr[3];
        uint32_t w1010;
        std::memcpy(&w1010, data + 1010u * 16u, 4);
        const Stage1Prog &prog = kStage1[g_s1];
        const bool normalise = prog.normalise && (w1010 & 0xFFFFu) == 0u;
        const float *L[3] = {q(data, 1012), q(data, 1013), q(data, 1014)};
        const float *C[3] = {q(data, 1016), q(data, 1017), q(data, 1018)};
        const float *P[4] = {q(data, 1020), q(data, 1021), q(data, 1022), q(data, 1023)};
        g_p01Out.resize(static_cast<size_t>(count) * 16u);
        for (uint32_t i = 0; i < count; ++i)
        {
            static const float kOne[4] = {1.0f, 0.0f, 0.0f, 0.0f};
            const float *pos = q(data, b + i), *wt = prog.weights ? q(data, b + 640u + i) : kOne, *nrm = q(data, b + 160u + i);
            float v[4] = {0, 0, 0, 0}, n[3] = {0, 0, 0};
            for (int k = 0; k < prog.bones; ++k)
            {
                const float *M[4] = {q(data, 801u + 4u * k), q(data, 802u + 4u * k), q(data, 803u + 4u * k), q(data, 804u + 4u * k)};
                for (int c = 0; c < 4; ++c)
                {
                    const float sk = M[0][c] * pos[0] + M[1][c] * pos[1] + M[2][c] * pos[2] + M[3][c] * pos[3];
                    v[c] += sk * wt[k];
                }
                for (int c = 0; c < 3; ++c)
                {
                    const float sn = M[0][c] * nrm[0] + M[1][c] * nrm[1] + M[2][c] * nrm[2];
                    n[c] += sn * wt[k];
                }
            }
            float cl[4];
            for (int c = 0; c < 4; ++c)
                cl[c] = P[0][c] * v[0] + P[1][c] * v[1] + P[2][c] * v[2] + P[3][c] * v[3];
            const float qv = 1.0f / cl[3];
            const float fog = std::fmin(std::fmax(fogA + cl[3] * fogB, 0.0f), 255.0f);
            if (normalise)
            {
                const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                const float inv = 1.0f / len;
                for (float &c : n)
                    c *= inv;
            }
            float d[4];
            for (int c = 0; c < 4; ++c)
                d[c] = std::fmax(L[0][c] * n[0] + L[1][c] * n[1] + L[2][c] * n[2], 0.0f);
            float *o = &g_p01Out[static_cast<size_t>(i) * 16u];
            for (int c = 0; c < 4; ++c)
                o[c] = v[c];
            o[4] = n[0];
            o[5] = n[1];
            o[6] = n[2];
            o[7] = qv;
            o[8] = cl[0] * qv;
            o[9] = cl[1] * qv;
            o[10] = cl[2] * qv;
            o[11] = fog;
            for (int c = 0; c < 4; ++c)
                o[12 + c] = C[0][c] * d[0] + C[1][c] * d[1] + C[2][c] * d[2];
        }
        g_p01Base = b;
        g_p01Count = count;
        return true;
    }

    void p01Report()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - g_p01Last <= std::chrono::seconds(5))
            return;
        g_p01Last = now;
        if (g_p01Verts)
            std::fprintf(stderr, "[nvu1] p01 hash=%llx batches=%llu verts=%llu match: pos %.2f%% normal %.2f%% screen %.2f%% light %.2f%%%c",
                         static_cast<unsigned long long>(g_p01Hash), static_cast<unsigned long long>(g_p01Batches),
                         static_cast<unsigned long long>(g_p01Verts), 100.0 * g_p01Ok[0] / g_p01Verts,
                         100.0 * g_p01Ok[1] / g_p01Verts, 100.0 * g_p01Ok[2] / g_p01Verts, 100.0 * g_p01Ok[3] / g_p01Verts, 10);
        if (g_p01Verts)
            std::fprintf(stderr, "[nvu1] stage1 per program (all fields ok): p01 %llu/%llu p02 %llu/%llu p03 %llu/%llu p04 %llu/%llu%c",
                         static_cast<unsigned long long>(g_s1Ok[0]), static_cast<unsigned long long>(g_s1Verts[0]),
                         static_cast<unsigned long long>(g_s1Ok[1]), static_cast<unsigned long long>(g_s1Verts[1]),
                         static_cast<unsigned long long>(g_s1Ok[2]), static_cast<unsigned long long>(g_s1Verts[2]),
                         static_cast<unsigned long long>(g_s1Ok[3]), static_cast<unsigned long long>(g_s1Verts[3]), 10);
        if (g_p01NativeBatches)
            std::fprintf(stderr, "[nvu1] stage1 native batches=%llu%c", static_cast<unsigned long long>(g_p01NativeBatches), 10);
    }

    bool isP01(const uint8_t *code, uint32_t startPC)
    {
        if (!code)
            return false;
        g_s1 = -1;
        for (int k = 0; k < 4; ++k)
            if (kStage1[k].start == startPC)
                g_s1 = k;
        if (g_s1 < 0)
            return false;
        // HCTP_NATIVE_S1=mask: which of p01..p04 (bits 0..3) run natively (default: p01 only)
        static const char *s_maskEnv = std::getenv("HCTP_NATIVE_S1");
        static const unsigned s_mask = s_maskEnv ? static_cast<unsigned>(std::strtoul(s_maskEnv, nullptr, 0)) : 1u;
        if (!g_on && !(s_mask & (1u << g_s1)))
            return false;
        static uint64_t s_sig[4] = {0, 0, 0, 0}, s_hash[4] = {0, 0, 0, 0};
        uint64_t sig;
        std::memcpy(&sig, code + kStage1[g_s1].start, 8);
        if (sig != s_sig[g_s1])
        {
            s_sig[g_s1] = sig;
            s_hash[g_s1] = p01Hash(code);
        }
        g_p01Hash = s_hash[g_s1];
        // known-good hashes (verified): p01 adfb2a208b546483; others accepted when HCTP_S1_ANYHASH=1 or listed
        static const char *s_anyEnv = std::getenv("HCTP_S1_ANYHASH");
        static const bool s_any = s_anyEnv && s_anyEnv[0] == '1';
        if (g_on || s_any)
            return true;
        static const uint64_t kGood[4] = {0xadfb2a208b546483ull, 0ull, 0ull, 0ull};
        return kGood[g_s1] != 0ull && s_hash[g_s1] == kGood[g_s1];
    }

    bool nearV(const float *a, const float *b, int n, float tol)
    {
        for (int c = 0; c < n; ++c)
            if (!(std::fabs(a[c] - b[c]) <= tol * std::fmax(1.0f, std::fabs(b[c]))))
                return false;
        return true;
    }
}

bool hctpNativeVu1Mscal(const uint8_t *code, uint8_t *data, uint32_t startPC)
{
    static const bool s_init = []
    {
        const char *e = std::getenv("HCTP_NATIVE_P01");
        g_p01Native = e && e[0] == '1';
        return true;
    }();
    (void)s_init;
    g_p01Pending = false;
    if (!data || !isP01(code, startPC))
        return false;
    if (g_on)
    {
        g_p01Pending = p01Compute(data);
        return false;
    }
    if (!g_p01Native || !p01Compute(data))
        return false;
    for (uint32_t i = 0; i < g_p01Count; ++i)
    {
        const float *o = &g_p01Out[static_cast<size_t>(i) * 16u];
        std::memcpy(data + ((g_p01Base + i) & 0x3FFu) * 16u, o, 16);
        std::memcpy(data + ((g_p01Base + 160u + i) & 0x3FFu) * 16u, o + 4, 16);
        std::memcpy(data + ((g_p01Base + 320u + i) & 0x3FFu) * 16u, o + 8, 16);
        std::memcpy(data + ((g_p01Base + 480u + i) & 0x3FFu) * 16u, o + 12, 16);
    }
    ++g_p01NativeBatches;
    p01Report();
    return true;
}

void hctpNativeVu1AfterMscal(const uint8_t *data)
{
    if (!g_p01Pending || !data)
        return;
    g_p01Pending = false;
    ++g_p01Batches;
    for (uint32_t i = 0; i < g_p01Count; ++i)
    {
        const float *o = &g_p01Out[static_cast<size_t>(i) * 16u];
        const float *vu[4] = {q(data, g_p01Base + i), q(data, g_p01Base + 160u + i), q(data, g_p01Base + 320u + i),
                              q(data, g_p01Base + 480u + i)};
        const bool ok[4] = {nearV(vu[0], o, 4, 1e-4f), nearV(vu[1], o + 4, 4, 1e-3f), nearV(vu[2], o + 8, 4, 1e-3f),
                            nearV(vu[3], o + 12, 3, 1e-2f)};
        ++g_p01Verts;
        for (int k = 0; k < 4; ++k)
            g_p01Ok[k] += ok[k];
        ++g_s1Verts[g_s1];
        g_s1Ok[g_s1] += ok[0] && ok[1] && ok[2] && ok[3];
        if (!(ok[0] && ok[1] && ok[2] && ok[3]) && g_p01Dumped < 8)
        {
            ++g_p01Dumped;
            std::fprintf(stderr, "[nvu1] s1 p0%d mismatch v=%u ok=%d%d%d%d | p vu %g %g %g %g spec %g %g %g %g | n vu %g %g %g %g spec %g %g %g %g | s vu %g %g %g %g spec %g %g %g %g | L vu %g %g %g spec %g %g %g%c",
                         g_s1 + 1, i, ok[0], ok[1], ok[2], ok[3], vu[0][0], vu[0][1], vu[0][2], vu[0][3], o[0], o[1], o[2], o[3],
                         vu[1][0], vu[1][1], vu[1][2], vu[1][3], o[4], o[5], o[6], o[7], vu[2][0], vu[2][1], vu[2][2],
                         vu[2][3], o[8], o[9], o[10], o[11], vu[3][0], vu[3][1], vu[3][2], o[12], o[13], o[14], 10);
        }
    }
    p01Report();
}
