// HCTP native model draws: the p08 shader spec (notes\VU1_PROGRAMS.md "SHADER SPEC") as plain C++.
// HCTP_NATIVE_VERIFY=1: before each later p08 chunk (MSCNT at the loop head) the corners are computed from VU1 data
// memory with the spec's formulas; after the VU1 ran, they are compared with the packet p08 wrote at TOP+33 (per
// corner: STQ, RGBA, XYZF2 incl. the backface ADC bit). Every 5 s: corners compared / matching per field.
// HCTP_NATIVE_P08=1: chunks that need no clipping are drawn from the spec (the packet goes straight to the GS) and
// the VU1 does not run for them; chunks with a CLIP flag stay on the VU1 (p05/p06 near-plane clipper).
#include "hctp_native_vu1.h"

#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_vu1_compiled.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <xmmintrin.h>

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
    // p11 (overlay 0x2a00): p08 without the ST scroll and without the strip-parity flip in the backface test
    constexpr uint64_t kP11Hash = 0x1c814dd03a6b2482ull;
    constexpr uint32_t kP11LoopHead = 0x2fb0u;
    bool g_p11 = false; // HCTP_NATIVE_P11=1
    uint64_t g_p11Seen = 0;
    // p09 (resident at VU 0x3040): light-relative reflection map: ST = ([1009].xy + L * n) * [1009].zw, then * q;
    // L columns = [1012], [1013], [1014] (.xy)
    constexpr uint64_t kP09Hash = 0xe50add3f88ac1641ull;
    constexpr uint32_t kP09LoopHead = 0x3720u;
    bool g_p09 = false; // HCTP_NATIVE_P09=1
    uint64_t g_p09Seen = 0;
    // Imported wrestler materials can carry the reflection-map path even when their
    // source model was authored without the game's intended skin treatment.  Keep
    // this opt-in: it is a broad VU model-path switch, not a renderer default.
    bool g_noGloss = false; // HCTP_NO_GLOSS=1
    // p19 My_Code_TriStripChar_NonClip (overlay 0x2a00): p18's math, but no backface test, no clipping and no ADC
    // (the microcode has no ISW): every corner is drawn
    constexpr uint64_t kP19Hash = 0x43232620e643a207ull;
    constexpr uint32_t kP19LoopHead = 0x2d50u;
    bool g_p19 = false; // HCTP_NATIVE_P19=1
    uint64_t g_p19Seen = 0;
    // p15 My_Code_TriStripProjectionMap (overlay 0x2a00): projector T = [1005] + [1002]*V.x + [1003]*V.y + [1004]*V.z
    // (V = view position [index]), STQ = (T.x/T.w*q, T.y/T.w*q, q); RGBA = min([1006]' * light + [1007]*[1019], 255)
    // (no corner colour); position / clip / backface as p08
    constexpr uint64_t kP15Hash = 0x8c043e47d2d51d3aull;
    constexpr uint32_t kP15LoopHead = 0x30f0u;
    bool g_p15 = false; // HCTP_NATIVE_P15=1
    uint64_t g_p15Seen = 0;
    int g_variant = 0; // 0 = p08, 1 = p18, 2 = p10, 3 = p11, 4 = p09, 5 = p19, 6 = p15 (the chunk computeChunk last accepted)

    uint64_t hashAt(const uint8_t *code, uint32_t at)
    {
        uint64_t h = 1469598103934665603ull;
        for (uint32_t i = 0; i < 0x200u; ++i)
            h = (h ^ code[at + i]) * 1099511628211ull;
        return h;
    }
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
        if (st.pc == kP09LoopHead)
        {
            static uint64_t s_sig09 = 0, s_hash09 = 0;
            uint64_t sig09;
            std::memcpy(&sig09, code + 0x3050u, 8);
            if (sig09 != s_sig09)
            {
                s_sig09 = sig09;
                s_hash09 = hashAt(code, 0x3040u);
            }
            if (s_hash09 != kP09Hash || !(g_p09 || g_on))
                return false;
            g_variant = g_noGloss ? 0 : 4;
            ++g_p09Seen;
        }
        else if (st.pc == kP10LoopHead)
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
            g_variant = g_noGloss ? 0 : 2;
            ++g_p10Seen;
        }
        else if (st.pc == kP15LoopHead && s_hash == kP15Hash && (g_p15 || g_on))
        {
            g_variant = 6;
            ++g_p15Seen;
        }
        else if (st.pc == kP19LoopHead && s_hash == kP19Hash && (g_p19 || g_on))
        {
            g_variant = 5;
            ++g_p19Seen;
        }
        else if (st.pc == kP11LoopHead && s_hash == kP11Hash && (g_p11 || g_on))
        {
            g_variant = 3;
            ++g_p11Seen;
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
        const bool p18 = g_variant == 1 || g_variant == 5; // p19 = p18's colour / STQ

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
                float col = g_variant == 6 ? s1[c] * light[c] + s2[c] : c1[c] * s1[c] * light[c] + c1[c] * s2[c];
                col = std::fmin(col, 255.0f);
                o.rgba[c] = ftoi(col, 1.0f);
                o.col[c] = col;
            }
            const float *Nn = q(data, cornerIndex(data, top, k) + 160u);
            float st3[3] = {g_variant == 2 ? scroll[0] + Nn[0] * scroll[2] : c0[0] + scroll[0],
                                  g_variant == 2 ? scroll[1] + Nn[1] * scroll[3] : c0[1] + scroll[1], g_variant == 2 ? 1.0f : c0[2]};
            if (g_variant == 4)
            {
                const float *L0 = q(data, 1012), *L1 = q(data, 1013), *L2 = q(data, 1014);
                st3[0] = (scroll[0] + L0[0] * Nn[0] + L1[0] * Nn[1] + L2[0] * Nn[2]) * scroll[2];
                st3[1] = (scroll[1] + L0[1] * Nn[0] + L1[1] * Nn[1] + L2[1] * Nn[2]) * scroll[3];
                st3[2] = 1.0f;
            }
            if (g_variant == 6)
            {
                const float *V = q(data, cornerIndex(data, top, k));
                const float *m2 = q(data, 1002), *m3 = q(data, 1003), *m4 = q(data, 1004), *m5 = q(data, 1005);
                float T[4];
                for (int c = 0; c < 4; ++c)
                {
                    float acc = m5[c] * 1.0f;
                    acc += m2[c] * V[0];
                    acc += m3[c] * V[1];
                    T[c] = acc + m4[c] * V[2];
                }
                const float qd = 1.0f / T[3];
                st3[0] = T[0] * qd;
                st3[1] = T[1] * qd;
                st3[2] = 1.0f;
            }
            if (g_variant == 3)
            {
                st3[0] = c0[0];
                st3[1] = c0[1];
            }
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
            o.flagged = g_variant != 5 && (std::fabs(cx) > band || std::fabs(cy) > band || cz < -guard);
            if (o.flagged)
                g_chunkClip = true;
            else if (g_on && o.clip[3] != 0.0f)
            {
                ++g_projN;
                g_projOk += std::fabs(o.clip[0] / o.clip[3] - P[0]) < 0.05f && std::fabs(o.clip[1] / o.clip[3] - P[1]) < 0.05f;
            }
            // backface (3rd corner on): cross z of (P[k-2]-P[k-1]) x (P[k-1]-P[k]), sign + strip parity, & vi8 -> ADC
            if (k >= 2u && g_variant != 5)
            {
                const float *P0 = q(data, cornerIndex(data, top, k - 2u) + 320u);
                const float *P1 = q(data, cornerIndex(data, top, k - 1u) + 320u);
                const float ax = P0[0] - P1[0], ay = P0[1] - P1[1], bx = P1[0] - P[0], by = P1[1] - P[1];
                const float crossZ = ax * by - bx * ay;
                const int32_t v = (crossZ < 0.0f ? 32 : 0) + ((g_variant != 3 && ((k - 2u) & 1u)) ? 32 : 0);
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
        if (g_p15Seen)
            std::fprintf(stderr, "[nvu1] p15 chunks seen=%llu%c", static_cast<unsigned long long>(g_p15Seen), 10);
        if (g_p19Seen)
            std::fprintf(stderr, "[nvu1] p19 chunks seen=%llu%c", static_cast<unsigned long long>(g_p19Seen), 10);
        if (g_p09Seen)
            std::fprintf(stderr, "[nvu1] p09 chunks seen=%llu%c", static_cast<unsigned long long>(g_p09Seen), 10);
        if (g_p11Seen)
            std::fprintf(stderr, "[nvu1] p11 chunks seen=%llu%c", static_cast<unsigned long long>(g_p11Seen), 10);
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


// ---------------------------------------------------------------------------------------------------------------
// p12 My_Code_LineStrip / p13 My_Code_LineList (overlay 0x2a00, MSCNT chunks like p08): corner records at TOP+1
// (ST + index in .w, colour), packet at TOP+33. Per corner: STQ = ST * P.w, RGBA = FTOI0(min(colour * [1006] *
// (255,255,255,128), 255)), XYZF2 = FTOI4(P), P = [index+320]. CLIP of (P.x-2048, P.y-2048, [index].z-2048) against
// 2047; ADC (w = 0x8000) when FCAND 1007 hits: this corner's +x -x +y -y -z or the previous corner's x / y bits.
// p12 tests every corner (against the previous one, across chunks); p13 only the second corner of each pair.
namespace
{
    constexpr uint64_t kP12Hash = 0xcbfc02991d763c2dull; // 472 bytes
    constexpr uint64_t kP13Hash = 0x1cf36cba54f38a53ull; // 512 bytes
    constexpr uint32_t kP12LoopHead = 0x2bc8u, kP13LoopHead = 0x2c60u;
    bool g_lines = false; // HCTP_NATIVE_LINES=1
    uint32_t g_lineFlags = 0; // CLIP flag history (6 bits per CLIP), carried across p12 chunks
    std::vector<uint8_t> g_linePacket;
    bool g_linePending = false;
    uint32_t g_lineTop = 0;
    uint64_t g_lineChunks = 0, g_lineQw = 0, g_lineOk = 0;

    uint32_t lineClip(const float *v) // CLIP.xyz against w = 2047
    {
        const float a = 2047.0f;
        return (v[0] > a ? 1u : 0u) | (v[0] < -a ? 2u : 0u) | (v[1] > a ? 4u : 0u) | (v[1] < -a ? 8u : 0u) |
               (v[2] > a ? 16u : 0u) | (v[2] < -a ? 32u : 0u);
    }

    // 12 / 13 when the overlay + resume PC is one of them (packet in g_linePacket), else 0
    int lineChunk(const uint8_t *code, const uint8_t *data, const VU1State &st, uint32_t top)
    {
        if (!code || !data || (st.pc != kP12LoopHead && st.pc != kP13LoopHead))
            return 0;
        static uint64_t s_sig = 0, s_h12 = 0, s_h13 = 0;
        uint64_t sig;
        std::memcpy(&sig, code + 0x2a10u, 8);
        if (sig != s_sig)
        {
            s_sig = sig;
            uint64_t h = 1469598103934665603ull, h2 = 0;
            for (uint32_t i = 0; i < 0x200u; ++i)
            {
                h = (h ^ code[0x2a00u + i]) * 1099511628211ull;
                if (i == 471u)
                    h2 = h;
            }
            s_h12 = h2;
            s_h13 = h;
        }
        const int prog = (st.pc == kP12LoopHead && s_h12 == kP12Hash) ? 12 : (st.pc == kP13LoopHead && s_h13 == kP13Hash) ? 13 : 0;
        if (!prog)
            return 0;
        uint32_t header;
        std::memcpy(&header, data + (top & 0x3FFu) * 16u, 4);
        const uint32_t count = header & 0x7FFFu;
        if (count == 0u || count > 48u || (prog == 13 && (count & 1u)))
            return 0;
        const float *v1 = q(data, 1006);
        const float s1[4] = {v1[0] * 255.0f, v1[1] * 255.0f, v1[2] * 255.0f, v1[3] * 128.0f};
        g_linePacket.resize((1u + 3u * count) * 16u);
        uint8_t *o = g_linePacket.data();
        std::memcpy(o, data + (top & 0x3FFu) * 16u, 16);
        uint32_t flags = g_lineFlags;
        for (uint32_t k = 0; k < count; ++k)
        {
            const float *c0 = q(data, top + 1u + 2u * k), *c1 = q(data, top + 2u + 2u * k);
            const uint32_t idx = cornerIndex(data, top, k);
            const float *P = q(data, idx + 320u);
            float stq[4];
            int32_t rgba[4], xyz[4];
            for (int c = 0; c < 4; ++c)
            {
                stq[c] = c0[c] * P[3];
                rgba[c] = ftoi(std::fmin(c1[c] * s1[c], 255.0f), 1.0f);
                xyz[c] = ftoi(P[c], 16.0f);
            }
            const float cv[3] = {P[0] - 2048.0f, P[1] - 2048.0f, q(data, idx)[2] - 2048.0f};
            flags = ((flags << 6) | lineClip(cv)) & 0xFFFFFFu;
            const bool test = prog == 12 || (k & 1u);
            if (test && (flags & 1007u))
                xyz[3] = 0x8000;
            uint8_t *c = o + (1u + 3u * k) * 16u;
            std::memcpy(c, stq, 16);
            std::memcpy(c + 16, rgba, 16);
            std::memcpy(c + 32, xyz, 16);
        }
        if (prog == 12)
            g_lineFlags = flags;
        return prog;
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
    const char *n11 = std::getenv("HCTP_NATIVE_P11");
    g_p11 = n11 && n11[0] == '1';
    const char *n09 = std::getenv("HCTP_NATIVE_P09");
    g_p09 = n09 && n09[0] == '1';
    const char *ng = std::getenv("HCTP_NO_GLOSS");
    g_noGloss = ng && ng[0] == '1';
    const char *n19 = std::getenv("HCTP_NATIVE_P19");
    g_p19 = n19 && n19[0] == '1';
    const char *n15 = std::getenv("HCTP_NATIVE_P15");
    g_p15 = n15 && n15[0] == '1';
    const char *nl = std::getenv("HCTP_NATIVE_LINES");
    g_lines = nl && nl[0] == '1';
    if (g_on || g_native)
        std::fprintf(stderr, "[nvu1] p08 spec verify %d native %d%c", g_on ? 1 : 0, g_native ? 1 : 0, 10);
}

bool hctpNativeVu1TryDraw(const uint8_t *code, uint8_t *data, const VU1State &st, uint32_t top, GS &gs)
{
    if (g_lines && !g_on && lineChunk(code, data, st, top))
    {
        for (size_t i = 0; i < g_linePacket.size() / 16u; ++i)
            std::memcpy(data + ((top + 33u + static_cast<uint32_t>(i)) & 0x3FFu) * 16u, g_linePacket.data() + i * 16u, 16);
        gs.processGIFPacket(g_linePacket.data(), static_cast<uint32_t>(g_linePacket.size()), 1u);
        ++g_lineChunks;
        return true;
    }
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
    g_linePending = false;
    if (g_on && lineChunk(code, data, st, top))
    {
        g_linePending = true;
        g_lineTop = top;
    }
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
    if (g_linePending && data)
    {
        g_linePending = false;
        ++g_lineChunks;
        for (size_t i = 1; i < g_linePacket.size() / 16u; ++i)
        {
            const uint8_t *vu = data + ((g_lineTop + 33u + static_cast<uint32_t>(i)) & 0x3FFu) * 16u;
            const uint8_t *nat = g_linePacket.data() + i * 16u;
            bool same = true;
            for (int c = 0; c < 4 && same; ++c)
            {
                int32_t a, b;
                std::memcpy(&a, nat + c * 4, 4);
                std::memcpy(&b, vu + c * 4, 4);
                float fa, fb;
                std::memcpy(&fa, &a, 4);
                std::memcpy(&fb, &b, 4);
                same = std::abs(static_cast<int64_t>(a) - b) <= 1 || (i % 3u == 1u && std::fabs(fa - fb) <= 1e-4f * std::fmax(1.0f, std::fabs(fa)));
            }
            ++g_lineQw;
            g_lineOk += same;
        }
        static auto s_last = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - s_last > std::chrono::seconds(5))
        {
            s_last = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[nvu1] lines chunks=%llu qwords=%llu match=%.3f%%%c", static_cast<unsigned long long>(g_lineChunks),
                         static_cast<unsigned long long>(g_lineQw), g_lineQw ? 100.0 * g_lineOk / g_lineQw : 0.0, 10);
        }
    }
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
                         g_variant == 1 ? 18 : g_variant == 2 ? 10 : g_variant == 3 ? 11 : g_variant == 4 ? 9 : g_variant == 5 ? 19 : g_variant == 6 ? 15 : 8, g_top, k, stq[0], stq[1], stq[2], e.stq[0], e.stq[1], e.stq[2], rgba[0], rgba[1], rgba[2], rgba[3],
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

#include <algorithm>
#include <map>
#include <mutex>

void hctpNativeVu1Census(const uint8_t *code, uint32_t pc, bool mscal)
{
    static const bool s_on = std::getenv("HCTP_VU1_CENSUS") != nullptr;
    if (!s_on || !code)
        return;
    static std::mutex s_lock;
    static std::map<uint64_t, uint64_t> s_counts; // (hash of the 512 bytes at the program base) ^ pc -> runs
    static std::map<uint64_t, std::pair<uint32_t, uint64_t>> s_keys;
    static auto s_last = std::chrono::steady_clock::now();
    // program base: the overlay slot 0x2a00 for PCs inside it, the resident programs' own starts otherwise
    const uint32_t base = (pc >= 0x2a00u && pc < 0x3040u) ? 0x2a00u : (pc >= 0x3040u && pc < 0x3740u) ? 0x3040u
                          : (pc >= 0x3740u && pc < 0x3d90u) ? 0x3740u : (pc & ~0xFFu);
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < 0x200u && base + i < 0x4000u; ++i)
        h = (h ^ code[base + i]) * 1099511628211ull;
    const uint64_t key = h ^ (static_cast<uint64_t>(pc) << 1) ^ (mscal ? 1u : 0u);
    std::lock_guard<std::mutex> lock(s_lock);
    ++s_counts[key];
    s_keys[key] = {pc | (mscal ? 0x80000000u : 0u), h};
    const auto now = std::chrono::steady_clock::now();
    if (now - s_last < std::chrono::seconds(10))
        return;
    s_last = now;
    std::vector<std::pair<uint64_t, uint64_t>> v(s_counts.begin(), s_counts.end());
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
    for (size_t i = 0; i < v.size() && i < 16; ++i)
    {
        const auto &k = s_keys[v[i].first];
        std::fprintf(stderr, "[vu1census] %s pc=0x%04x hash=%016llx runs=%llu%c", (k.first & 0x80000000u) ? "MSCAL" : "MSCNT",
                     k.first & 0xFFFFu, static_cast<unsigned long long>(k.second), static_cast<unsigned long long>(v[i].second), 10);
    }
    s_counts.clear();
}

// ---------------------------------------------------------------------------------------------------------------
// Whole-program natives (MSCAL-started VU1 programs; names from the ELF's symbol table, notes\VU1_PROGRAMS.md).
// Each native reads / writes VU1 data memory like the microcode and returns the GS packets it would XGKICK.
extern thread_local std::vector<uint8_t> *g_hctpPath1Capture;

namespace
{
    using Kicks = std::vector<std::vector<uint8_t>>;

    float *qwf(uint8_t *data, uint32_t qw) { return reinterpret_cast<float *>(data + (qw & 0x3FFu) * 16u); }
    int32_t *qwi(uint8_t *data, uint32_t qw) { return reinterpret_cast<int32_t *>(data + (qw & 0x3FFu) * 16u); }
    int32_t ilw(const uint8_t *data, uint32_t qw, int field) // ILW: the low 16 bits of a field, sign-extended
    {
        int32_t v;
        std::memcpy(&v, data + (qw & 0x3FFu) * 16u + field * 4, 4);
        return static_cast<int16_t>(v & 0xFFFF);
    }

    // qwords of the GIF packet starting at qw (tags until EOP; PACKED nloop*nreg, REGLIST nloop*nreg/2, IMAGE nloop)
    uint32_t gifPacketQwords(const uint8_t *data, uint32_t qw)
    {
        uint32_t n = 0;
        for (int guard = 0; guard < 64; ++guard)
        {
            uint64_t tag;
            std::memcpy(&tag, data + ((qw + n) & 0x3FFu) * 16u, 8);
            const uint32_t nloop = static_cast<uint32_t>(tag & 0x7FFFu);
            const uint32_t flg = static_cast<uint32_t>((tag >> 58) & 3u);
            uint32_t nreg = static_cast<uint32_t>((tag >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            n += 1u;
            if (flg == 0u)
                n += nloop * nreg;
            else if (flg == 1u)
                n += (nloop * nreg + 1u) / 2u;
            else
                n += nloop;
            if ((tag >> 15) & 1u)
                break;
        }
        return n;
    }

    void kickAt(const uint8_t *data, uint32_t qw, Kicks &out)
    {
        const uint32_t n = gifPacketQwords(data, qw);
        std::vector<uint8_t> p(n * 16u);
        for (uint32_t i = 0; i < n; ++i)
            std::memcpy(p.data() + i * 16u, data + ((qw + i) & 0x3FFu) * 16u, 16);
        out.push_back(std::move(p));
    }

    int32_t ftoi4(float v) { return ftoi(v, 16.0f); }

    // p17 My_Code_Audience (the crowd), MSCAL 0x2a00. Kicks the register packet at 256; per batch ([850].x
    // batches, 5 qwords each from 851: matrix rows + one register value stored to 261) kicks 260, then per group
    // ([850].y groups, headers from 262, vertex pairs (attr, pos) from 267, the same for every batch):
    // out = header, then per vertex attr and (FTOI4(xyz / w of M*pos), attr.w); double buffered at 0 / 128.
    bool runP17(uint8_t *data, Kicks &kicks)
    {
        int32_t batches = ilw(data, 850u, 0);
        if (batches <= 0 || batches > 64)
            return false;
        kickAt(data, 256u, kicks);
        uint32_t vi9 = 851u, vi12 = 0u;
        for (; batches > 0; --batches, vi9 += 5u)
        {
            const float *m[4] = {qwf(data, vi9), qwf(data, vi9 + 1u), qwf(data, vi9 + 2u), qwf(data, vi9 + 3u)};
            std::memcpy(data + 261u * 16u, data + (vi9 + 4u) * 16u, 16);
            kickAt(data, 260u, kicks);
            uint32_t vi11 = 262u, vi5 = 267u;
            int32_t groups = ilw(data, 850u, 1);
            if (groups <= 0 || groups > 512)
                return false;
            for (; groups > 0; --groups, ++vi11, vi12 = 128u - vi12)
            {
                const uint32_t count = static_cast<uint32_t>(ilw(data, vi11, 0)) & 255u;
                if ((count & 3u) != 0u || count == 0u)
                    return false; // the microcode loops in fours until zero
                std::memcpy(data + vi12 * 16u, data + (vi11 & 0x3FFu) * 16u, 16);
                for (uint32_t k = 0; k < count; ++k, vi5 += 2u)
                {
                    const float *a = qwf(data, vi5), *pos = qwf(data, vi5 + 1u);
                    float r[4];
                    for (int c = 0; c < 4; ++c)
                    {
                        float acc = m[0][c] * pos[0];
                        acc += m[1][c] * pos[1];
                        acc += m[2][c] * pos[2];
                        r[c] = acc + m[3][c] * pos[3];
                    }
                    const float qv = 1.0f / r[3];
                    const uint32_t o = vi12 + 1u + 2u * k;
                    float attr[4];
                    std::memcpy(attr, a, 16);
                    std::memcpy(data + (o & 0x3FFu) * 16u, attr, 16);
                    int32_t *x = qwi(data, o + 1u);
                    std::memcpy(x, attr, 16);
                    x[0] = ftoi4(r[0] * qv);
                    x[1] = ftoi4(r[1] * qv);
                    x[2] = ftoi4(r[2] * qv);
                }
                kickAt(data, vi12, kicks);
            }
        }
        return true;
    }


    // CLIP of fs.xyz against |w|: 6 flag bits (+x -x +y -y +z -z)
    uint32_t clipBits(const float *v, float w)
    {
        const float a = std::fabs(w);
        return (v[0] > a ? 1u : 0u) | (v[0] < -a ? 2u : 0u) | (v[1] > a ? 4u : 0u) | (v[1] < -a ? 8u : 0u) |
               (v[2] > a ? 16u : 0u) | (v[2] < -a ? 32u : 0u);
    }

    // p16 My_dma_Particle (sprites), MSCAL 0x2a00. Matrix rows [485..488], corner offsets [494] / [495], UV qwords
    // [490] / [493], packet tag [489], [480].x outer x [480].y inner particles, scale [480].z, clip bounds [481].w /
    // [482].w around the centre [483]. Inputs: position from 0, colour from 80, size from 160 (one qword each).
    // Output from 501, 5 qwords per particle: colour, UV0, FTOI4 corner A, UV1, FTOI4 corner B; a particle whose
    // corner A is above +bound or corner B below -bound (y, against [482].w) gets zero corners. One kick at 500.
    bool runP16(uint8_t *data, Kicks &kicks)
    {
        const int32_t outer = ilw(data, 480u, 0), inner = ilw(data, 480u, 1);
        if (outer <= 0 || inner <= 0 || outer * inner > 100)
            return false;
        const float *m[4] = {qwf(data, 485u), qwf(data, 486u), qwf(data, 487u), qwf(data, 488u)};
        const float *off0 = qwf(data, 494u), *off1 = qwf(data, 495u), *centre = qwf(data, 483u);
        const float scale = qwf(data, 480u)[2], b23 = qwf(data, 481u)[3], b24 = qwf(data, 482u)[3];
        float uv0[4], uv1[4];
        std::memcpy(uv0, qwf(data, 490u), 16);
        std::memcpy(uv1, qwf(data, 493u), 16);
        auto project = [&](const float *v, float *r)
        {
            for (int c = 0; c < 4; ++c)
            {
                float acc = m[0][c] * v[0];
                acc += m[1][c] * v[1];
                acc += m[2][c] * v[2];
                r[c] = acc + m[3][c] * v[3];
            }
            const float qv = 1.0f / r[3];
            for (int c = 0; c < 4; ++c)
                r[c] *= qv;
        };
        uint32_t in = 0u, out = 501u;
        for (int32_t i = 0; i < outer * inner; ++i, ++in, out += 5u)
        {
            const float *size = qwf(data, 160u + in), *pos = qwf(data, in);
            float colour[4], base[4], ca[4], cb[4], pa[4], pb[4];
            std::memcpy(colour, qwf(data, 80u + in), 16);
            for (int c = 0; c < 4; ++c)
            {
                base[c] = pos[c] - size[c] * scale;
                ca[c] = off0[c] + base[c];
                cb[c] = off1[c] + base[c];
            }
            project(ca, pa);
            project(cb, pb);
            float da[4], db[4];
            for (int c = 0; c < 4; ++c)
            {
                da[c] = pa[c] - centre[c];
                db[c] = pb[c] - centre[c];
            }
            // flags after the 4 CLIPs: [vf4 vs 23][vf13 vs 23][vf4 vs 24][vf13 vs 24]; FCAND 0x108 = bit 3 (-y of the
            // last) | bit 8 (+y of the third)
            const uint32_t flags = (clipBits(da, b23) << 18) | (clipBits(db, b23) << 12) | (clipBits(da, b24) << 6) | clipBits(db, b24);
            int32_t xa[4], xb[4];
            if (flags & 0x108u)
            {
                const int32_t z[4] = {0, 0, 0, 16};
                std::memcpy(xa, z, 16);
                std::memcpy(xb, z, 16);
            }
            else
                for (int c = 0; c < 4; ++c)
                {
                    xa[c] = ftoi4(pa[c]);
                    xb[c] = ftoi4(pb[c]);
                }
            std::memcpy(data + (out & 0x3FFu) * 16u, colour, 16);
            std::memcpy(data + ((out + 1u) & 0x3FFu) * 16u, uv0, 16);
            std::memcpy(data + ((out + 2u) & 0x3FFu) * 16u, xa, 16);
            std::memcpy(data + ((out + 3u) & 0x3FFu) * 16u, uv1, 16);
            std::memcpy(data + ((out + 4u) & 0x3FFu) * 16u, xb, 16);
        }
        std::memcpy(data + 500u * 16u, data + 489u * 16u, 16);
        kickAt(data, 500u, kicks);
        return true;
    }

    // p07 Sp_Loop (morph weights), MSCAL 0x2800: converts the integer weights in .w of [480..480+N) (N = [1009].x, a
    // multiple of 4, groups of 4) in place: w = MAX(FTOI0(ITOF0(w) * c.z + c.x), c.w), c = [1011]; other fields are
    // untouched. Ends at [E] 0x2900 (the second entry 0x2908, the target blend, needs register state: stays on the VU1).
    bool runP07(uint8_t *data, Kicks &)
    {
        const int32_t n = ilw(data, 1009u, 0);
        if (n <= 0 || (n & 3) != 0 || 480 + n > 1009)
            return false;
        const float *c = qwf(data, 1011u);
        const __m128 cw = _mm_set_ss(c[3]);
        // the VU1 path rounds these toward zero (ITOF0 of the raw weight bits and the product: verified against it)
        const unsigned int savedCsr = _mm_getcsr();
        _mm_setcsr((savedCsr & ~0x6000u) | 0x6000u);
        for (int32_t j = 0; j < n; ++j)
        {
            int32_t *q = qwi(data, 480u + static_cast<uint32_t>(j));
            // SSE intrinsics (not C++ float maths): kept in order with the MXCSR change around them
            const __m128 prod = _mm_mul_ss(_mm_cvtsi32_ss(_mm_setzero_ps(), q[3]), _mm_set_ss(c[2]));
            const float v = _mm_cvtss_f32(_mm_add_ss(prod, _mm_set_ss(c[0])));
            const int32_t iv = ftoi(v, 1.0f);
            // MAX takes its operands as floats with denormals flushed to signed zero (the FTOI'd weight's bits are
            // usually a denormal)
            // (and Inf / NaN bits - a saturated FTOI - clamped to the signed largest float), as the VU1 path's operand()
            const int32_t sign = iv & static_cast<int32_t>(0x80000000u), expo = iv & 0x7F800000;
            const int32_t flushed = expo == 0 ? sign : expo == 0x7F800000 ? (sign | 0x7F7FFFFF) : iv;
            float f;
            std::memcpy(&f, &flushed, 4);
            f = _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(f), cw)); // MAX on the bit patterns, as the VU1 path does
            std::memcpy(&q[3], &f, 4);
        }
        _mm_setcsr(savedCsr);
        return true;
    }

    struct Generic
    {
        const char *name;
        uint32_t base, bytes, entry;
        uint64_t hash;
        bool (*run)(uint8_t *, Kicks &);
        const char *env;
        bool on;
        uint64_t runs, nativeRuns, qwords, okQwords, fails;
        bool memCompare = false; // verify VU1 data memory (programs that only write memory, no GS packets)
        bool off = false; // HCTP_NATIVE_Pxx=0
    };
    Generic g_generic[] = {
        {"p17 Audience", 0x2a00u, 0x200u, 0x2a00u, 0x6d85202f7ec6d013ull, runP17, "HCTP_NATIVE_P17", false, 0, 0, 0, 0, 0},
        {"p16 Particle", 0x2a00u, 0x200u, 0x2a00u, 0x7d42d0fa68ec24e5ull, runP16, "HCTP_NATIVE_P16", false, 0, 0, 0, 0, 0},
        {"p07 MorphWeights", 0x2800u, 0x190u, 0x2800u, 0xffbdc4d8743d017aull, runP07, "HCTP_NATIVE_P07", false, 0, 0, 0, 0, 0, true},
        // p14 runs from MSCNT (hctpNativeVu1MscntP14, hctp_native_vu1_p14.inl): here for its switch and verify counters
        {"p14 Shadow", 0x2a00u, 0x200u, 0xFFFFFFFFu, 0x00e7b7f008461125ull, nullptr, "HCTP_NATIVE_P14", false, 0, 0, 0, 0, 0},
    };
    bool g_genericInit = false;
    Generic *g_genPending = nullptr;
    std::vector<uint8_t> g_genExpected, g_genCaptured;
    std::vector<uint8_t> g_genScratch;
    auto g_genLast = std::chrono::steady_clock::now();

    uint64_t hashRange(const uint8_t *code, uint32_t at, uint32_t bytes)
    {
        uint64_t h = 1469598103934665603ull;
        for (uint32_t i = 0; i < bytes; ++i)
            h = (h ^ code[at + i]) * 1099511628211ull;
        return h;
    }

    void genericReport()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - g_genLast < std::chrono::seconds(5))
            return;
        g_genLast = now;
        for (const Generic &g : g_generic)
            if (g.runs)
                std::fprintf(stderr, "[nvu1g] %s runs=%llu native=%llu fails=%llu qwords=%llu match=%.3f%%%c", g.name,
                             static_cast<unsigned long long>(g.runs), static_cast<unsigned long long>(g.nativeRuns),
                             static_cast<unsigned long long>(g.fails), static_cast<unsigned long long>(g.qwords),
                             g.qwords ? 100.0 * g.okQwords / g.qwords : 0.0, 10);
    }
}

bool hctpNativeVu1MscalGeneric(const uint8_t *code, uint8_t *data, uint32_t startPC, GS &gs)
{
    if (!g_genericInit)
    {
        g_genericInit = true;
        for (Generic &g : g_generic)
        {
            g.on = std::getenv(g.env) && std::getenv(g.env)[0] == '1';
            g.off = std::getenv(g.env) && std::getenv(g.env)[0] == '0'; // =0: always the VU1, even with natives on
        }
    }
    if (!code || !data)
        return false;
    for (Generic &g : g_generic)
    {
        if (g.entry != startPC || g.off || (!g.on && !g_on))
            continue;
        {
            // live switch: <HCTP_SHOT_DIR>\\vu17 present = p17 on the real VU1 (A/B of the native port)
            static const char *s_dir = std::getenv("HCTP_SHOT_DIR");
            static bool s_vu17 = false;
            static auto s_chk = std::chrono::steady_clock::now();
            if (s_dir && std::chrono::steady_clock::now() - s_chk > std::chrono::milliseconds(500))
            {
                s_chk = std::chrono::steady_clock::now();
                char path[512];
                std::snprintf(path, sizeof(path), "%s\\vu17", s_dir);
                FILE *f = std::fopen(path, "rb");
                if ((f != nullptr) != s_vu17)
                    std::fprintf(stderr, "[abswitch] vu17=%d%c", f != nullptr, 10);
                s_vu17 = f != nullptr;
                if (f)
                    std::fclose(f);
            }
            if (s_vu17 && g.run == runP17)
                continue;
        }
        // the overlay changes rarely: hash only when its first words change
        static uint64_t s_sig[8] = {}, s_hash[8] = {};
        const size_t gi = static_cast<size_t>(&g - g_generic);
        uint64_t sig;
        std::memcpy(&sig, code + g.base + 8u, 8);
        if (sig != s_sig[gi])
        {
            s_sig[gi] = sig;
            s_hash[gi] = hashRange(code, g.base, g.bytes);
        }
        if (s_hash[gi] != g.hash)
            continue;
        ++g.runs;
        Kicks kicks;
        if (g_on)
        {
            // verify: run on a copy, capture what the VU1 sends, compare in hctpNativeVu1AfterMscalGeneric
            g_genScratch.assign(data, data + 0x4000u);
            if (!g.run(g_genScratch.data(), kicks))
            {
                ++g.fails;
                return false;
            }
            g_genExpected.clear();
            for (const auto &k : kicks)
                g_genExpected.insert(g_genExpected.end(), k.begin(), k.end());
            g_genCaptured.clear();
            g_hctpPath1Capture = &g_genCaptured;
            g_genPending = &g;
            return false;
        }
        if (!g.run(data, kicks))
        {
            ++g.fails;
            return false; // inputs outside what the native handles: the VU1 runs it
        }
        for (const auto &k : kicks)
            gs.processGIFPacket(k.data(), static_cast<uint32_t>(k.size()), 1u);
        ++g.nativeRuns;
        genericReport();
        return true;
    }
    return false;
}

void hctpNativeVu1AfterMscalGeneric(const uint8_t *data)
{
    if (!g_genPending)
        return;
    g_hctpPath1Capture = nullptr;
    Generic &g = *g_genPending;
    g_genPending = nullptr;
    if (g.memCompare && data && g_genScratch.size() >= 0x4000u)
    {
        // the native's data memory vs the VU1's after the run, per qword (floats within a relative 1e-5)
        uint64_t ok = 0;
        for (uint32_t i = 0; i < 1024u; ++i)
        {
            const uint8_t *a = g_genScratch.data() + i * 16u, *b = data + i * 16u;
            bool same = std::memcmp(a, b, 16) == 0;
            if (!same)
            {
                static int s_dump = 0;
                if (s_dump++ < 8)
                {
                    uint32_t ua[4], ub[4];
                    std::memcpy(ua, a, 16);
                    std::memcpy(ub, b, 16);
                    std::fprintf(stderr, "[nvu1g] %s mem mismatch qw %u: native %08x %08x %08x %08x vu %08x %08x %08x %08x%c", g.name, i,
                                 ua[0], ua[1], ua[2], ua[3], ub[0], ub[1], ub[2], ub[3], 10);
                }
            }
            ok += same;
        }
        g.qwords += 1024u;
        g.okQwords += ok;
        genericReport();
        return;
    }
    // qword compare; FTOI'd integers may differ by 1 (DIV / rounding), floats by a relative 1e-5
    const size_t n = std::min(g_genExpected.size(), g_genCaptured.size()) / 16u;
    uint64_t ok = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const uint8_t *a = g_genExpected.data() + i * 16u, *b = g_genCaptured.data() + i * 16u;
        bool same = std::memcmp(a, b, 16) == 0;
        if (!same)
        {
            same = true;
            for (int c = 0; c < 4 && same; ++c)
            {
                int32_t x, y;
                std::memcpy(&x, a + c * 4, 4);
                std::memcpy(&y, b + c * 4, 4);
                float fx, fy;
                std::memcpy(&fx, &x, 4);
                std::memcpy(&fy, &y, 4);
                same = std::abs(static_cast<int64_t>(x) - y) <= 1 ||
                       (std::isfinite(fx) && std::isfinite(fy) && std::fabs(fx - fy) <= 1e-5f * std::fmax(1.0f, std::fabs(fx)));
            }
        }
        ok += same;
        static int s_dump = 0;
        if (!same && s_dump < 8)
        {
            ++s_dump;
            uint32_t ua[4], ub[4];
            std::memcpy(ua, a, 16);
            std::memcpy(ub, b, 16);
            std::fprintf(stderr, "[nvu1g] %s mismatch qw %zu/%zu: native %08x %08x %08x %08x vu %08x %08x %08x %08x%c", g.name, i, n,
                         ua[0], ua[1], ua[2], ua[3], ub[0], ub[1], ub[2], ub[3], 10);
        }
    }
    if (g_genExpected.size() != g_genCaptured.size())
    {
        static int s_len = 0;
        if (s_len++ < 8)
            std::fprintf(stderr, "[nvu1g] %s packet bytes native %zu vu %zu%c", g.name, g_genExpected.size(), g_genCaptured.size(), 10);
    }
    g.qwords += std::max(g_genExpected.size(), g_genCaptured.size()) / 16u;
    g.okQwords += ok;
    genericReport();
}

#include "hctp_native_vu1_p14.inl"
