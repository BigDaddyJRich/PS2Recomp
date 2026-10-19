// Included at the end of hctp_native_vu1.cpp (uses its Kicks / kickAt / qwf / ilw / Generic verify harness).
//
// p14 My_Code_Shadow (character shadow volumes), overlay 0x2a00. MSCALF 0x2a00 only clears vi9 ([E] at 0x2a08); the
// work is MSCNT-started: 0x2a18 = first batch (loads the constants), 0x3038 = each later batch (B 0x2aa0). Per batch
// (TOP): [TOP].x & 0x7fff vertices as a strip, records of 2 qwords from TOP+1 (vertex index in .w). Per triangle:
// n = (v1-v0) x (v2-v0), n.L ([1008]) sign vs the strip parity (vi4) -> lit triangles only; the three points are
// extruded by L and projected ([1020..1023], perspective divide); 8 triangles (both caps, 3 sides x 2) from the
// stage-1 screen points (+320 stream, w = position z) and the extruded ones; each: -2048 offset, CLIP vs 2047 on
// x/y and z = w + ([1007].x - 2047) -> any vertex outside drops the whole volume (vi10); the screen winding (MAC S.z
// of the cross product) vs the parity -> front list (vi11, in the packet) or back list (vi12, at vi14+8). Packet at
// vi9+160: [1011] / colour 127 / [1009] + front count, front tris; [1011] / 0 / [1009] + back count, back tris;
// [1011] / [1006] / [1010] + (0x8000 | front count), front tris again; XGKICK; vi9 = 80 - vi9. 0x3200 (where the
// 2-D clip case falls through) is a JR back: ShadowClip2D at 0x3210 is not reached from here.
// The registers live in the VU1 state, so a batch the VU1 runs instead carries on the same. Float rules: the VU1
// path's (vu1fast operand / plainResult / exactResult), in the microcode's order.
namespace
{
    using vu1fast::operand;
    float vMul(float a, float b) { return vu1fast::plainResult(operand(a) * operand(b)); }
    float vAdd(float a, float b) { return vu1fast::plainResult(operand(a) + operand(b)); }
    float vSub(float a, float b) { return vu1fast::plainResult(operand(a) - operand(b)); }
    float vMadd(float c, float a, float b)
    {
        c = operand(c);
        a = operand(a);
        b = operand(b);
        return vu1fast::exactResult(c + a * b, static_cast<double>(c) + static_cast<double>(a) * static_cast<double>(b));
    }
    float vMsub(float c, float a, float b)
    {
        c = operand(c);
        a = operand(a);
        b = operand(b);
        return vu1fast::exactResult(c - a * b, static_cast<double>(c) - static_cast<double>(a) * static_cast<double>(b));
    }
    float vDiv(float num, float den)
    {
        num = operand(num);
        den = operand(den);
        if (den == 0.0f)
            return std::signbit(num) != std::signbit(den) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        return vu1fast::result(num / den);
    }
    bool signOf(float v)
    {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        return (b >> 31) != 0u;
    }
    int32_t ftoiVu(float v, float scale) { return vu1fast::floatToInt(operand(v), scale); }
    int32_t i16(int32_t v) { return static_cast<int16_t>(v); }

    struct P14
    {
        float (*vf)[4];
        int32_t *vi;
        float *acc;
        uint32_t clip;
        uint8_t *data;
        Kicks *kicks;
        float q = 0.0f, i = 0.0f;

        float *mem(uint32_t qw) { return qwf(data, qw); }
        void ld(int r, uint32_t qw) { std::memcpy(vf[r], mem(qw), 16); }
        void put(uint32_t qw, const void *v, unsigned mask) // mask bit c = component c (x = bit 0)
        {
            uint8_t *d = data + (qw & 0x3FFu) * 16u;
            for (int c = 0; c < 4; ++c)
                if (mask & (1u << c))
                    std::memcpy(d + c * 4, static_cast<const uint8_t *>(v) + c * 4, 4);
        }
        void isw(uint32_t qw, int c, int32_t v)
        {
            const uint32_t w = static_cast<uint16_t>(v & 0xFFFF);
            std::memcpy(data + (qw & 0x3FFu) * 16u + c * 4, &w, 4);
        }
        int32_t ilwW(int32_t qw) { return ilw(data, static_cast<uint32_t>(qw), 3); }

        // the matrix rows vf28..31: acc = r0*v.x, += r1*v.y, += r2*v.z; v = acc + r3*v.w
        void xform(int r)
        {
            float a[4];
            const float x = vf[r][0], y = vf[r][1], z = vf[r][2], w = vf[r][3];
            for (int c = 0; c < 4; ++c)
                a[c] = vMul(vf[28][c], x);
            for (int c = 0; c < 4; ++c)
                a[c] = vMadd(a[c], vf[29][c], y);
            for (int c = 0; c < 4; ++c)
                a[c] = vMadd(a[c], vf[30][c], z);
            for (int c = 0; c < 4; ++c)
                vf[r][c] = vMadd(a[c], vf[31][c], w);
            std::memcpy(acc, a, 16);
        }

        // OPMULA / OPMSUB: out = a x b (xyz), the products through ACC; returns the sign of z (MAC S.z)
        bool cross(const float *a, const float *b, float *out)
        {
            acc[0] = vMul(a[1], b[2]);
            acc[1] = vMul(a[2], b[0]);
            acc[2] = vMul(a[0], b[1]);
            const float x = vMsub(acc[0], b[1], a[2]), y = vMsub(acc[1], b[2], a[0]), z = vMsub(acc[2], b[0], a[1]);
            out[0] = x;
            out[1] = y;
            out[2] = z;
            return signOf(z);
        }

        // DrawTriangle (0x3048) on vf20 / vf21 / vf22
        void drawTriangle()
        {
            float *v20 = vf[20], *v21 = vf[21], *v22 = vf[22];
            for (int c = 0; c < 4; ++c)
            {
                vf[16][c] = vSub(v21[c], v20[c]);
                vf[17][c] = vSub(v22[c], v20[c]);
            }
            i = vu1fast::fromBits(0xc5000000u); // -2048
            for (int c = 0; c < 2; ++c)
            {
                vf[24][c] = vAdd(v20[c], i);
                vf[25][c] = vAdd(v21[c], i);
                vf[26][c] = vAdd(v22[c], i);
            }
            for (int c = 0; c < 4; ++c)
                vf[3][c] = vAdd(vf[0][c], vf[0][3]);
            acc[2] = vAdd(vf[0][2], v20[3]);
            vf[24][2] = vMadd(acc[2], vf[3][2], vf[15][0]);
            acc[2] = vAdd(vf[0][2], v21[3]);
            vf[25][2] = vMadd(acc[2], vf[3][2], vf[15][0]);
            acc[2] = vAdd(vf[0][2], v22[3]);
            vf[26][2] = vMadd(acc[2], vf[3][2], vf[15][0]);
            clip = ((clip << 6) | vu1fast::clipFlags(vf[24], vf[15])) & 0xFFFFFFu;
            clip = ((clip << 6) | vu1fast::clipFlags(vf[25], vf[15])) & 0xFFFFFFu;
            clip = ((clip << 6) | vu1fast::clipFlags(vf[26], vf[15])) & 0xFFFFFFu;
            const bool sz = cross(vf[16], vf[17], vf[23]);
            vi[3] = 32;
            int32_t t24[4], t25[4], t26[4];
            for (int c = 0; c < 4; ++c)
                t24[c] = ftoiVu(v20[c], 16.0f);
            for (int c = 0; c < 4; ++c)
                t25[c] = ftoiVu(v21[c], 16.0f);
            std::memcpy(vf[24], t24, 16);
            std::memcpy(vf[25], t25, 16);
            const bool behind = (clip & 0x20820u) != 0u;
            vi[1] = (clip & 0xf3cfu) ? 1 : 0;
            if (behind) // a vertex behind (z): dropped, counted (0x31e0)
            {
                vi[10] = i16(vi[10] + 1);
                return;
            }
            for (int c = 0; c < 4; ++c)
                t26[c] = ftoiVu(v22[c], 16.0f);
            std::memcpy(vf[26], t26, 16);
            vi[2] = sz ? 32 : 0; // FMAND (also run in the branch's delay slot)
            if (vi[1] != 0) // off the 2-D range: dropped, counted (0x31f8; 0x3200 returns)
            {
                vi[10] = i16(vi[10] + 1);
                return;
            }
            vi[2] = i16(vi[2] + vi[4]) & vi[3];
            uint32_t at;
            if (vi[2] == 0)
            {
                at = static_cast<uint32_t>(i16(vi[14] + 3 * vi[12] + 8));
                vi[12] = i16(vi[12] + 1);
            }
            else
            {
                at = static_cast<uint32_t>(i16(vi[5] + 3 * vi[11] + 3));
                vi[11] = i16(vi[11] + 1);
            }
            vi[1] = static_cast<int32_t>(at);
            put(at, t24, 15u);
            put(at + 1u, t25, 15u);
            put(at + 2u, t26, 15u);
        }

        void emit(int a, int b, int c)
        {
            for (int k = 0; k < 4; ++k)
                vf[20][k] = vMul(vf[a][k], vf[0][3]);
            std::memcpy(vf[21], vf[b], 16);
            std::memcpy(vf[22], vf[c], 16);
            drawTriangle();
        }

        // the LQI / SQI copy loops (3 qwords per triangle); the registers end as the loop leaves them
        void copyTris(int32_t &from, int32_t &count, uint32_t &to)
        {
            ld(20, static_cast<uint32_t>(from)); // (the first LQI is also in the zero-count branch's delay slot)
            if (count == 0)
            {
                from = i16(from + 1);
                return;
            }
            while (count != 0)
            {
                ld(20, static_cast<uint32_t>(from));
                ld(21, static_cast<uint32_t>(from + 1));
                ld(22, static_cast<uint32_t>(from + 2));
                from = i16(from + 3);
                count = i16(count - 1);
                put(to, vf[20], 15u);
                put(to + 1u, vf[21], 15u);
                put(to + 2u, vf[22], 15u);
                to += 3u;
            }
        }

        void packet()
        {
            for (int c = 0; c < 4; ++c)
                vf[18][c] = vSub(vf[0][c], vf[0][c]);
            i = vu1fast::fromBits(0x42fe0000u); // 127
            float col[4];
            for (int c = 0; c < 4; ++c)
                col[c] = vAdd(vf[0][c], i);
            int32_t c127[4], c0[4];
            for (int c = 0; c < 4; ++c)
            {
                c127[c] = ftoiVu(col[c], 1.0f);
                c0[c] = ftoiVu(vf[18][c], 1.0f);
            }
            std::memcpy(vf[14], c127, 16);
            std::memcpy(vf[18], c0, 16);
            vi[1] = 1006;
            ld(24, 1011u);
            ld(25, 1009u);
            const uint32_t v5 = static_cast<uint32_t>(vi[5]);
            put(v5 + 1u, c127, 15u);
            put(v5, vf[24], 15u);
            put(v5 + 2u, vf[25], 7u);
            isw(v5 + 2u, 3, vi[11]);
            uint32_t v2 = static_cast<uint32_t>(i16(3 * vi[11] + vi[5] + 3));
            put(v2, vf[24], 15u);
            put(v2 + 1u, c0, 15u);
            put(v2 + 2u, vf[25], 7u);
            isw(v2 + 2u, 3, vi[12]);
            v2 += 3u;
            vi[3] = i16(vi[14] + 8);
            copyTris(vi[3], vi[12], v2);
            ld(18, 1006u);
            ld(25, 1010u);
            vi[1] = i16(i16(0x7fff + 1) + vi[11]);
            put(v2, vf[24], 15u);
            put(v2 + 1u, vf[18], 15u);
            put(v2 + 2u, vf[25], 7u);
            isw(v2 + 2u, 3, vi[1]);
            v2 += 3u;
            vi[3] = i16(vi[5] + 3);
            copyTris(vi[3], vi[11], v2);
            vi[2] = 80;
            kickAt(data, static_cast<uint32_t>(vi[5]), *kicks);
            vi[9] = i16(vi[2] - vi[9]);
        }

        // one batch, 0x2aa0 to the [E] at 0x3028; false = input the native doesn't take (checked before any change)
        bool batch(uint32_t top)
        {
            const int32_t header = ilw(data, top, 0);
            const int32_t count = header & 0x7fff;
            if (count < 3 || count > 500)
                return false;
            vi[5] = static_cast<int32_t>(top & 0x3FFu);
            vi[2] = header;
            vi[3] = 0x7fff;
            vi[7] = i16(vi[5] + 1);
            vi[6] = vi[2] & vi[3];
            vi[12] = 0;
            vi[1] = ilwW(vi[7]);
            vi[2] = ilwW(vi[7] + 2);
            vi[7] = i16(vi[7] + 4);
            vi[6] = i16(vi[6] - 2);
            ld(5, static_cast<uint32_t>(vi[1]));
            ld(6, static_cast<uint32_t>(vi[2]));
            vf[24][2] = mem(static_cast<uint32_t>(vi[1]))[2];
            vf[25][2] = mem(static_cast<uint32_t>(vi[2]))[2];
            std::memcpy(vf[8], mem(static_cast<uint32_t>(vi[1] + 320)), 12);
            std::memcpy(vf[9], mem(static_cast<uint32_t>(vi[2] + 320)), 12);
            vf[8][3] = vMul(vf[0][3], vf[24][2]);
            vf[9][3] = vMul(vf[0][3], vf[25][2]);
            vi[4] = 0;
            for (;;)
            {
                // Loop1 (0x2b38): slide the window
                vi[1] = ilwW(vi[7]);
                std::memcpy(vf[4], vf[5], 16);
                std::memcpy(vf[5], vf[6], 16);
                vi[7] = i16(vi[7] + 2);
                ld(6, static_cast<uint32_t>(vi[1]));
                vi[5] = i16(vi[9] + 160);
                for (int c = 0; c < 4; ++c)
                    vf[16][c] = vSub(vf[5][c], vf[4][c]);
                vf[24][2] = mem(static_cast<uint32_t>(vi[1]))[2];
                for (int c = 0; c < 4; ++c)
                    vf[17][c] = vSub(vf[6][c], vf[4][c]);
                std::memcpy(vf[7], vf[8], 16);
                std::memcpy(vf[8], vf[9], 16);
                std::memcpy(vf[9], mem(static_cast<uint32_t>(vi[1] + 320)), 12);
                vf[9][3] = vMul(vf[0][3], vf[24][2]);
                float n[3];
                cross(vf[16], vf[17], n);
                std::memcpy(vf[16], n, 12);
                vi[11] = 0;
                vi[12] = 0;
                for (int c = 0; c < 3; ++c)
                    vf[3][c] = vAdd(vf[0][c], vf[0][3]);
                for (int c = 0; c < 4; ++c)
                    vf[17][c] = vMul(vf[16][c], vf[13][c]);
                acc[2] = vAdd(vf[17][2], vf[17][0]);
                vf[3][2] = vMadd(acc[2], vf[3][2], vf[17][1]);
                vi[3] = 32;
                vi[1] = signOf(vf[3][2]) ? 32 : 0; // FMAND on the MADD's S.z
                for (int c = 0; c < 4; ++c)
                    vf[10][c] = vAdd(vf[4][c], vf[13][c]);
                for (int c = 0; c < 4; ++c)
                    vf[11][c] = vAdd(vf[5][c], vf[13][c]);
                vi[1] = i16(vi[1] + vi[4]);
                for (int c = 0; c < 4; ++c)
                    vf[12][c] = vAdd(vf[6][c], vf[13][c]);
                vi[1] = vi[1] & vi[3];
                if (vi[1] == 0)
                {
                    xform(10);
                    xform(11);
                    xform(12);
                    q = vDiv(vf[0][3], vf[10][3]);
                    for (int c = 0; c < 3; ++c)
                        vf[10][c] = vMul(vf[10][c], q);
                    q = vDiv(vf[0][3], vf[11][3]);
                    for (int c = 0; c < 3; ++c)
                        vf[11][c] = vMul(vf[11][c], q);
                    q = vDiv(vf[0][3], vf[12][3]);
                    vi[10] = 0;
                    for (int c = 0; c < 3; ++c)
                        vf[12][c] = vMul(vf[12][c], q);
                    vi[8] = 1442; // BAL link (the last one: 1470)
                    emit(7, 9, 8);
                    emit(9, 7, 10);
                    emit(9, 10, 12);
                    emit(8, 9, 12);
                    emit(8, 12, 11);
                    emit(7, 8, 11);
                    emit(7, 11, 10);
                    emit(10, 11, 12);
                    vi[8] = 1470;
                    vi[1] = i16(vi[11] + vi[12]);
                    if (vi[1] != 0)
                    {
                        if (vi[10] == 0)
                            packet();
                        else
                        {
                            for (int c = 0; c < 4; ++c)
                                vf[18][c] = vSub(vf[0][c], vf[0][c]);
                            i = vu1fast::fromBits(0x42fe0000u);
                        }
                    }
                }
                // 0x3000
                vi[6] = i16(vi[6] - 1);
                vi[1] = 32;
                vi[4] = i16(vi[1] - vi[4]);
                if (vi[6] == 0)
                    return true;
            }
        }
    };
}

bool hctpNativeVu1MscntP14(const uint8_t *code, uint8_t *data, VU1State &st, uint32_t top, GS &gs)
{
    if (!code || !data || (st.pc != 0x2a18u && st.pc != 0x3038u))
        return false;
    if (!g_genericInit)
    {
        g_genericInit = true;
        for (Generic &g : g_generic)
            g.on = std::getenv(g.env) && std::getenv(g.env)[0] == '1';
    }
    Generic &g = g_generic[sizeof(g_generic) / sizeof(g_generic[0]) - 1];
    if (!g.on && !g_on)
        return false;
    static uint64_t s_sig = 0, s_hash = 0;
    uint64_t sig;
    std::memcpy(&sig, code + 0x2a08u, 8);
    if (sig != s_sig)
    {
        s_sig = sig;
        s_hash = hashRange(code, 0x2a00u, 0x200u);
    }
    uint32_t ret;
    std::memcpy(&ret, code + 0x3200u, 4); // lower word of the pair at 0x3200: JR vi8 (0x48000000 | is << 11)
    if (s_hash != g.hash || (ret & 0xFE000000u) != 0x48000000u)
        return false;
    ++g.runs;

    auto run = [&](uint8_t *mem, VU1State &state, Kicks &kicks) -> bool
    {
        P14 p{state.vf, state.vi, state.acc, state.clip, mem, &kicks};
        p.q = state.q;
        p.i = state.i;
        // the counts are checked before anything is written
        const int32_t count = ilw(mem, top & 0x3FFu, 0) & 0x7fff;
        if (count < 3 || count > 500)
            return false;
        if (state.pc == 0x2a18u)
        {
            p.ld(28, 1020u);
            p.ld(29, 1021u);
            p.ld(30, 1022u);
            p.ld(31, 1023u);
            p.vi[14] = 848;
            p.vi[1] = 1006;
            p.ld(15, 1007u);
            p.ld(1, 1006u);
            p.ld(13, 1008u);
            p.vf[15][3] = vMul(p.vf[0][3], vu1fast::fromBits(0x44ffe000u)); // 2047
            p.vf[15][2] = vAdd(p.vf[0][2], vu1fast::fromBits(0xc5000000u)); // -2048
            p.vf[15][0] = vSub(p.vf[15][0], p.vf[15][3]);
            p.i = vu1fast::fromBits(0xc5000000u);
        }
        if (!p.batch(top & 0x3FFu))
            return false;
        state.clip = p.clip;
        state.q = p.q;
        state.i = p.i;
        state.pc = 0x3038u;
        return true;
    };

    Kicks kicks;
    if (g_on)
    {
        // verify: on copies of the memory and the registers; the VU1 runs for real and its PATH1 output is compared
        // (hctpNativeVu1AfterMscalGeneric after the resume)
        g_genScratch.assign(data, data + 0x4000u);
        VU1State copy = st;
        if (!run(g_genScratch.data(), copy, kicks))
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
    VU1State work = st;
    if (!run(data, work, kicks))
    {
        ++g.fails;
        return false;
    }
    st = work;
    for (const auto &k : kicks)
        gs.processGIFPacket(k.data(), static_cast<uint32_t>(k.size()), 1u);
    ++g.nativeRuns;
    genericReport();
    return true;
}
