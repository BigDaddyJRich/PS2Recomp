// Direct3D 11 GS renderer for the HCTP native-renderer build. Ported from WWE Crush Hour PC (KaikoClanworth1,
// GPL-3.0, https://github.com/KaikoClanworth1/wwe-crush-hour-pc); RenderWare-specific native geometry and the
// Crush Hour mod overlays removed. SubmitNative is the slot for HCTP's own native model draws.
#include "runtime/gs/gs_d3d11_backend.h"
#include "runtime/ps2_host_profile.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "host/win32_host.h"
namespace ps2host
{
    void setPresentWide(bool wide); // win32_host.cpp: Widescreen=1 frame shown 16:9 (else 4:3 with bars)
}
#include "runtime/gs/ps2_gif_arbiter.h"
extern thread_local GifPathId g_gifCurrentPath;

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <nmmintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>
namespace ps2host { uint64_t d3dStateEpoch(); } // win32_host.cpp: bumped when the host draws

using Microsoft::WRL::ComPtr;
using namespace GSInternal;

namespace
{
    std::atomic<GSD3D11Backend::Impl *> g_activeImpl{nullptr};

    // ------------------------------------------------------------------ helpers
    bool isIndexed4(uint8_t psm) { return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH; }
    bool isIndexed8(uint8_t psm) { return psm == GS_PSM_T8 || psm == GS_PSM_T8H; }
    bool isIndexed(uint8_t psm) { return isIndexed4(psm) || isIndexed8(psm); }
    bool is16(uint8_t psm) { return psm == GS_PSM_CT16 || psm == GS_PSM_CT16S || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S; }
    bool is24(uint8_t psm) { return psm == GS_PSM_CT24 || psm == GS_PSM_Z24; }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 stores entries 8..15 and 16..23 of every 32-entry group swapped.
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    void pageSize(uint8_t psm, uint32_t &pw, uint32_t &ph)
    {
        switch (bitsPerPixel(psm))
        {
        case 16: pw = 64; ph = 64; break;
        case 8: pw = 128; ph = 64; break;
        case 4: pw = 128; ph = 128; break;
        default: pw = 64; ph = 32; break;
        }
    }

    // Conservative page span [first, last] (8 KB pages) for a rectangle of a buffer.
    void pageSpan(uint32_t baseBlock, uint32_t bw, uint8_t psm, uint32_t y0, uint32_t h,
                  uint32_t &first, uint32_t &last)
    {
        uint32_t pw, ph;
        pageSize(psm, pw, ph);
        bw = std::max<uint32_t>(bw, 1u);
        const uint32_t pagesPerRow = std::max<uint32_t>(1u, (bw * 64u + pw - 1u) / pw);
        const uint32_t rowStart = y0 / ph;
        const uint32_t rowEnd = (y0 + std::max<uint32_t>(h, 1u) - 1u) / ph;
        first = baseBlock / 32u + rowStart * pagesPerRow;
        last = baseBlock / 32u + (rowEnd + 1u) * pagesPerRow - 1u + ((baseBlock & 31u) ? 1u : 0u);
        first = std::min<uint32_t>(first, 511u);
        last = std::min<uint32_t>(last, 511u);
    }

    bool spansOverlap(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) { return a0 <= b1 && b0 <= a1; }

    // Highest block index (0-31, within its page) a transfer rectangle touches, for the 16/32-bit formats whose
    // block tables we have; 31 (whole page) for anything else or when the rectangle leaves its first page.
    uint32_t maxBlockInRect(uint8_t psm, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h)
    {
        uint32_t bwPx, bhPx, cols, rows;
        const uint8_t *table = nullptr;
        if (psm == 2u || psm == 10u)
        {
            bwPx = 16u; bhPx = 8u; cols = 4u; rows = 8u;
            table = psm == 2u ? &GSPSMCT16::blockTable16[0][0] : &GSPSMCT16::blockTable16S[0][0];
        }
        else if (psm == 0u || psm == 1u)
        {
            bwPx = 8u; bhPx = 8u; cols = 8u; rows = 4u;
            table = &GSPSMCT32::blockTable32[0][0];
        }
        else
            return 31u;
        const uint32_t pw = bwPx * cols, ph = bhPx * rows;
        if (w == 0u || h == 0u || x0 + w > pw || y0 + h > ph)
            return 31u;
        uint32_t best = 0u;
        for (uint32_t r = y0 / bhPx; r <= (y0 + h - 1u) / bhPx; ++r)
            for (uint32_t c = x0 / bwPx; c <= (x0 + w - 1u) / bwPx; ++c)
                best = std::max<uint32_t>(best, table[r * cols + c]);
        return best;
    }

    // pageSpan for a transfer rectangle: an unaligned base only spills into the following page when the
    // rectangle's blocks actually reach it (RenderWare packs small textures right before render targets).
    void pageSpanRect(uint32_t baseBlock, uint32_t bw, uint8_t psm, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h,
                      uint32_t &first, uint32_t &last)
    {
        pageSpan(baseBlock, bw, psm, y0, h, first, last);
        if ((baseBlock & 31u) && last > first && (baseBlock & 31u) + maxBlockInRect(psm, x0, y0, w, h) < 32u)
            last -= 1u; // pageSpan's unconditional spill page is not reached
    }

    uint32_t expand16(uint32_t c, const GSTexaReg &texa, bool applyTexa)
    {
        const uint32_t r = c & 31u, g = (c >> 5) & 31u, b = (c >> 10) & 31u;
        uint32_t out = ((r << 3) | (r >> 2)) | (((g << 3) | (g >> 2)) << 8) | (((b << 3) | (b >> 2)) << 16);
        uint32_t a;
        if (!applyTexa)
            a = (c & 0x8000u) ? 0x80u : 0u;
        else if (c & 0x8000u)
            a = texa.ta1;
        else
            a = (texa.aem && (c & 0x7FFFu) == 0u) ? 0u : texa.ta0;
        return out | (a << 24);
    }

    uint32_t expand24(uint32_t c, const GSTexaReg &texa)
    {
        c &= 0x00FFFFFFu;
        const uint32_t a = (texa.aem && c == 0u) ? 0u : texa.ta0;
        return c | (a << 24);
    }

    uint32_t rd32ee(const uint8_t *ee, uint32_t a)
    {
        uint32_t v;
        std::memcpy(&v, ee + (a & 0x1FFFFFCu), 4);
        return v;
    }

    uint64_t fnv1a(const void *data, size_t n, uint64_t h = 1469598103934665603ull)
    {
        const uint8_t *p = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    // ------------------------------------------------------------------ shaders
    const char *kGsShader = R"(
cbuffer Draw : register(b0)
{
    float4 rtSize;      // native w, h; z: 2D-layer x scale for widescreen (1 = none), w: its centre x
    float4 texDims;     // GS texture w, h (TW/TH), storage w, h (for normalization)
    float4 fogColor;    // 0..255
    float4 p0;          // x tfx, y tcc, z tme, w fge
    float4 p1;          // x ate, y atst, z aref, w afail pass select (0 = keep passing, 1 = keep failing)
    float4 p2;          // x fst, y fba, z tex alpha scale (255 decoded / 128 render target), w dst alpha mode
    float4 region;      // minu, maxu, minv, maxv
    float4 p3;          // x wms, y wmt, z storage origin x, w storage origin y
};
Texture2D gsTex : register(t0);
SamplerState gsSamp : register(s0);

struct VsIn  { float4 pos : POSITION; float4 col : COLOR0; float4 tc : TEXCOORD0; };
struct PsIn  { float4 pos : SV_Position; noperspective float4 col : COLOR0; noperspective float4 tc : TEXCOORD0; };
struct PsOut { float4 c0 : SV_Target0; float4 c1 : SV_Target1; };

PsIn vs_main(VsIn v)
{
    PsIn o;
    const float hx = rtSize.w + (v.pos.x - rtSize.w) * abs(rtSize.z); // widescreen 2D layer (1 otherwise)
    o.pos = float4(hx / rtSize.x * 2.0 - 1.0, 1.0 - v.pos.y / rtSize.y * 2.0, v.pos.z, 1.0);
    o.col = v.col;
    o.tc = v.tc;
    return o;
}

float wrapCoord(float c, float size, float mode, float mn, float mx)
{
    if (mode == 0.0) return c;                                // repeat (sampler wraps)
    if (mode == 1.0) return clamp(c, 0.0, size - 0.001);      // clamp
    if (mode == 2.0) return clamp(c, mn, mx + 0.999);         // region clamp
    uint ic = (uint)floor(c);                                  // region repeat
    return (float)(((ic & (uint)mn) | (uint)mx)) + 0.5;
}

bool alphaTest(float a, float atst, float aref)
{
    if (atst == 0.0) return false;
    if (atst == 1.0) return true;
    if (atst == 2.0) return a < aref;
    if (atst == 3.0) return a <= aref;
    if (atst == 4.0) return a == aref;
    if (atst == 5.0) return a >= aref;
    if (atst == 6.0) return a > aref;
    return a != aref;
}

PsOut ps_main(PsIn i)
{
    float4 c = i.col;   // GS units 0..255, fog in w of tc? (fog packed in col? no: tc.w)
    float fog = i.tc.w;
    if (p0.z != 0.0)
    {
        float2 texel;
        if (p2.x != 0.0) texel = i.tc.xy;
        else texel = (i.tc.xy / i.tc.z) * texDims.xy;
        texel.x = wrapCoord(texel.x, texDims.x, p3.x, region.x, region.y);
        texel.y = wrapCoord(texel.y, texDims.y, p3.y, region.z, region.w);
        float2 uv = (texel + p3.zw) / texDims.zw;
        float4 t = gsTex.SampleLevel(gsSamp, uv, 0);
        t.rgb *= 255.0;
        t.a *= p2.z;
        float tfx = p0.x;
        bool tcc = p0.y != 0.0;
        if (tfx == 0.0)
        {
            c.rgb = t.rgb * c.rgb / 128.0;
            c.a = tcc ? t.a * c.a / 128.0 : c.a;
        }
        else if (tfx == 1.0)
        {
            c.rgb = t.rgb;
            c.a = tcc ? t.a : c.a;
        }
        else if (tfx == 2.0)
        {
            c.rgb = t.rgb * c.rgb / 128.0 + c.a;
            c.a = tcc ? t.a + c.a : c.a;
        }
        else
        {
            c.rgb = t.rgb * c.rgb / 128.0 + c.a;
            c.a = tcc ? t.a : c.a;
        }
        c = min(floor(c), 255.0);
    }
    if (p0.w != 0.0)
        c.rgb = floor((fog * c.rgb + (255.0 - fog) * fogColor.rgb) / 256.0);

    if (p1.x != 0.0)
    {
        bool ok = alphaTest(floor(c.a), p1.y, p1.z);
        if (p1.w == 0.0 && !ok) discard;
        if (p1.w != 0.0 && ok) discard;
    }

    float outA = c.a;
    if (p2.y != 0.0) outA = max(outA, 128.0);
    PsOut o;
    o.c0 = float4(saturate(c.rgb / 255.0), saturate(outA / 128.0));
    o.c1 = float4(0, 0, 0, saturate(c.a / 128.0));
    return o;
}

// ---- utility passes (full-screen triangle) ----
struct BlitOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
BlitOut vs_blit(uint id : SV_VertexID)
{
    BlitOut o;
    float2 p = float2((id << 1) & 2, id & 2);
    o.pos = float4(p * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = p;
    return o;
}
Texture2D blitTex : register(t0);
SamplerState blitSamp : register(s0);
cbuffer Blit : register(b0) { float4 blitRect; };  // uv offset xy, uv scale zw
float4 ps_blit(BlitOut i) : SV_Target
{
    return blitTex.SampleLevel(blitSamp, blitRect.xy + i.uv * blitRect.zw, 0);
}
)";

    struct Vtx
    {
        float x, y, z, w;
        float r, g, b, a;
        float s, t, q, fog;
    };

    struct DrawCB
    {
        float rtSize[4];
        float texDims[4];
        float fogColor[4];
        float p0[4];
        float p1[4];
        float p2[4];
        float region[4];
        float p3[4];
    };

    // Everything that changes how a batch of primitives is rendered.
    struct DrawKey
    {
        uint32_t fbp, fbw, fpsm, fbmsk;
        uint32_t zbp, zpsm, zmsk;
        uint32_t tbp0, tbw, tpsm, tw, th, tcc, tfx, cbp, cpsm, csm, csa;
        uint64_t clamp, alpha, test, tex1, fba;
        uint32_t tme, abe, fge, fst, line;
        uint32_t scx0, scx1, scy0, scy1;
        uint32_t fog;
        uint32_t texa;
        uint32_t clutVersion;
        uint32_t colclamp;
        uint32_t hud; // widescreen 2D-layer x scale (float bits; 0 = none)

        bool operator==(const DrawKey &o) const { return std::memcmp(this, &o, sizeof(DrawKey)) == 0; }
    };
}

struct GSD3D11Backend::Impl
{
    GSCpuBackend mem;
    uint8_t *vram = nullptr;
    uint32_t vramSize = 0;
    bool gpuReady = false;
    bool gpuFailed = false;
    uint32_t scale = 3;

    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<ID3D11VertexShader> vs, vsBlit;
    ComPtr<ID3D11PixelShader> ps, psBlit;
    ComPtr<ID3D11InputLayout> layout;
    ComPtr<ID3D11Buffer> vb, cb, blitCb;
    uint32_t vbCapacity = 0;
    uint32_t vbWriteOffset = 0; // append-only ring: NO_OVERWRITE maps, DISCARD on wrap
    // Draw-state cache for flushPending. Anything else that changes context state (blit, present,
    // latch) bumps pipeGen so the next draw rebinds everything.
    uint64_t pipeGen = 1;
    struct BoundDrawState
    {
        uint64_t gen = 0;
        uint64_t hostEpoch = 0;
        ID3D11RenderTargetView *rtv = nullptr;
        ID3D11DepthStencilView *dsv = nullptr;
        D3D11_VIEWPORT vp{};
        D3D11_RECT sc{};
        ID3D11ShaderResourceView *srv = nullptr;
        ID3D11SamplerState *smp = nullptr;
        ID3D11BlendState *bs = nullptr;
        float bf[4]{};
        ID3D11DepthStencilState *ds = nullptr;
        bool cbValid = false;
        unsigned char cbData[256]{};
    } bound;
    ComPtr<ID3D11SamplerState> samplers[4]; // [linear][repeat]
    ComPtr<ID3D11SamplerState> pointClamp, linearClamp;
    ComPtr<ID3D11RasterizerState> raster;
    std::unordered_map<uint64_t, ComPtr<ID3D11BlendState>> blendStates;
    std::unordered_map<uint32_t, ComPtr<ID3D11DepthStencilState>> depthStates;
    ComPtr<ID3D11BlendState> blendOpaque;
    ComPtr<ID3D11DepthStencilState> depthOff;

    struct RenderTarget
    {
        uint32_t fbpBlock = 0, fbw = 0;
        uint8_t psm = 0;
        uint32_t w = 0, h = 0; // native
        ComPtr<ID3D11Texture2D> tex;
        ComPtr<ID3D11RenderTargetView> rtv;
        ComPtr<ID3D11ShaderResourceView> srv;
        bool gpuDirty = false;
        bool needsUpload = false;
        bool skipNextSync = false; // the pending upload replaces this target's dirty rows: no readback
        // Native rows whose VRAM contents changed since the last upload (inclusive).
        uint32_t upY0 = 0xFFFFFFFFu, upY1 = 0u;
        // Native pixel rows written on the GPU since the last sync (inclusive range).
        uint32_t dirtyY0 = 0xFFFFFFFFu, dirtyY1 = 0u;
        // Native pixel columns written on the GPU since the last sync (inclusive range).
        uint32_t dirtyX0 = 0xFFFFFFFFu, dirtyX1 = 0u;
        uint64_t lastUse = 0;
    };
    std::list<RenderTarget> rts;

    struct DepthTarget
    {
        uint32_t zbpBlock = 0;
        uint32_t w = 0, h = 0;
        ComPtr<ID3D11Texture2D> tex;
        ComPtr<ID3D11DepthStencilView> dsv;
    };
    std::list<DepthTarget> depths;

    struct CachedTexture
    {
        ComPtr<ID3D11Texture2D> tex;
        ComPtr<ID3D11ShaderResourceView> srv;
        uint32_t w = 0, h = 0;
        uint64_t pageStamp = 0;
        uint64_t lastUse = 0;
    };
    std::unordered_map<uint64_t, CachedTexture> texCache;
    std::unordered_map<uint64_t, CachedTexture> contentCache; // keyed by texture params + VRAM page contents

    std::array<uint32_t, 512> pageVersion{};
    // PS2X_STALE_LOG: last host->local upload that touched each page (who overwrote a bound texture).
    struct PageWrite { uint32_t seq = 0, dbp = 0, dbw = 0, dpsm = 0, w = 0, h = 0, y = 0; };
    std::array<PageWrite, 512> pageHist{};
    uint32_t uploadSeq = 0;
    const bool staleLog = std::getenv("PS2X_STALE_LOG") != nullptr;
    // PS2X_CLOBBER_LOG=<n>: the last writer of every 256-byte GS block (host->local uploads, local->local copies,
    // render-target write-backs). A draw sampling a VRAM texture or CLUT whose blocks were last written by something
    // other than its own upload prints one line per (texture, writer) pair, up to n lines.
    struct BlockWriter { uint32_t seq = 0; char kind = 0; uint32_t base = 0, bw = 0, psm = 0, x = 0, y = 0, w = 0, h = 0; uint64_t frame = 0; };
    const int clobberLogInit = std::getenv("PS2X_CLOBBER_LOG") ? std::atoi(std::getenv("PS2X_CLOBBER_LOG")) : 0;
    int clobberLeft = clobberLogInit;
    std::vector<uint32_t> blockLast;     // per block: writer ring index + 1
    std::vector<BlockWriter> writers;    // ring
    uint32_t writerNext = 0;
    std::vector<uint64_t> clobberSeen;

    static uint32_t gsBlockOf(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_T4: return (GSPSMT4::addrPSMT4(base, bw, x, y) >> 9) & 0x3FFFu;
        case GS_PSM_T8: return (GSPSMT8::addrPSMT8(base, bw, x, y) >> 8) & 0x3FFFu;
        case GS_PSM_CT16: case GS_PSM_CT16S: case GS_PSM_Z16: case GS_PSM_Z16S:
            return (GSPSMCT16::addrPSMCT16(base, bw, x, y) >> 8) & 0x3FFFu;
        default: return (GSPSMCT32::addrPSMCT32(base, bw, x, y) >> 8) & 0x3FFFu;
        }
    }
    template <class F>
    static void forEachGsBlock(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t w, uint32_t h, F f)
    {
        uint32_t cw = 8, ch = 8;
        switch (psm)
        {
        case GS_PSM_T4: cw = 32; ch = 16; break;
        case GS_PSM_T8: cw = 16; ch = 16; break;
        case GS_PSM_CT16: case GS_PSM_CT16S: case GS_PSM_Z16: case GS_PSM_Z16S: cw = 16; ch = 8; break;
        default: break;
        }
        if (w == 0u || h == 0u || w > 4096u || h > 4096u)
            return;
        for (uint32_t yy = y & ~(ch - 1u); yy < y + h; yy += ch)
            for (uint32_t xx = x & ~(cw - 1u); xx < x + w; xx += cw)
                f(gsBlockOf(psm, base, std::max<uint32_t>(bw, 1u), xx, yy));
    }
    void recordWriter(char kind, uint32_t base, uint32_t bw, uint32_t psm, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
    {
        if (clobberLogInit <= 0)
            return;
        if (blockLast.empty())
        {
            blockLast.assign(16384u, 0u);
            writers.resize(1u << 16);
        }
        const uint32_t idx = writerNext++ & ((1u << 16) - 1u);
        writers[idx] = {writerNext, kind, base, bw, psm, x, y, w, h, frameCounter};
        forEachGsBlock(psm, base, bw, x, y, w, h, [&](uint32_t b) { blockLast[b] = idx + 1u; });
    }
    void checkClobber(const char *what, uint32_t base, uint32_t bw, uint32_t psm, uint32_t w, uint32_t h, uint32_t tbp0)
    {
        if (clobberLeft <= 0 || blockLast.empty())
            return;
        const BlockWriter *bad = nullptr;
        uint32_t badBlock = 0;
        forEachGsBlock(psm, base, bw, 0, 0, w, h, [&](uint32_t b)
        {
            if (bad || !blockLast[b])
                return;
            const BlockWriter &wr = writers[blockLast[b] - 1u];
            // Uploads and copies are RenderWare's texture cache at work (it batches several textures and CLUTs into
            // one transfer). Suspect: a render-target write-back (this backend's own VRAM write) over a texture other
            // than the target itself.
            if (wr.kind != 'R' || wr.base == base || wr.base == tbp0)
                return;
            bad = &wr;
            badBlock = b;
        });
        if (!bad)
            return;
        const uint64_t key = (static_cast<uint64_t>(tbp0) << 40) ^ (static_cast<uint64_t>(base) << 20) ^
                             (static_cast<uint64_t>(bad->kind) << 56) ^ bad->base;
        if (std::find(clobberSeen.begin(), clobberSeen.end(), key) != clobberSeen.end())
            return;
        clobberSeen.push_back(key);
        --clobberLeft;
        std::printf("[clobber] frame=%llu %s base=%u bw=%u psm=%u %ux%u (tex tbp=%u): block %u last written by %c base=%u bw=%u psm=%u "
                    "rect=(%u,%u %ux%u) seq=%u frame=%llu (now seq %u)%c",
                    static_cast<unsigned long long>(frameCounter), what, base, bw, psm, w, h, tbp0, badBlock, bad->kind, bad->base,
                    bad->bw, bad->psm, bad->x, bad->y, bad->w, bad->h, bad->seq, static_cast<unsigned long long>(bad->frame),
                    writerNext, 10);
        std::fflush(stdout);
    }
    std::array<uint16_t, 512> clut{};
    std::array<uint32_t, 2> clutCbp{0xFFFFFFFFu, 0xFFFFFFFFu};
    uint32_t clutVersion = 0;

    std::vector<Vtx> pending;
    DrawKey pendingKey{};
    // Widescreen=1 (the 3D cameras get a 16:9 view window and the 4:3 frame is shown stretched to 16:9): the 2D
    // layer -- RwIm2D and parallel-projection (HUD camera) draws -- is drawn 3/4 as wide about the frame's centre,
    // so HUD, text and menu panels keep their proportions. Draws nearly as wide as the frame (fades, flashes,
    // the motion blur, full-screen backgrounds) stay full width. PS2X_WS_HUD=0 turns it off.
    float hudScale = 1.0f;
    static bool hudWidescreen()
    {
        static const bool s_on = []()
        {
            const char *v = std::getenv("PS2X_WS_HUD");
            return ps2host::config().widescreen && !(v && *v == '0');
        }();
        return s_on;
    }
    static constexpr float kHudScale = 0.75f;      // (4/3) / (16/9)
    static constexpr float kHudFullWidth = 576.0f; // 90% of the 640-pixel frame
    // Frames without a 3D scene (movies, loading and text screens) are shown 4:3 with side bars instead, their 2D
    // untouched: a frame is "wide" when the previous one drew perspective world / object meshes into the frame;
    // the 2D correction and the presentation aspect use the same flag (latched with the frame in Present).
    bool frame3D = false;     // this frame drew perspective sectors / atomics into a frame-sized target
    bool hudOn = false;       // this frame's 2D correction (hudWidescreen() && the previous frame had 3D)
    bool displayWide = true;  // the latched frame: 16:9 (or 4:3 with bars)
    GSDrawState pendingState{};
    bool hasPending = false;
    uint64_t frameCounter = 0;

    // Feedback copy (texture == render target).
    ComPtr<ID3D11Texture2D> feedbackTex;
    ComPtr<ID3D11ShaderResourceView> feedbackSrv;

    // Latched display frame.
    ComPtr<ID3D11Texture2D> displayTex;
    ComPtr<ID3D11ShaderResourceView> displaySrv;
    uint32_t displayW = 0, displayH = 0;
    float displayU = 1.0f, displayV = 1.0f;
    bool hasDisplay = false;

    uint32_t lastUploadDbp = 0;
    // Page span of the host->local transfer in progress (pixel data arrives after BeginTransfer).
    uint32_t uploadFirst = 1u, uploadLast = 0u;
    uint64_t statNativeDraws = 0;
    uint64_t statDraws = 0, statSyncs = 0, statUploads = 0, statTexDecodes = 0, statTexContentHits = 0, statSkippedSyncs = 0;
    const bool statsEnabled = std::getenv("PS2X_GS_STATS") != nullptr;
    std::unordered_map<uint64_t, uint32_t> decodeHist; // PS2X_GS_STATS: decodes per tbp/psm/size since last report
    std::vector<uint32_t> rawScratch;                  // decodedTexture: raw texels of the texture being looked up
    uint64_t contentEvictFrame = 0;
    const char *syncReason = "?";
    char syncDetail[96] = "";
    const bool syncLogEnabled = std::getenv("PS2X_SYNC_LOG") != nullptr;
    struct ClutCacheEntry
    {
        uint64_t stamp = 0;
        std::array<uint32_t, 256> raw{};
    };
    std::unordered_map<uint64_t, ClutCacheEntry> clutCache;
    // A texture that lies inside a render target (screen-grab effects: the burning-car heat haze samples
    // 256x256 of the frame buffer every frame) is sampled straight from the GPU target. The VRAM path meant
    // a full GPU->CPU readback per such draw (two per frame, match fps 5-20 while cars burn). The stale
    // green car reflections this was once blamed for came from the sceGszbufaddr VRAM-layout bug.
    // PS2X_NO_RT_SUBTEX=1 restores the readback path.
    const bool disableRtSubTex = std::getenv("PS2X_NO_RT_SUBTEX") != nullptr;
    char lastPresent[160] = "none";
    std::unordered_map<std::string, uint64_t> syncReasons;
    std::unordered_map<uint32_t, uint64_t> fmtStats;

    // ---------------------------------------------------------------- setup
    bool initGpu()
    {
        if (gpuReady)
            return true;
        if (gpuFailed || !ps2host::isReady() || !ps2host::device())
            return false;
        gpuFailed = true; // cleared on success
        dev = ps2host::device();
        ctx = ps2host::context();

        auto compile = [&](const char *entry, const char *target, ComPtr<ID3DBlob> &out) -> bool
        {
            ComPtr<ID3DBlob> err;
            if (FAILED(D3DCompile(kGsShader, std::strlen(kGsShader), "gs_d3d11", nullptr, nullptr, entry, target,
                                  D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &out, &err)))
            {
                std::fprintf(stderr, "[gs-d3d11] shader %s failed: %s\n", entry,
                             err ? static_cast<const char *>(err->GetBufferPointer()) : "?");
                return false;
            }
            return true;
        };
        ComPtr<ID3DBlob> vsb, psb, vsbb, psbb;
        if (!compile("vs_main", "vs_5_0", vsb) || !compile("ps_main", "ps_5_0", psb) ||
            !compile("vs_blit", "vs_5_0", vsbb) || !compile("ps_blit", "ps_5_0", psbb))
            return false;
        dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs);
        dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps);
        dev->CreateVertexShader(vsbb->GetBufferPointer(), vsbb->GetBufferSize(), nullptr, &vsBlit);
        dev->CreatePixelShader(psbb->GetBufferPointer(), psbb->GetBufferSize(), nullptr, &psBlit);

        const D3D11_INPUT_ELEMENT_DESC il[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        dev->CreateInputLayout(il, 3, vsb->GetBufferPointer(), vsb->GetBufferSize(), &layout);

        D3D11_BUFFER_DESC cbd{};
        cbd.ByteWidth = sizeof(DrawCB);
        cbd.Usage = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        dev->CreateBuffer(&cbd, nullptr, &cb);
        cbd.ByteWidth = 16;
        dev->CreateBuffer(&cbd, nullptr, &blitCb);

        for (int lin = 0; lin < 2; ++lin)
            for (int rep = 0; rep < 2; ++rep)
            {
                D3D11_SAMPLER_DESC sd{};
                sd.Filter = lin ? D3D11_FILTER_MIN_MAG_MIP_LINEAR : D3D11_FILTER_MIN_MAG_MIP_POINT;
                sd.AddressU = sd.AddressV = sd.AddressW = rep ? D3D11_TEXTURE_ADDRESS_WRAP : D3D11_TEXTURE_ADDRESS_CLAMP;
                sd.MaxLOD = D3D11_FLOAT32_MAX;
                dev->CreateSamplerState(&sd, &samplers[lin * 2 + rep]);
            }
        pointClamp = samplers[0];
        linearClamp = samplers[2];

        D3D11_RASTERIZER_DESC rd{};
        rd.FillMode = D3D11_FILL_SOLID;
        rd.CullMode = D3D11_CULL_NONE;
        rd.ScissorEnable = TRUE;
        rd.DepthClipEnable = FALSE;
        dev->CreateRasterizerState(&rd, &raster);

        D3D11_BLEND_DESC bd{};
        bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        dev->CreateBlendState(&bd, &blendOpaque);
        D3D11_DEPTH_STENCIL_DESC dd{};
        dd.DepthEnable = FALSE;
        dev->CreateDepthStencilState(&dd, &depthOff);

        gpuReady = true;
        gpuFailed = false;
        std::printf("[gs-d3d11] hardware GS renderer ready (internal scale %ux)\n", scale);
        return true;
    }

    // ---------------------------------------------------------------- VRAM <-> RT
    void markPagesDirty(uint32_t first, uint32_t last)
    {
        for (uint32_t p = first; p <= last && p < 512u; ++p)
            ++pageVersion[p];
    }

    void rtPageSpan(const RenderTarget &rt, uint32_t &first, uint32_t &last) const
    {
        pageSpan(rt.fbpBlock, rt.fbw, rt.psm, 0, rt.h, first, last);
    }

    bool createTexture(uint32_t w, uint32_t h, UINT bind, ComPtr<ID3D11Texture2D> &tex,
                       DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM)
    {
        D3D11_TEXTURE2D_DESC td{};
        td.Width = std::max<uint32_t>(w, 1u);
        td.Height = std::max<uint32_t>(h, 1u);
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = fmt;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = bind;
        return SUCCEEDED(dev->CreateTexture2D(&td, nullptr, &tex));
    }

    void blit(ID3D11ShaderResourceView *src, ID3D11RenderTargetView *dst, float dstW, float dstH,
              float u0, float v0, float us, float vs_, bool linear, float dstX = 0.0f, float dstY = 0.0f)
    {
        ++pipeGen;
        D3D11_MAPPED_SUBRESOURCE m{};
        if (SUCCEEDED(ctx->Map(blitCb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
        {
            const float r[4] = {u0, v0, us, vs_};
            std::memcpy(m.pData, r, sizeof(r));
            ctx->Unmap(blitCb.Get(), 0);
        }
        D3D11_VIEWPORT vp{dstX, dstY, dstW, dstH, 0, 1};
        ctx->RSSetViewports(1, &vp);
        D3D11_RECT sc{static_cast<LONG>(dstX), static_cast<LONG>(dstY), static_cast<LONG>(dstX + dstW), static_cast<LONG>(dstY + dstH)};
        ctx->RSSetScissorRects(1, &sc);
        ctx->RSSetState(raster.Get());
        ctx->OMSetRenderTargets(1, &dst, nullptr);
        ctx->OMSetBlendState(blendOpaque.Get(), nullptr, 0xFFFFFFFFu);
        ctx->OMSetDepthStencilState(depthOff.Get(), 0);
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(vsBlit.Get(), nullptr, 0);
        ctx->PSSetShader(psBlit.Get(), nullptr, 0);
        ID3D11Buffer *b = blitCb.Get();
        ctx->PSSetConstantBuffers(0, 1, &b);
        ctx->PSSetShaderResources(0, 1, &src);
        ID3D11SamplerState *s = linear ? linearClamp.Get() : pointClamp.Get();
        ctx->PSSetSamplers(0, 1, &s);
        ctx->Draw(3, 0);
        ID3D11ShaderResourceView *nullSrv = nullptr;
        ctx->PSSetShaderResources(0, 1, &nullSrv);
        ID3D11RenderTargetView *nullRtv = nullptr;
        ctx->OMSetRenderTargets(1, &nullRtv, nullptr);
    }

    // Decode a frame-buffer region of local memory to RGBA8 (alpha normalized as a/128).
    void decodeFrame(uint32_t fbpBlock, uint32_t fbw, uint8_t psm, uint32_t w, uint32_t h, std::vector<uint32_t> &out,
                     uint32_t y0 = 0u)
    {
        out.assign(static_cast<size_t>(w) * h, 0u);
        const uint32_t readPsm = is16(psm) ? psm : (is24(psm) ? GS_PSM_CT24 : GS_PSM_CT32);
        mem.ReadVramRect(readPsm, fbpBlock, fbw, 0u, y0, w, h, out.data());
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
            {
                uint32_t c = out[static_cast<size_t>(y) * w + x];
                uint32_t a;
                if (is16(psm))
                {
                    c = expand16(c, GSTexaReg{}, false);
                    a = c >> 24;
                }
                else if (is24(psm))
                    a = 0x80u;
                else
                    a = c >> 24;
                const uint32_t an = std::min<uint32_t>(255u, (a * 255u + 64u) / 128u);
                out[static_cast<size_t>(y) * w + x] = (c & 0x00FFFFFFu) | (an << 24);
            }
    }

    void uploadRtFromVram(RenderTarget &rt)
    {
        // Only the rows invalidated since the last upload (whole target for a fresh one).
        uint32_t y0 = 0u, y1 = rt.h - 1u;
        if (rt.needsUpload && rt.upY0 != 0xFFFFFFFFu)
        {
            y0 = std::min(rt.upY0, rt.h - 1u);
            y1 = std::min(rt.upY1, rt.h - 1u);
        }
        const uint32_t hh = y1 - y0 + 1u;
        {
            static int s_log = std::getenv("PS2X_RT_UPLOAD_LOG") ? 200 : 0;
            if (s_log > 0)
            {
                --s_log;
                std::printf("[rt-upload] fbp=%u fbw=%u psm=%u %ux%u rows=%u-%u reason=%s lastUpload=%u\n", rt.fbpBlock, rt.fbw, rt.psm,
                            rt.w, rt.h, y0, y1, syncReason ? syncReason : "?", lastUploadDbp);
                std::fflush(stdout);
            }
        }
        std::vector<uint32_t> pixels;
        decodeFrame(rt.fbpBlock, rt.fbw, rt.psm, rt.w, hh, pixels, y0);
        ComPtr<ID3D11Texture2D> staging;
        if (!createTexture(rt.w, hh, D3D11_BIND_SHADER_RESOURCE, staging))
            return;
        ctx->UpdateSubresource(staging.Get(), 0, nullptr, pixels.data(), rt.w * 4u, 0);
        ComPtr<ID3D11ShaderResourceView> srv;
        dev->CreateShaderResourceView(staging.Get(), nullptr, &srv);
        blit(srv.Get(), rt.rtv.Get(), static_cast<float>(rt.w * scale), static_cast<float>(hh * scale),
             0, 0, 1, 1, false, 0.0f, static_cast<float>(y0 * scale));
        rt.needsUpload = false;
        rt.upY0 = 0xFFFFFFFFu;
        rt.upY1 = 0u;
        ++statUploads;
    }

    // Write GPU render-target contents back to local memory (point-sampled down to native).
    void syncRtToVram(RenderTarget &rt)
    {
        if (!rt.gpuDirty)
            return;
        {
            // PS2X_SYNC_LOG=<sec>: log readbacks starting that many seconds after the first one.
            static const char *s_logEnv = std::getenv("PS2X_SYNC_LOG");
            static const double s_logSec = s_logEnv ? std::atof(s_logEnv) : 0.0;
            static const auto s_logBoot = std::chrono::steady_clock::now();
            static int s_logged = 0;
            if (s_logEnv && s_logged < 400 &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - s_logBoot).count() >= s_logSec)
            {
                ++s_logged;
                std::printf("[sync] reason=%s fbp=%u (block %u) fbw=%u psm=%u %ux%u dirtyRows=%u-%u frame=%llu %s\n", syncReason,
                            rt.fbpBlock / 32u, rt.fbpBlock, rt.fbw, rt.psm, rt.w, rt.h, rt.dirtyY0, rt.dirtyY1,
                            static_cast<unsigned long long>(frameCounter), syncDetail);
            }
        }
        flushPending();
        ComPtr<ID3D11Texture2D> nativeTex;
        if (!createTexture(rt.w, rt.h, D3D11_BIND_RENDER_TARGET, nativeTex))
            return;
        ComPtr<ID3D11RenderTargetView> nativeRtv;
        dev->CreateRenderTargetView(nativeTex.Get(), nullptr, &nativeRtv);
        const float half = 0.5f / static_cast<float>(scale);
        blit(rt.srv.Get(), nativeRtv.Get(), static_cast<float>(rt.w), static_cast<float>(rt.h),
             0.0f, 0.0f, 1.0f, 1.0f, false);
        (void)half;

        D3D11_TEXTURE2D_DESC td{};
        nativeTex->GetDesc(&td);
        td.Usage = D3D11_USAGE_STAGING;
        td.BindFlags = 0;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> staging;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &staging)))
            return;
        ctx->CopyResource(staging.Get(), nativeTex.Get());
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m)))
            return;
        // Only the rows the GPU drew. The rest of the target merely overlaps VRAM (e.g. a 256x288 env-map
        // raster runs 32 rows into the next allocation) and holds what was there when the target was
        // created; writing it back stamped stale copies over textures uploaded since (green car / HUD
        // garbage on texture-heavy arenas). PS2X_SYNC_ALL_ROWS=1 restores the old behaviour.
        static const bool s_allRows = std::getenv("PS2X_SYNC_ALL_ROWS") != nullptr;
        // Same for columns: an effect drawn into a corner of a wide target must not write back the rest
        // of its rows (stale textures over the HUD during the car-destroyed sequence).
        const uint32_t yFirst = s_allRows ? 0u : rt.dirtyY0;
        const uint32_t yEnd = s_allRows ? rt.h : std::min(rt.h, rt.dirtyY1 + 1u);
        const uint32_t xFirst = (s_allRows || rt.dirtyX0 == 0xFFFFFFFFu) ? 0u : std::min(rt.dirtyX0, rt.w);
        const uint32_t xEnd = (s_allRows || rt.dirtyX0 == 0xFFFFFFFFu) ? rt.w : std::min(rt.w, rt.dirtyX1 + 1u);
        for (uint32_t y = yFirst; y < yEnd; ++y)
        {
            const uint32_t *row = reinterpret_cast<const uint32_t *>(static_cast<const uint8_t *>(m.pData) + y * m.RowPitch);
            for (uint32_t x = xFirst; x < xEnd; ++x)
            {
                const uint32_t c = row[x];
                const uint32_t aRaw = ((c >> 24) * 128u + 127u) / 255u;
                if (is16(rt.psm))
                {
                    const uint32_t r = (c & 0xFFu) >> 3, g = ((c >> 8) & 0xFFu) >> 3, b = ((c >> 16) & 0xFFu) >> 3;
                    const uint32_t v = r | (g << 5) | (b << 10) | (aRaw >= 0x40u ? 0x8000u : 0u);
                    mem.WriteVram(rt.psm, rt.fbpBlock, rt.fbw, x, y, v);
                }
                else if (is24(rt.psm))
                    mem.WriteVram(GS_PSM_CT24, rt.fbpBlock, rt.fbw, x, y, c & 0x00FFFFFFu);
                else
                    mem.WriteVram(GS_PSM_CT32, rt.fbpBlock, rt.fbw, x, y, (c & 0x00FFFFFFu) | (aRaw << 24));
            }
        }
        ctx->Unmap(staging.Get(), 0);
        if (yEnd > yFirst && xEnd > xFirst)
            recordWriter('R', rt.fbpBlock, rt.fbw, rt.psm, xFirst, yFirst, xEnd - xFirst, yEnd - yFirst);
        rt.gpuDirty = false;
        rt.dirtyY0 = 0xFFFFFFFFu;
        rt.dirtyY1 = 0u;
        rt.dirtyX0 = 0xFFFFFFFFu;
        rt.dirtyX1 = 0u;
        uint32_t first, last;
        rtPageSpan(rt, first, last);
        markPagesDirty(first, last);
        ++statSyncs;
        ++syncReasons[syncReason];
    }

    void syncOverlapping(uint32_t first, uint32_t last)
    {
        for (auto &rt : rts)
        {
            if (rt.skipNextSync)
            {
                rt.skipNextSync = false; // BeginTransfer: this upload rewrites the rows it would read back
                continue;
            }
            if (!rt.gpuDirty)
                continue;
            uint32_t a, b;
            rtPageSpan(rt, a, b);
            if (!spansOverlap(first, last, a, b))
                continue;
            // Only sync when the requested pages intersect rows the GPU actually wrote. Games keep
            // CLUTs and small textures in the unused lines below the visible framebuffer.
            uint32_t pw, ph;
            pageSize(rt.psm, pw, ph);
            const uint32_t pagesPerRow = std::max<uint32_t>(1u, (rt.fbw * 64u + pw - 1u) / pw);
            const uint32_t basePage = rt.fbpBlock / 32u;
            const uint32_t lo = std::max(first, basePage) - basePage;
            const uint32_t hi = std::min(last, b) - basePage;
            const uint32_t rowY0 = (lo / pagesPerRow) * ph;
            const uint32_t rowY1 = (hi / pagesPerRow + 1u) * ph - 1u;
            if (rt.dirtyY0 != 0xFFFFFFFFu && (rowY1 < rt.dirtyY0 || rowY0 > rt.dirtyY1))
                continue;
            syncRtToVram(rt);
        }
    }

    void invalidateOverlapping(uint32_t first, uint32_t last)
    {
        for (auto &rt : rts)
        {
            uint32_t a, b;
            rtPageSpan(rt, a, b);
            if (!spansOverlap(first, last, a, b))
                continue;
            uint32_t pw, ph;
            pageSize(rt.psm, pw, ph);
            const uint32_t pagesPerRow = std::max<uint32_t>(1u, (rt.fbw * 64u + pw - 1u) / pw);
            const uint32_t basePage = rt.fbpBlock / 32u;
            const uint32_t lo = std::max(first, basePage) - basePage;
            const uint32_t hi = std::min(last, b) - basePage;
            const uint32_t rowY0 = (lo / pagesPerRow) * ph;
            const uint32_t rowY1 = (hi / pagesPerRow + 1u) * ph - 1u;
            if (rowY0 >= rt.h)
                continue;
            rt.needsUpload = true;
            rt.upY0 = std::min(rt.upY0, rowY0);
            rt.upY1 = std::max(rt.upY1, std::min(rowY1, rt.h - 1u));
        }
    }

    RenderTarget *findRt(uint32_t fbpBlock)
    {
        for (auto &rt : rts)
            if (rt.fbpBlock == fbpBlock)
                return &rt;
        return nullptr;
    }

    RenderTarget *getRt(uint32_t fbpBlock, uint32_t fbw, uint8_t psm, uint32_t minH)
    {
        fbw = std::max<uint32_t>(fbw, 1u);
        if (psm == GS_PSM_Z32 || psm == GS_PSM_Z24 || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S)
            psm = (psm == GS_PSM_Z16 || psm == GS_PSM_Z16S) ? GS_PSM_CT16 : GS_PSM_CT32;
        const uint32_t w = std::min<uint32_t>(fbw * 64u, 2048u);
        uint32_t h = std::max<uint32_t>(minH, 256u);
        h = std::min<uint32_t>((h + 31u) & ~31u, 2048u);

        RenderTarget *rt = findRt(fbpBlock);
        if (rt && rt->fbw == fbw && bitsPerPixel(rt->psm) == bitsPerPixel(psm) && rt->h >= h)
        {
            rt->psm = psm;
            if (rt->needsUpload)
                uploadRtFromVram(*rt);
            rt->lastUse = frameCounter;
            return rt;
        }
        uint32_t keepH = h;
        if (rt)
        {
            // Format / width / height change: write the old contents back, then rebuild.
            syncReason = "rt-realloc";
            syncRtToVram(*rt);
            if (rt->fbw == fbw)
                keepH = std::max(keepH, rt->h);
            for (auto it = rts.begin(); it != rts.end(); ++it)
                if (&*it == rt)
                {
                    rts.erase(it);
                    break;
                }
        }
        {
            static const bool s_log = std::getenv("PS2X_RT_NEW_LOG") != nullptr;
            static const auto s_t0 = std::chrono::steady_clock::now();
            if (s_log)
            {
                std::printf("[rt-new] t=%.1f fbp=%u fbw=%u psm=%u h=%u rts=%zu frame=%llu\n",
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - s_t0).count(), fbpBlock, fbw,
                            psm, keepH, rts.size(), static_cast<unsigned long long>(frameCounter));
                std::fflush(stdout);
            }
        }
        RenderTarget nrt;
        nrt.fbpBlock = fbpBlock;
        nrt.fbw = fbw;
        nrt.psm = psm;
        nrt.w = w;
        nrt.h = keepH;
        if (!createTexture(w * scale, keepH * scale, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, nrt.tex))
            return nullptr;
        dev->CreateRenderTargetView(nrt.tex.Get(), nullptr, &nrt.rtv);
        dev->CreateShaderResourceView(nrt.tex.Get(), nullptr, &nrt.srv);
        rts.push_back(std::move(nrt));
        RenderTarget &out = rts.back();
        uploadRtFromVram(out);
        out.lastUse = frameCounter;
        return &out;
    }

    DepthTarget *getDepth(uint32_t zbpBlock, uint32_t w, uint32_t h)
    {
        for (auto &d : depths)
            if (d.zbpBlock == zbpBlock)
            {
                if (d.w == w && d.h == h)
                    return &d;
                d.tex.Reset();
                d.dsv.Reset();
                d.w = w;
                d.h = h;
                if (!createTexture(w * scale, h * scale, D3D11_BIND_DEPTH_STENCIL, d.tex, DXGI_FORMAT_D32_FLOAT))
                    return nullptr;
                dev->CreateDepthStencilView(d.tex.Get(), nullptr, &d.dsv);
                ctx->ClearDepthStencilView(d.dsv.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);
                return &d;
            }
        DepthTarget d;
        d.zbpBlock = zbpBlock;
        d.w = w;
        d.h = h;
        if (!createTexture(w * scale, h * scale, D3D11_BIND_DEPTH_STENCIL, d.tex, DXGI_FORMAT_D32_FLOAT))
            return nullptr;
        dev->CreateDepthStencilView(d.tex.Get(), nullptr, &d.dsv);
        ctx->ClearDepthStencilView(d.dsv.Get(), D3D11_CLEAR_DEPTH, 0.0f, 0);
        depths.push_back(std::move(d));
        return &depths.back();
    }

    // ---------------------------------------------------------------- textures
    void loadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
    {
        if (!isIndexed(tex0.psm))
            return;
        switch (tex0.cld)
        {
        case 1: break;
        case 2: clutCbp[0] = tex0.cbp; break;
        case 3: clutCbp[1] = tex0.cbp; break;
        case 4:
            if (clutCbp[0] == tex0.cbp) return;
            clutCbp[0] = tex0.cbp;
            break;
        case 5:
            if (clutCbp[1] == tex0.cbp) return;
            clutCbp[1] = tex0.cbp;
            break;
        default:
            return;
        }
        // The CLUT source may live in a render target that has only been drawn on the GPU.
        uint32_t first, last;
        if (tex0.csm == 0u)
        {
            // CSM1 palettes are tiny: T8 = 16x16 (4 CT32 / 2 CT16 blocks), T4 = 8x2 (1 block). A 64-wide page
            // span reaches into the next page, which is often a camera-texture render target.
            const bool c16c = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
            const uint32_t nblocks = isIndexed4(tex0.psm) ? 1u : (c16c ? 2u : 4u);
            first = tex0.cbp / 32u;
            last = (tex0.cbp + nblocks - 1u) / 32u;
        }
        else
            pageSpan(tex0.cbp, 1, tex0.cpsm, 0, 16, first, last);
        syncReason = "clut";
        if (syncLogEnabled)
            if (syncLogEnabled) std::snprintf(syncDetail, sizeof(syncDetail), "cbp=%u cpsm=%u csm=%u tex psm=%u tbp=%u pages=%u-%u", tex0.cbp, tex0.cpsm,
                          tex0.csm, tex0.psm, tex0.tbp0, first, last);
        syncOverlapping(first, last);
        syncDetail[0] = 0;

        const bool four = isIndexed4(tex0.psm);
        const bool c16 = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
        const bool c32 = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
        if (!c16 && !c32)
            return;
        const uint32_t count = four ? 16u : 256u;
        const uint32_t csaMask = c16 ? 0x1Fu : 0x0Fu;
        const uint32_t dstBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;
        const bool csm1Suffix = tex0.csm == 0u && c32 && !four;
        const uint32_t firstEntry = csm1Suffix ? dstBase : 0u;
        // Palette loads happen on most TEX0 writes; reading 256 swizzled entries each time was ~8% of a
        // match frame. Reuse the raw entries while the source pages are unchanged.
        uint64_t ckey = fnv1a(&tex0.cbp, sizeof(tex0.cbp));
        {
            const uint32_t k[] = {tex0.cpsm, tex0.csm, tex0.csa, four ? 1u : 0u, texclut.cbw, texclut.cou, texclut.cov};
            ckey = fnv1a(k, sizeof(k), ckey);
        }
        const uint64_t cstamp = texturePageStamp(first, last);
        ClutCacheEntry *cached = nullptr;
        {
            auto it = clutCache.find(ckey);
            if (it != clutCache.end() && it->second.stamp == cstamp)
                cached = &it->second;
        }
        ClutCacheEntry fresh{};
        for (uint32_t e = firstEntry; e < count; ++e)
        {
            uint32_t raw;
            if (cached)
                raw = cached->raw[e];
            else
            {
            uint32_t sx, sy, sw = 1u;
            if (tex0.csm == 0u)
            {
                const uint32_t si = swizzleClutIndexCSM1(e);
                sx = si & 0x0Fu;
                sy = si >> 4u;
            }
            else
            {
                sw = texclut.cbw ? texclut.cbw : 1u;
                sx = (static_cast<uint32_t>(texclut.cou) << 4u) + e;
                sy = texclut.cov;
            }
            raw = mem.ReadVram(tex0.cpsm, tex0.cbp, sw, sx, sy);
            fresh.raw[e] = raw;
            }
            const uint32_t d = (csm1Suffix ? e : dstBase + e) & (c16 ? 0x1FFu : 0xFFu);
            if (c16)
                clut[d] = static_cast<uint16_t>(raw);
            else
            {
                clut[d] = static_cast<uint16_t>(raw & 0xFFFFu);
                clut[d + 256u] = static_cast<uint16_t>(raw >> 16u);
            }
        }
        if (!cached)
        {
            if (clutCache.size() > 4096u)
                clutCache.clear();
            fresh.stamp = cstamp;
            clutCache[ckey] = fresh;
        }
        ++clutVersion;
    }

    uint32_t clutColor(const GSTex0Reg &tex0, const GSTexaReg &texa, uint32_t index)
    {
        const bool c16 = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
        const uint32_t base = (static_cast<uint32_t>(tex0.csa) & (c16 ? 0x1Fu : 0x0Fu)) << 4u;
        const uint32_t si = isIndexed4(tex0.psm) ? (index & 0x0Fu) : index;
        uint32_t ci = (base + si) & (c16 ? 0x1FFu : 0xFFu);
        if (!c16 && tex0.csm == 0u && isIndexed8(tex0.psm))
            ci = std::min<uint32_t>((si & 0xF0u) + base, 240u) + (si & 0x0Fu);
        if (c16)
            return expand16(clut[ci], texa, true);
        const uint32_t raw = static_cast<uint32_t>(clut[ci]) | (static_cast<uint32_t>(clut[ci + 256u]) << 16u);
        return tex0.cpsm == GS_PSM_CT24 ? expand24(raw, texa) : raw;
    }

    uint64_t statRasterDecodes = 0, statRasterFallbacks = 0;




    uint64_t texturePageStamp(uint32_t first, uint32_t last) const
    {
        uint64_t s = 0;
        for (uint32_t p = first; p <= last; ++p)
            s = s * 1315423911ull + pageVersion[p];
        return s;
    }

    ID3D11ShaderResourceView *decodedTexture(const GSDrawState &st, uint32_t &outW, uint32_t &outH)
    {
        const GSTex0Reg &t = st.context.tex0;
        const uint32_t tw = st.textureWidth, th = st.textureHeight;
        uint32_t first, last;
        pageSpan(t.tbp0, t.tbw, t.psm, 0, th, first, last);
        syncReason = "texture";
        if (syncLogEnabled)
            if (syncLogEnabled) std::snprintf(syncDetail, sizeof(syncDetail), "tbp=%u tbw=%u psm=%u %ux%u pages=%u-%u", t.tbp0, t.tbw, t.psm, tw, th,
                          first, last);
        syncOverlapping(first, last);
        syncDetail[0] = 0;

        uint64_t key = fnv1a(&t.tbp0, sizeof(t.tbp0));
        const uint32_t k2[] = {t.tbw, t.psm, tw, th};
        key = fnv1a(k2, sizeof(k2), key);
        if (isIndexed(t.psm))
        {
            const uint32_t k3[] = {t.cpsm, t.csa, t.csm};
            key = fnv1a(k3, sizeof(k3), key);
            const bool c16 = t.cpsm == GS_PSM_CT16 || t.cpsm == GS_PSM_CT16S;
            key = fnv1a(clut.data(), c16 ? 512u * 2u : 512u * 2u, key);
        }
        if (is16(t.psm) || is24(t.psm) || (isIndexed(t.psm) && t.cpsm != GS_PSM_CT32))
        {
            const uint32_t k4[] = {st.texa.ta0, st.texa.ta1, st.texa.aem ? 1u : 0u};
            key = fnv1a(k4, sizeof(k4), key);
        }
        // A texture wider than its buffer width spills into the next page row (GS page index is
        // (y/ph)*bufferPages + x/pw), past the span computed from the rows alone.
        {
            uint32_t pw, ph;
            pageSize(t.psm, pw, ph);
            const uint32_t bwp = std::max<uint32_t>(1u, (std::max<uint32_t>(t.tbw, 1u) * 64u + pw - 1u) / pw);
            const uint32_t widest = t.tbp0 / 32u + ((th - 1u) / ph) * bwp + (tw - 1u) / pw + ((t.tbp0 & 31u) ? 1u : 0u);
            last = std::max(last, std::min<uint32_t>(widest, 511u));
        }
        static const bool s_noCache = std::getenv("PS2X_TEX_NOCACHE") != nullptr; // debug: decode every lookup
        const uint64_t stamp = texturePageStamp(first, last);
        auto it = texCache.find(key);
        if (!s_noCache && it != texCache.end() && it->second.pageStamp == stamp)
        {
            it->second.lastUse = frameCounter;
            outW = it->second.w;
            outH = it->second.h;
            return it->second.srv.Get();
        }

        // RenderWare re-streams the same texture bytes into a shared VRAM area every frame, which
        // bumps the page versions. Reuse an earlier decode of identical contents, in two levels:
        // 1) a cheap linear hash of the whole pages (the steady state: same bytes re-streamed);
        // 2) a hash of the texture's own texels. Menu textures share pages, so a neighbour's upload
        //    changes the page hash although this texture is unchanged; without level 2 every texture
        //    in the page was re-decoded (thousands per menu transition).
        const uint32_t tbw = std::max<uint32_t>(t.tbw, 1u);
        std::vector<uint32_t> &raw = rawScratch;
        bool rawRead = false;
        uint64_t contentKey = key;
        uint64_t texelKey = 0;
        if (vram)
        {
            // Hardware CRC32 over four independent lanes (pages are 8 KB, so words is a multiple of 4):
            // this runs on every page-stamp miss, i.e. for most textures every frame.
            const uint64_t *w = reinterpret_cast<const uint64_t *>(vram + static_cast<size_t>(first) * 8192u);
            const size_t words = static_cast<size_t>(last - first + 1u) * 8192u / 8u;
            uint64_t c0 = contentKey, c1 = contentKey >> 16, c2 = contentKey >> 32, c3 = ~contentKey;
            for (size_t i = 0; i < words; i += 4)
            {
                c0 = _mm_crc32_u64(c0, w[i]);
                c1 = _mm_crc32_u64(c1, w[i + 1]);
                c2 = _mm_crc32_u64(c2, w[i + 2]);
                c3 = _mm_crc32_u64(c3, w[i + 3]);
            }
            contentKey = ((c0 << 32) | c1) ^ (((c2 << 32) | c3) * 0x9E3779B97F4A7C15ull) ^ key;
            auto cit = contentCache.find(contentKey);
            if (cit == contentCache.end())
            {
                raw.resize(static_cast<size_t>(tw) * th);
                mem.ReadVramRect(t.psm, t.tbp0, tbw, 0u, 0u, tw, th, raw.data());
                rawRead = true;
                uint64_t th2 = key ^ 0xC2B2AE3D27D4EB4Full;
                for (const uint32_t v : raw)
                {
                    th2 = (th2 ^ v) * 0x100000001B3ull;
                    th2 ^= th2 >> 29;
                }
                texelKey = th2;
                cit = contentCache.find(texelKey);
                if (cit != contentCache.end())
                {
                    contentCache[contentKey] = cit->second; // later page-hash lookups hit level 1
                    cit = contentCache.find(contentKey);
                }
            }
            if (!s_noCache && cit != contentCache.end())
            {
                cit->second.lastUse = frameCounter;
                CachedTexture shared = cit->second;
                shared.pageStamp = stamp;
                texCache[key] = shared;
                outW = shared.w;
                outH = shared.h;
                ++statTexContentHits;
                return shared.srv.Get();
            }
        }

        if (!rawRead)
        {
            raw.resize(static_cast<size_t>(tw) * th);
            mem.ReadVramRect(t.psm, t.tbp0, tbw, 0u, 0u, tw, th, raw.data());
        }
        std::vector<uint32_t> px(static_cast<size_t>(tw) * th);
        for (uint32_t y = 0; y < th; ++y)
            for (uint32_t x = 0; x < tw; ++x)
            {
                const uint32_t v = raw[static_cast<size_t>(y) * tw + x];
                uint32_t c;
                if (isIndexed(t.psm))
                    c = clutColor(t, st.texa, v);
                else if (is16(t.psm))
                    c = expand16(v, st.texa, true);
                else if (is24(t.psm))
                    c = expand24(v, st.texa);
                else
                    c = v;
                px[static_cast<size_t>(y) * tw + x] = c;
            }
        // Always a fresh texture: entries may be shared with contentCache.
        CachedTexture ct;
        if (!createTexture(tw, th, D3D11_BIND_SHADER_RESOURCE, ct.tex))
            return nullptr;
        static const char *s_texDumpDir = std::getenv("PS2X_TEX_DUMP");
        if (const char *dumpDir = s_texDumpDir)
        {
            static int s_texDumps = 0;
            static const long s_skipTbp = std::getenv("PS2X_TEX_DUMP_SKIP") ? std::atol(std::getenv("PS2X_TEX_DUMP_SKIP")) : -1;
            static const uint64_t s_dumpFrame = std::getenv("PS2X_TEX_DUMP_FRAME") ? std::strtoull(std::getenv("PS2X_TEX_DUMP_FRAME"), nullptr, 10) : 0;
            static const long s_onlyTbp = std::getenv("PS2X_TEX_DUMP_TBP") ? std::atol(std::getenv("PS2X_TEX_DUMP_TBP")) : -1;
            static const double s_dumpSec = std::getenv("PS2X_TEX_DUMP_SEC") ? std::atof(std::getenv("PS2X_TEX_DUMP_SEC")) : 0.0;
            static const auto s_dumpBoot = std::chrono::steady_clock::now();
            const bool dumpTimeOk = std::chrono::duration<double>(std::chrono::steady_clock::now() - s_dumpBoot).count() >= s_dumpSec;
            if (s_texDumps < 1500 && static_cast<long>(t.tbp0) != s_skipTbp && frameCounter >= s_dumpFrame && dumpTimeOk &&
                tw * th >= 1024u && (s_onlyTbp < 0 || static_cast<long>(t.tbp0) == s_onlyTbp))
            {
                char path[512];
                std::snprintf(path, sizeof(path), "%s/tex_%04d_tbp%u_psm%u_cpsm%u_%ux%u.rgba", dumpDir, s_texDumps++,
                              t.tbp0, t.psm, t.cpsm, tw, th);
                if (FILE *f = std::fopen(path, "wb"))
                {
                    std::fwrite(px.data(), 4, px.size(), f);
                    std::fclose(f);
                }
            }
        }
        ctx->UpdateSubresource(ct.tex.Get(), 0, nullptr, px.data(), tw * 4u, 0);
        if (!ct.srv)
            dev->CreateShaderResourceView(ct.tex.Get(), nullptr, &ct.srv);
        ct.w = tw;
        ct.h = th;
        ct.pageStamp = stamp;
        ct.lastUse = frameCounter;
        texCache[key] = ct;
        if (vram)
        {
            // Evict idle entries at most once a second: during a menu transition every entry is recent,
            // and a full scan on each insert made texture streaming quadratic.
            if (contentCache.size() > 6000u && frameCounter >= contentEvictFrame + 60u)
            {
                contentEvictFrame = frameCounter;
                for (auto ci = contentCache.begin(); ci != contentCache.end();)
                    ci = (ci->second.lastUse + 300u < frameCounter) ? contentCache.erase(ci) : std::next(ci);
            }
            contentCache[contentKey] = ct;
            if (rawRead)
                contentCache[texelKey] = ct;
        }
        ++statTexDecodes;
        if (statsEnabled)
            ++decodeHist[(static_cast<uint64_t>(t.tbp0) << 40) | (static_cast<uint64_t>(t.psm) << 32) | (tw << 16) | th];
        outW = tw;
        outH = th;
        return texCache[key].srv.Get();
    }

    // ---------------------------------------------------------------- state objects
    ID3D11BlendState *blendState(const GSDrawState &st, bool afailRgbOnly, bool colorOff, float blendFactor[4])
    {
        const auto &c = st.context;
        uint32_t mask = D3D11_COLOR_WRITE_ENABLE_ALL;
        const uint32_t fbmsk = c.frame.fbmsk;
        if ((fbmsk & 0x000000FFu) == 0x000000FFu) mask &= ~D3D11_COLOR_WRITE_ENABLE_RED;
        if ((fbmsk & 0x0000FF00u) == 0x0000FF00u) mask &= ~D3D11_COLOR_WRITE_ENABLE_GREEN;
        if ((fbmsk & 0x00FF0000u) == 0x00FF0000u) mask &= ~D3D11_COLOR_WRITE_ENABLE_BLUE;
        if ((fbmsk & 0xFF000000u) == 0xFF000000u || is24(c.frame.psm) || afailRgbOnly)
            mask &= ~D3D11_COLOR_WRITE_ENABLE_ALPHA;
        if (colorOff)
            mask = 0;

        const bool abe = st.prim.abe;
        const uint32_t a = static_cast<uint32_t>(c.alpha & 3u), b = static_cast<uint32_t>((c.alpha >> 2) & 3u);
        const uint32_t cs = static_cast<uint32_t>((c.alpha >> 4) & 3u), d = static_cast<uint32_t>((c.alpha >> 6) & 3u);
        const uint32_t fix = static_cast<uint32_t>((c.alpha >> 32) & 0xFFu);
        const float f = std::min(1.0f, static_cast<float>(fix) / 128.0f);
        blendFactor[0] = blendFactor[1] = blendFactor[2] = blendFactor[3] = f;

        const uint64_t key = (abe ? 1ull : 0ull) | (static_cast<uint64_t>(a) << 1) | (static_cast<uint64_t>(b) << 3) |
                             (static_cast<uint64_t>(cs) << 5) | (static_cast<uint64_t>(d) << 7) |
                             (static_cast<uint64_t>(mask) << 9);
        auto it = blendStates.find(key);
        if (it != blendStates.end())
            return it->second.Get();

        D3D11_BLEND_DESC bd{};
        auto &rt = bd.RenderTarget[0];
        rt.RenderTargetWriteMask = static_cast<UINT8>(mask);
        if (abe)
        {
            // Result = (A - B) * C + D with A,B,D in {Cs, Cd, 0}; express as k*src (+|-) m*dst.
            auto coef = [&](uint32_t which, int &k1, int &k0)
            {
                k1 = (a == which ? 1 : 0) - (b == which ? 1 : 0);
                k0 = (d == which ? 1 : 0);
            };
            int s1, s0, d1, d0;
            coef(0, s1, s0);
            coef(1, d1, d0);
            const D3D11_BLEND cf = cs == 0 ? D3D11_BLEND_SRC1_ALPHA : cs == 1 ? D3D11_BLEND_DEST_ALPHA : D3D11_BLEND_BLEND_FACTOR;
            const D3D11_BLEND icf = cs == 0 ? D3D11_BLEND_INV_SRC1_ALPHA : cs == 1 ? D3D11_BLEND_INV_DEST_ALPHA : D3D11_BLEND_INV_BLEND_FACTOR;
            auto factor = [&](int k1, int k0, bool &negative) -> D3D11_BLEND
            {
                negative = false;
                if (k1 == 0) return k0 ? D3D11_BLEND_ONE : D3D11_BLEND_ZERO;
                if (k1 == 1) return k0 ? D3D11_BLEND_ONE : cf; // 1 + C approximated as ONE
                if (k0 == 1) return icf;                         // 1 - C
                negative = true;                                 // -C
                return cf;
            };
            bool sneg, dneg;
            rt.SrcBlend = factor(s1, s0, sneg);
            rt.DestBlend = factor(d1, d0, dneg);
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            if (sneg && !dneg)
                rt.BlendOp = D3D11_BLEND_OP_REV_SUBTRACT;
            else if (dneg && !sneg)
                rt.BlendOp = D3D11_BLEND_OP_SUBTRACT;
            rt.BlendEnable = TRUE;
            rt.SrcBlendAlpha = D3D11_BLEND_ONE;
            rt.DestBlendAlpha = D3D11_BLEND_ZERO;
            rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
        }
        ComPtr<ID3D11BlendState> bs;
        dev->CreateBlendState(&bd, &bs);
        blendStates[key] = bs;
        return bs.Get();
    }

    ID3D11DepthStencilState *depthState(bool enable, uint32_t func, bool write)
    {
        const uint32_t key = (enable ? 1u : 0u) | (func << 1) | (write ? 8u : 0u);
        auto it = depthStates.find(key);
        if (it != depthStates.end())
            return it->second.Get();
        D3D11_DEPTH_STENCIL_DESC dd{};
        dd.DepthEnable = enable ? TRUE : FALSE;
        dd.DepthWriteMask = write ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
        static const D3D11_COMPARISON_FUNC funcs[4] = {D3D11_COMPARISON_NEVER, D3D11_COMPARISON_ALWAYS,
                                                       D3D11_COMPARISON_GREATER_EQUAL, D3D11_COMPARISON_GREATER};
        dd.DepthFunc = funcs[func & 3u];
        ComPtr<ID3D11DepthStencilState> ds;
        dev->CreateDepthStencilState(&dd, &ds);
        depthStates[key] = ds;
        return ds.Get();
    }

    // ---------------------------------------------------------------- batching
    DrawKey makeKey(const GSDrawState &st) const
    {
        DrawKey k{};
        const auto &c = st.context;
        k.fbp = c.frame.fbp;
        k.fbw = c.frame.fbw;
        k.fpsm = c.frame.psm;
        k.fbmsk = c.frame.fbmsk;
        k.zbp = c.zbuf.zbp;
        k.zpsm = c.zbuf.psm;
        k.zmsk = c.zbuf.zmask ? 1u : 0u;
        k.tme = st.prim.tme ? 1u : 0u;
        if (k.tme)
        {
            k.tbp0 = c.tex0.tbp0;
            k.tbw = c.tex0.tbw;
            k.tpsm = c.tex0.psm;
            k.tw = c.tex0.tw;
            k.th = c.tex0.th;
            k.tcc = c.tex0.tcc;
            k.tfx = c.tex0.tfx;
            k.cbp = c.tex0.cbp;
            k.cpsm = c.tex0.cpsm;
            k.csm = c.tex0.csm;
            k.csa = c.tex0.csa;
            k.clamp = c.clamp;
            k.tex1 = c.tex1;
            k.fst = st.prim.fst ? 1u : 0u;
            k.texa = st.texa.ta0 | (st.texa.ta1 << 8) | (st.texa.aem ? 0x10000u : 0u);
            k.clutVersion = isIndexed(c.tex0.psm) ? clutVersion : 0u;
        }
        k.abe = st.prim.abe ? 1u : 0u;
        if (k.abe)
            k.alpha = c.alpha;
        k.test = c.test;
        k.fba = c.fba;
        k.fge = st.prim.fge ? 1u : 0u;
        if (k.fge)
            k.fog = st.fogR | (st.fogG << 8) | (st.fogB << 16);
        k.line = (st.prim.type == GS_PRIM_LINE || st.prim.type == GS_PRIM_LINESTRIP) ? 1u : 0u;
        k.scx0 = c.scissor.x0;
        k.scx1 = c.scissor.x1;
        k.scy0 = c.scissor.y0;
        k.scy1 = c.scissor.y1;
        if (hudScale != 1.0f)
            std::memcpy(&k.hud, &hudScale, 4);
        return k;
    }

    static float zNorm(double z, uint8_t zpsm)
    {
        double maxz = 4294967296.0;
        if (zpsm == GS_PSM_Z24) maxz = 16777216.0;
        else if (zpsm == GS_PSM_Z16 || zpsm == GS_PSM_Z16S) maxz = 65536.0;
        return static_cast<float>(std::clamp(z / maxz, 0.0, 1.0));
    }

    void submit(const GSPrimitiveBatch &batch)
    {
        const GSDrawState &st = batch.state;
        if (st.prim.type == GS_PRIM_POINT)
            return;
        const DrawKey key = makeKey(st);
        if (hasPending && !(key == pendingKey))
            flushPending();
        if (!hasPending)
        {
            pendingKey = key;
            pendingState = st;
            hasPending = true;
        }

        const auto &c = st.context;
        const float ofx = static_cast<float>(c.xyoffset.ofx) / 16.0f;
        const float ofy = static_cast<float>(c.xyoffset.ofy) / 16.0f;
        const uint8_t zpsm = c.zbuf.psm;
        auto mk = [&](const GSVertex &v, const GSVertex &colorSrc) -> Vtx
        {
            Vtx o{};
            o.x = v.x - ofx;
            o.y = v.y - ofy;
            o.z = zNorm(v.z, zpsm);
            o.w = 1.0f;
            o.r = colorSrc.r;
            o.g = colorSrc.g;
            o.b = colorSrc.b;
            o.a = colorSrc.a;
            if (st.prim.fst)
            {
                o.s = static_cast<float>(v.u) / 16.0f;
                o.t = static_cast<float>(v.v) / 16.0f;
                o.q = 1.0f;
            }
            else
            {
                o.s = v.s;
                o.t = v.t;
                o.q = (v.q == 0.0f) ? 1.0f : v.q;
            }
            o.fog = st.prim.fge ? static_cast<float>(colorSrc.fog) : 255.0f;
            return o;
        };

        switch (st.prim.type)
        {
        case GS_PRIM_SPRITE:
        {
            const GSVertex &v0 = batch.vertices[0], &v1 = batch.vertices[1];
            Vtx a = mk(v0, v1), b = mk(v1, v1);
            a.z = b.z;
            if (!st.prim.fst)
            {
                // Sprites interpolate S/Q and T/Q per corner; resolve to affine texels.
                a.s /= a.q; a.t /= a.q; a.q = 1.0f;
                b.s /= b.q; b.t /= b.q; b.q = 1.0f;
            }
            Vtx tr = a, bl = a;
            tr.x = b.x; tr.s = b.s;
            bl.y = b.y; bl.t = b.t;
            pending.push_back(a);
            pending.push_back(tr);
            pending.push_back(bl);
            pending.push_back(tr);
            pending.push_back(b);
            pending.push_back(bl);
            break;
        }
        case GS_PRIM_TRIANGLE:
        case GS_PRIM_TRISTRIP:
        case GS_PRIM_TRIFAN:
        {
            const GSVertex &flat = batch.vertices[2];
            for (int i = 0; i < 3; ++i)
                pending.push_back(mk(batch.vertices[i], st.prim.iip ? batch.vertices[i] : flat));
            break;
        }
        case GS_PRIM_LINE:
        case GS_PRIM_LINESTRIP:
        {
            // Lines become thin quads (one native pixel wide).
            const GSVertex &flat = batch.vertices[1];
            Vtx a = mk(batch.vertices[0], st.prim.iip ? batch.vertices[0] : flat);
            Vtx b = mk(batch.vertices[1], flat);
            const float dx = b.x - a.x, dy = b.y - a.y;
            const float len = std::max(0.001f, std::sqrt(dx * dx + dy * dy));
            const float nx = -dy / len * 0.5f, ny = dx / len * 0.5f;
            Vtx a0 = a, a1 = a, b0 = b, b1 = b;
            a0.x += nx; a0.y += ny; a1.x -= nx; a1.y -= ny;
            b0.x += nx; b0.y += ny; b1.x -= nx; b1.y -= ny;
            pending.push_back(a0); pending.push_back(b0); pending.push_back(a1);
            pending.push_back(b0); pending.push_back(b1); pending.push_back(a1);
            break;
        }
        default:
            break;
        }
        if (pending.size() > 60000)
            flushPending();
    }

    // Vertex / index buffers for a RenderWare world sector mesh, built once from EE memory.
    // Upload n bytes into a dynamic buffer, growing it as needed.
    bool uploadDynamic(ComPtr<ID3D11Buffer> &buf, uint32_t &cap, UINT bind, const void *data, uint32_t bytes)
    {
        if (!buf || cap < bytes)
        {
            D3D11_BUFFER_DESC bd{};
            bd.ByteWidth = std::max<uint32_t>(bytes, 256u << 10);
            bd.Usage = D3D11_USAGE_DYNAMIC;
            bd.BindFlags = bind;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            buf.Reset();
            if (FAILED(dev->CreateBuffer(&bd, nullptr, &buf)))
                return false;
            cap = bd.ByteWidth;
        }
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(ctx->Map(buf.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
            return false;
        std::memcpy(m.pData, data, bytes);
        ctx->Unmap(buf.Get(), 0);
        return true;
    }









    void flushPending()
    {
        if (!hasPending)
            return;
        ps2_host_profile::Scope profScope(ps2_host_profile::GsDraw);
        hasPending = false;
        if (pending.empty() || !gpuReady)
        {
            pending.clear();
            return;
        }
        const GSDrawState &st = pendingState;
        const auto &c = st.context;

        const uint32_t fbpBlock = framePageBaseToBlock(c.frame.fbp);
        // Size by what is actually drawn: a full-VRAM scissor would otherwise grow the frame buffer
        // over the texture area, and every texture upload then forces an RT re-upload.
        float maxY = 0.0f;
        for (const Vtx &v : pending)
            maxY = std::max(maxY, v.y);
        const uint32_t drawnH = static_cast<uint32_t>(std::max(0.0f, std::ceil(maxY))) + 1u;
        const uint32_t needH = std::min<uint32_t>({2048u, static_cast<uint32_t>(c.scissor.y1) + 1u, drawnH});
        RenderTarget *rt = getRt(fbpBlock, c.frame.fbw, c.frame.psm, needH);
        if (!rt)
        {
            pending.clear();
            return;
        }

        // ---- texture
        ID3D11ShaderResourceView *texSrv = nullptr;
        bool texFromRt = false;
        float texDims[4] = {1, 1, 1, 1};
        float origin[2] = {0, 0};
        float alphaScale = 255.0f;
        if (st.prim.tme)
        {
            const auto &t = c.tex0;
            texDims[0] = static_cast<float>(st.textureWidth);
            texDims[1] = static_cast<float>(st.textureHeight);
            RenderTarget *srcRt = !isIndexed(t.psm) ? findRt(t.tbp0) : nullptr;
            if (!srcRt && !isIndexed(t.psm) && !disableRtSubTex)
            {
                // Texture at a page-aligned offset inside a render target (e.g. a screen effect sampling
                // part of the frame buffer): sample the GPU target with an origin instead of reading back.
                for (auto &r : rts)
                {
                    if (t.tbp0 <= r.fbpBlock || ((t.tbp0 - r.fbpBlock) & 31u) != 0u || r.fbw != std::max<uint32_t>(t.tbw, 1u) ||
                        bitsPerPixel(r.psm) != bitsPerPixel(t.psm) || r.needsUpload)
                        continue;
                    uint32_t pw, ph;
                    pageSize(r.psm, pw, ph);
                    const uint32_t pagesPerRow = std::max<uint32_t>(1u, (r.fbw * 64u + pw - 1u) / pw);
                    const uint32_t page = (t.tbp0 - r.fbpBlock) / 32u;
                    const uint32_t ox = (page % pagesPerRow) * pw, oy = (page / pagesPerRow) * ph;
                    if (oy >= r.h || ox + st.textureWidth > r.w || oy + st.textureHeight > r.h)
                        continue;
                    srcRt = &r;
                    origin[0] = static_cast<float>(ox);
                    origin[1] = static_cast<float>(oy);
                    break;
                }
            }
            {
                static const bool s_count = std::getenv("PS2X_RTTEX_LOG") != nullptr;
                if (s_count)
                {
                    static uint64_t s_into = 0, s_from = 0, s_fromRt = 0;
                    static auto s_last = std::chrono::steady_clock::now();
                    if (framePageBaseToBlock(c.frame.fbp) == 14336u)
                        ++s_into;
                    if (t.tbp0 == 14336u)
                    {
                        ++s_from;
                        if (srcRt)
                            ++s_fromRt;
                    }
                    if (std::chrono::steady_clock::now() - s_last > std::chrono::seconds(2))
                    {
                        s_last = std::chrono::steady_clock::now();
                        std::printf("[rt14336] drawsInto=%llu samples=%llu samplesFromRt=%llu\n", (unsigned long long)s_into,
                                    (unsigned long long)s_from, (unsigned long long)s_fromRt);
                        std::fflush(stdout);
                        s_into = s_from = s_fromRt = 0;
                    }
                }
                static int s_any = std::getenv("PS2X_RTTEX_LOG") ? 5 : 0;
                if (s_any > 0 && t.tbp0 >= 14000u && t.tbp0 < 15000u && t.psm == 0u)
                {
                    --s_any;
                    std::printf("[rt-tex-any] tbp=%u psm=%u srcRt=%p rts=%zu\n", t.tbp0, t.psm, (void *)srcRt, rts.size());
                    std::fflush(stdout);
                }
                static int s_rtTexLog = std::getenv("PS2X_RTTEX_LOG") ? 40 : 0;
                if (s_rtTexLog > 0 && !srcRt && t.tbp0 == 14336u)
                {
                    --s_rtTexLog;
                    std::printf("[rt-tex] tbp=%u: no RT; rts:", t.tbp0);
                    for (const auto &r : rts)
                        std::printf(" [fbp=%u fbw=%u psm=%u %ux%u]", r.fbpBlock, r.fbw, r.psm, r.w, r.h);
                    std::printf("\n");
                    std::fflush(stdout);
                }
                if (s_rtTexLog > 0 && srcRt && t.tbp0 == 14336u)
                {
                    --s_rtTexLog;
                    std::printf("[rt-tex] tbp=%u tbw=%u psm=%u -> rt fbw=%u psm=%u %ux%u needsUpload=%d gpuDirty=%d target=%u\n", t.tbp0,
                                t.tbw, t.psm, srcRt->fbw, srcRt->psm, srcRt->w, srcRt->h, srcRt->needsUpload ? 1 : 0,
                                srcRt->gpuDirty ? 1 : 0, rt ? rt->fbpBlock : 0u);
                    std::fflush(stdout);
                }
            }
            {
                // PS2X_RTFALLBACK_LOG=<n>: a texture at a render target's address that is decoded from VRAM instead.
                static int s_fbLog = std::getenv("PS2X_RTFALLBACK_LOG") ? std::atoi(std::getenv("PS2X_RTFALLBACK_LOG")) : 0;
                if (s_fbLog > 0 && srcRt &&
                    !(srcRt->fbw == std::max<uint32_t>(t.tbw, 1u) && bitsPerPixel(srcRt->psm) == bitsPerPixel(t.psm) && !srcRt->needsUpload))
                {
                    --s_fbLog;
                    std::printf("[rt-fallback] frame=%llu tex tbp=%u tbw=%u psm=%u %ux%u -> rt fbw=%u psm=%u %ux%u needsUpload=%d up=%u-%u%c",
                                static_cast<unsigned long long>(frameCounter), t.tbp0, t.tbw, t.psm, st.textureWidth, st.textureHeight,
                                srcRt->fbw, srcRt->psm, srcRt->w, srcRt->h, srcRt->needsUpload ? 1 : 0, srcRt->upY0, srcRt->upY1, 10);
                    std::fflush(stdout);
                }
            }
            if (srcRt && srcRt->fbw == std::max<uint32_t>(t.tbw, 1u) && bitsPerPixel(srcRt->psm) == bitsPerPixel(t.psm) && !srcRt->needsUpload)
            {
                texDims[2] = static_cast<float>(srcRt->w);
                texDims[3] = static_cast<float>(srcRt->h);
                alphaScale = 128.0f;
                if (srcRt == rt)
                {
                    // Reading the target being drawn: sample a snapshot copy.
                    D3D11_TEXTURE2D_DESC td{};
                    rt->tex->GetDesc(&td);
                    D3D11_TEXTURE2D_DESC fd{};
                    if (feedbackTex)
                        feedbackTex->GetDesc(&fd);
                    if (!feedbackTex || fd.Width != td.Width || fd.Height != td.Height)
                    {
                        feedbackTex.Reset();
                        feedbackSrv.Reset();
                        createTexture(td.Width, td.Height, D3D11_BIND_SHADER_RESOURCE, feedbackTex);
                        dev->CreateShaderResourceView(feedbackTex.Get(), nullptr, &feedbackSrv);
                    }
                    ctx->CopyResource(feedbackTex.Get(), rt->tex.Get());
                    texSrv = feedbackSrv.Get();
                    texFromRt = true;
                }
                else
                    texSrv = srcRt->srv.Get();
                    texFromRt = true;
            }
            else
            {
                uint32_t w = 1, h = 1;
                texSrv = nullptr;
                if (!texSrv)
                {
                    texSrv = decodedTexture(st, w, h);
                    if (clobberLeft > 0)
                    {
                        const uint32_t cw = std::min<uint32_t>(st.textureWidth, std::max<uint32_t>(t.tbw, 1u) * 64u);
                        checkClobber("texture", t.tbp0, t.tbw, t.psm, cw, st.textureHeight, t.tbp0);
                        if (isIndexed(t.psm))
                            checkClobber("clut", t.cbp, 1u, t.cpsm, isIndexed8(t.psm) ? 16u : 8u, isIndexed8(t.psm) ? 16u : 2u, t.tbp0);
                    }
                }
                static const long s_staleTbp = std::getenv("PS2X_STALE_TBP") ? std::atol(std::getenv("PS2X_STALE_TBP")) : -1;
                if (staleLog && (t.psm == 10u || t.psm == 2u || static_cast<long>(t.tbp0) == s_staleTbp))
                {
                    // 16-bit textures are uploaded natively, so the last writer of their first page should be
                    // a same-format upload starting at TBP0; anything else means the draw reads another texture.
                    // PS2X_STALE_TBP=<tbp> reports every new writer state for that texture address.
                    static int s_left = 80;
                    static uint32_t s_lastSeqForTbp = ~0u;
                    const uint32_t page = std::min<uint32_t>(511u, t.tbp0 / 32u);
                    const PageWrite &pw = pageHist[page];
                    const bool watched = static_cast<long>(t.tbp0) == s_staleTbp && pw.seq != s_lastSeqForTbp;
                    if (watched)
                        s_lastSeqForTbp = pw.seq;
                    if (s_left > 0 && (watched || (t.psm == 10u || t.psm == 2u) && (pw.seq == 0u || pw.dpsm != t.psm || pw.dbp != t.tbp0)))
                    {
                        --s_left;
                        std::printf("[stale] draw tbp=%u tbw=%u psm=%u %ux%u <- page %u last upload #%u (now #%u): dbp=%u dbw=%u dpsm=%u %ux%u y=%u\n",
                                    t.tbp0, t.tbw, t.psm, w, h, page, pw.seq, uploadSeq, pw.dbp, pw.dbw, pw.dpsm, pw.w, pw.h, pw.y);
                        std::fflush(stdout);
                    }
                }
                texDims[2] = static_cast<float>(w);
                texDims[3] = static_cast<float>(h);
            }
        }

        // ---- depth
        const uint32_t ztest = static_cast<uint32_t>((c.test >> 16) & 1u);
        const uint32_t ztst = static_cast<uint32_t>((c.test >> 17) & 3u);
        const bool depthNeeded = !(ztest && ztst == 1u && c.zbuf.zmask) && !(!ztest && c.zbuf.zmask);
        DepthTarget *dt = nullptr;
        if (depthNeeded)
            dt = getDepth(framePageBaseToBlock(c.zbuf.zbp), rt->w, rt->h);
        const uint32_t zfunc = ztest ? ztst : 1u;

        // ---- constants
        DrawCB cbd{};
        cbd.rtSize[0] = static_cast<float>(rt->w);
        cbd.rtSize[1] = static_cast<float>(rt->h);
        {
            float hs = 1.0f;
            if (pendingKey.hud)
                std::memcpy(&hs, &pendingKey.hud, 4);
            if (std::max<uint32_t>(c.frame.fbw, 1u) * 64u < 640u)
                hs = 1.0f; // a camera texture (the menu panels' 512x256 raster, the arena screen's feed), not the frame
            cbd.rtSize[2] = hs;
            cbd.rtSize[3] = 0.5f * static_cast<float>(std::min<uint32_t>(rt->w, std::max<uint32_t>(c.frame.fbw, 1u) * 64u));
        }
        std::memcpy(cbd.texDims, texDims, sizeof(texDims));
        cbd.fogColor[0] = st.fogR;
        cbd.fogColor[1] = st.fogG;
        cbd.fogColor[2] = st.fogB;
        cbd.p0[0] = static_cast<float>(c.tex0.tfx);
        cbd.p0[1] = static_cast<float>(c.tex0.tcc);
        cbd.p0[2] = (st.prim.tme && texSrv) ? 1.0f : 0.0f;
        cbd.p0[3] = st.prim.fge ? 1.0f : 0.0f;
        const uint32_t ate = static_cast<uint32_t>(c.test & 1u);
        const uint32_t atst = static_cast<uint32_t>((c.test >> 1) & 7u);
        const uint32_t aref = static_cast<uint32_t>((c.test >> 4) & 0xFFu);
        const uint32_t afail = static_cast<uint32_t>((c.test >> 12) & 3u);
        cbd.p1[0] = ate ? 1.0f : 0.0f;
        cbd.p1[1] = static_cast<float>(atst);
        cbd.p1[2] = static_cast<float>(aref);
        cbd.p2[0] = st.prim.fst ? 1.0f : 0.0f;
        cbd.p2[1] = (c.fba & 1u) ? 1.0f : 0.0f;
        cbd.p2[2] = alphaScale;
        const uint32_t wms = static_cast<uint32_t>(c.clamp & 3u), wmt = static_cast<uint32_t>((c.clamp >> 2) & 3u);
        cbd.region[0] = static_cast<float>((c.clamp >> 4) & 0x3FFu);
        cbd.region[1] = static_cast<float>((c.clamp >> 14) & 0x3FFu);
        cbd.region[2] = static_cast<float>((c.clamp >> 24) & 0x3FFu);
        cbd.region[3] = static_cast<float>((c.clamp >> 34) & 0x3FFu);
        cbd.p3[0] = static_cast<float>(wms);
        cbd.p3[1] = static_cast<float>(wmt);
        cbd.p3[2] = origin[0];
        cbd.p3[3] = origin[1];

        // ---- vertex upload
        const uint32_t bytes = static_cast<uint32_t>(pending.size() * sizeof(Vtx));
        UINT firstVertex = 0;
        {
        if (bytes > vbCapacity)
        {
            vb.Reset();
            ++pipeGen; // new buffer must be bound
            vbWriteOffset = 0;
            D3D11_BUFFER_DESC bd{};
            bd.ByteWidth = std::max<uint32_t>(bytes, 8u << 20);
            bd.Usage = D3D11_USAGE_DYNAMIC;
            bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (FAILED(dev->CreateBuffer(&bd, nullptr, &vb)))
            {
                pending.clear();
                return;
            }
            vbCapacity = bd.ByteWidth;
        }
        D3D11_MAPPED_SUBRESOURCE m{};
        D3D11_MAP mapType = D3D11_MAP_WRITE_NO_OVERWRITE;
        if (vbWriteOffset + bytes > vbCapacity)
        {
            mapType = D3D11_MAP_WRITE_DISCARD;
            vbWriteOffset = 0;
        }
        if (FAILED(ctx->Map(vb.Get(), 0, mapType, 0, &m)))
        {
            pending.clear();
            return;
        }
        std::memcpy(static_cast<uint8_t *>(m.pData) + vbWriteOffset, pending.data(), bytes);
        ctx->Unmap(vb.Get(), 0);
        firstVertex = vbWriteOffset / static_cast<UINT>(sizeof(Vtx));
        vbWriteOffset += bytes;
        }

        // ---- pipeline
        const uint64_t hostEpoch = ps2host::d3dStateEpoch();
        if (bound.gen != pipeGen || bound.hostEpoch != hostEpoch)
        {
            bound = BoundDrawState{};
            bound.gen = pipeGen;
            bound.hostEpoch = hostEpoch;
            bound.vp.Width = -1.0f; // force
            bound.sc.right = -1;
            ctx->RSSetState(raster.Get());
            ctx->IASetInputLayout(layout.Get());
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            UINT stride = sizeof(Vtx), offset = 0;
            ID3D11Buffer *vbp = vb.Get();
            ctx->IASetVertexBuffers(0, 1, &vbp, &stride, &offset);
            ctx->VSSetShader(vs.Get(), nullptr, 0);
            ctx->PSSetShader(ps.Get(), nullptr, 0);
            ctx->GSSetShader(nullptr, nullptr, 0);
            ID3D11Buffer *cbp = cb.Get();
            ctx->VSSetConstantBuffers(0, 1, &cbp);
            ctx->PSSetConstantBuffers(0, 1, &cbp);
            ID3D11ShaderResourceView *nullSrv0 = nullptr;
            ctx->PSSetShaderResources(0, 1, &nullSrv0);
            ctx->OMSetRenderTargets(0, nullptr, nullptr);
        }
        ID3D11RenderTargetView *rtv = rt->rtv.Get();
        ID3D11DepthStencilView *dsv = dt ? dt->dsv.Get() : nullptr;
        if (rtv != bound.rtv || dsv != bound.dsv)
        {
            ctx->OMSetRenderTargets(1, &rtv, dsv);
            bound.rtv = rtv;
            bound.dsv = dsv;
        }
        D3D11_VIEWPORT vp{0, 0, static_cast<float>(rt->w * scale), static_cast<float>(rt->h * scale), 0, 1};
        if (std::memcmp(&vp, &bound.vp, sizeof(vp)) != 0)
        {
            ctx->RSSetViewports(1, &vp);
            bound.vp = vp;
        }
        D3D11_RECT sc{static_cast<LONG>(c.scissor.x0 * scale), static_cast<LONG>(c.scissor.y0 * scale),
                      static_cast<LONG>((c.scissor.x1 + 1u) * scale), static_cast<LONG>((c.scissor.y1 + 1u) * scale)};
        if (std::memcmp(&sc, &bound.sc, sizeof(sc)) != 0)
        {
            ctx->RSSetScissorRects(1, &sc);
            bound.sc = sc;
        }
        if (texSrv != bound.srv)
        {
            ctx->PSSetShaderResources(0, 1, &texSrv);
            bound.srv = texSrv;
        }
        const bool linear = st.linearFilter;
        const bool repeat = (wms == 0 && wmt == 0);
        ID3D11SamplerState *smp = samplers[(linear ? 2 : 0) + (repeat ? 1 : 0)].Get();
        if (smp != bound.smp)
        {
            ctx->PSSetSamplers(0, 1, &smp);
            bound.smp = smp;
        }

        auto drawPass = [&](float afailSelect, bool depthWrite, bool rgbOnly, bool colorOff)
        {
            cbd.p1[3] = afailSelect;
            static_assert(sizeof(cbd) <= sizeof(bound.cbData), "constant buffer cache too small");
            if (!bound.cbValid || std::memcmp(bound.cbData, &cbd, sizeof(cbd)) != 0)
            {
                D3D11_MAPPED_SUBRESOURCE cm{};
                if (SUCCEEDED(ctx->Map(cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &cm)))
                {
                    std::memcpy(cm.pData, &cbd, sizeof(cbd));
                    ctx->Unmap(cb.Get(), 0);
                    std::memcpy(bound.cbData, &cbd, sizeof(cbd));
                    bound.cbValid = true;
                }
            }
            float bf[4];
            ID3D11BlendState *bs = blendState(st, rgbOnly, colorOff, bf);
            if (bs != bound.bs || std::memcmp(bf, bound.bf, sizeof(bf)) != 0)
            {
                ctx->OMSetBlendState(bs, bf, 0xFFFFFFFFu);
                bound.bs = bs;
                std::memcpy(bound.bf, bf, sizeof(bf));
            }
            ID3D11DepthStencilState *ds = dt ? depthState(true, zfunc, depthWrite) : depthOff.Get();
            if (ds != bound.ds)
            {
                ctx->OMSetDepthStencilState(ds, 0);
                bound.ds = ds;
            }
            ctx->Draw(static_cast<UINT>(pending.size()), firstVertex);
        };

        static const int s_traceLimit = std::getenv("PS2X_GS_TRACE") ? std::atoi(std::getenv("PS2X_GS_TRACE")) : 0;
        static int s_traced = 0;
        static const uint64_t s_traceFrame = std::getenv("PS2X_GS_TRACE_FRAME") ? std::strtoull(std::getenv("PS2X_GS_TRACE_FRAME"), nullptr, 10) : 0;
        // PS2X_GS_TRACE_SEC: start tracing this many seconds after the first draw (frame counts vary per run).
        static const double s_traceSec = std::getenv("PS2X_GS_TRACE_SEC") ? std::atof(std::getenv("PS2X_GS_TRACE_SEC")) : 0.0;
        static const auto s_firstDraw = std::chrono::steady_clock::now();
        // PS2X_GS_TRACE_BANK1=<sec>: start that many seconds after the arena sound bank loaded (match start),
        // which does not drift with menu/load timing the way wall-clock windows do.
        static const double s_traceBank1 = std::getenv("PS2X_GS_TRACE_BANK1") ? std::atof(std::getenv("PS2X_GS_TRACE_BANK1")) : -1.0;
        extern std::atomic<double> g_ps2xBank1LoadedAt;
        extern std::atomic<double> g_ps2xTraceAnchorAt;
        // PS2X_GS_TRACE_ANCHOR=<sec>: count from the first call of PS2X_TRACE_ANCHOR_PC (see ps2_runtime.cpp).
        static const double s_traceAnchor = std::getenv("PS2X_GS_TRACE_ANCHOR") ? std::atof(std::getenv("PS2X_GS_TRACE_ANCHOR")) : -1.0;
        const double bank1At = g_ps2xBank1LoadedAt.load();
        const double anchorAt = g_ps2xTraceAnchorAt.load();
        const bool traceTimeOk = s_traceAnchor >= 0.0 ? (anchorAt >= 0.0 && ps2host::timeSeconds() - anchorAt >= s_traceAnchor)
                                 : s_traceBank1 >= 0.0
                                     ? (bank1At >= 0.0 && ps2host::timeSeconds() - bank1At >= s_traceBank1)
                                     : (s_traceSec <= 0.0 ||
                                        std::chrono::duration<double>(std::chrono::steady_clock::now() - s_firstDraw).count() >= s_traceSec);
        // PS2X_GS_TRACE_TBP=<tbp>: only draws sampling that texture address count (e.g. a locked env map).
        static const long s_traceTbp = std::getenv("PS2X_GS_TRACE_TBP") ? std::atol(std::getenv("PS2X_GS_TRACE_TBP")) : -1;
        const bool traceTexOk = (s_traceTbp < 0 || (pendingState.prim.tme && static_cast<long>(pendingState.context.tex0.tbp0) == s_traceTbp));
        if (s_traced < s_traceLimit && frameCounter >= s_traceFrame && traceTimeOk && traceTexOk && !pending.empty())
        {
            if (s_traced == 0)
                std::printf("[gs-trace] start at frame %llu\n", static_cast<unsigned long long>(frameCounter));
            ++s_traced;
            const Vtx &v0 = pending[0];
            float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
            for (const Vtx &v : pending)
            {
                bx0 = std::min(bx0, v.x);
                by0 = std::min(by0, v.y);
                bx1 = std::max(bx1, v.x);
                by1 = std::max(by1, v.y);
            }
            float cmin = 1e9f, cmax = -1e9f;
            for (const Vtx &v : pending)
            {
                cmin = std::min({cmin, v.r, v.g, v.b});
                cmax = std::max({cmax, v.r, v.g, v.b});
            }
            std::printf("[gs-bbox] %.0f,%.0f-%.0f,%.0f col=%.0f..%.0f ", bx0, by0, bx1, by1, cmin, cmax);
            // PS2X_GS_TRACE_VERTS=1: the first vertices of each traced draw (collapsed vs merely tiny meshes).
            static const bool s_traceVerts = std::getenv("PS2X_GS_TRACE_VERTS") != nullptr;
            if (s_traceVerts)
            {
                std::printf("[gs-verts]");
                for (size_t i = 0; i < pending.size() && i < 8; ++i)
                    std::printf(" (%.2f,%.2f,%.6f)", pending[i].x, pending[i].y, pending[i].z);
                std::printf(" ");
            }
            std::printf("[gs-trace] #%llu n=%zu fbp=%u psm=%u fbmsk=%08x tme=%d tex=%u/%u psm=%u %ux%u tfx=%u tcc=%u abe=%d alpha=%llx test=%llx "
                        "sc=%u,%u-%u,%u v0=(%.1f,%.1f z=%.4f rgba=%.0f,%.0f,%.0f,%.0f st=%.3f,%.3f,%.3f) zte=%u ztst=%u zmsk=%d zbp=%u zpsm=%u "
                        "clut=%u/%u csm=%u csa=%u fst=%d\n",
                        static_cast<unsigned long long>(statDraws), pending.size(), c.frame.fbp, c.frame.psm, c.frame.fbmsk,
                        st.prim.tme ? 1 : 0, c.tex0.tbp0, c.tex0.tbw, c.tex0.psm, st.textureWidth, st.textureHeight,
                        c.tex0.tfx, c.tex0.tcc, st.prim.abe ? 1 : 0, static_cast<unsigned long long>(c.alpha),
                        static_cast<unsigned long long>(c.test), c.scissor.x0, c.scissor.y0, c.scissor.x1, c.scissor.y1,
                        v0.x, v0.y, v0.z, v0.r, v0.g, v0.b, v0.a, v0.s, v0.t, v0.q, ztest, ztst, c.zbuf.zmask ? 1 : 0, static_cast<unsigned>(c.zbuf.zbp), static_cast<unsigned>(c.zbuf.psm),
                        static_cast<unsigned>(c.tex0.cbp), static_cast<unsigned>(c.tex0.cpsm), static_cast<unsigned>(c.tex0.csm),
                        static_cast<unsigned>(c.tex0.csa), st.prim.fst ? 1 : 0);
        }

        // Debug: PS2X_SKIP_CYCLE="tbp:psm,tbp:psm,..." hides one textured draw class at a time (4 s each,
        // starting once the frame counter reaches PS2X_SKIP_CYCLE_FRAME), to find which draws make up a surface.
        {
            static const char *s_cycle = std::getenv("PS2X_SKIP_CYCLE");
            if (s_cycle && st.prim.tme)
            {
                static std::vector<std::pair<uint32_t, uint32_t>> s_list = []()
                {
                    std::vector<std::pair<uint32_t, uint32_t>> l;
                    const char *p = std::getenv("PS2X_SKIP_CYCLE");
                    while (p && *p)
                    {
                        unsigned a = 0, b = 0;
                        if (std::sscanf(p, "%u:%u", &a, &b) == 2)
                            l.emplace_back(a, b);
                        p = std::strchr(p, ',');
                        if (p)
                            ++p;
                    }
                    return l;
                }();
                static const uint64_t s_startFrame = std::getenv("PS2X_SKIP_CYCLE_FRAME") ? std::strtoull(std::getenv("PS2X_SKIP_CYCLE_FRAME"), nullptr, 10) : 0;
                static auto s_t0 = std::chrono::steady_clock::time_point{};
                static int s_lastIdx = -2;
                static const double s_startSec = std::getenv("PS2X_SKIP_CYCLE_SEC") ? std::atof(std::getenv("PS2X_SKIP_CYCLE_SEC")) : 0.0;
                static const auto s_cycleBoot = std::chrono::steady_clock::now();
                const bool secOk = std::chrono::duration<double>(std::chrono::steady_clock::now() - s_cycleBoot).count() >= s_startSec;
                if (frameCounter >= s_startFrame && secOk && !s_list.empty())
                {
                    const auto now = std::chrono::steady_clock::now();
                    if (s_t0 == std::chrono::steady_clock::time_point{})
                        s_t0 = now;
                    static const int s_slotSec = std::getenv("PS2X_SKIP_CYCLE_SLOT") ? std::max(1, std::atoi(std::getenv("PS2X_SKIP_CYCLE_SLOT"))) : 4;
                    const int slot = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(now - s_t0).count() / s_slotSec);
                    const int idx = slot % static_cast<int>(s_list.size() + 1) - 1; // -1 = nothing skipped
                    if (idx != s_lastIdx)
                    {
                        s_lastIdx = idx;
                        std::printf("[skip-cycle] t=%lld idx=%d tbp=%u psm=%u\n",
                                    (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count(), idx,
                                    idx >= 0 ? s_list[idx].first : 0u, idx >= 0 ? s_list[idx].second : 0u);
                        std::fflush(stdout);
                    }
                    if (idx >= 0 && c.tex0.tbp0 == s_list[idx].first && c.tex0.psm == s_list[idx].second)
                    {
                        pending.clear();
                        return;
                    }
                }
            }
        }

        const bool zwrite = dt && !c.zbuf.zmask;
        if (ate && atst != 1u && afail != 0u)
        {
            drawPass(0.0f, zwrite, false, false);
            if (afail == 1u)      // FB_ONLY: failing pixels still update color
                drawPass(1.0f, false, false, false);
            else if (afail == 2u) // ZB_ONLY: failing pixels update depth only
            {
                if (zwrite)
                    drawPass(1.0f, true, false, true);
            }
            else                  // RGB_ONLY: failing pixels update rgb, no alpha/depth
                drawPass(1.0f, false, true, false);
        }
        else
            drawPass(0.0f, zwrite, false, false);

        if (texFromRt)
        {
            ID3D11ShaderResourceView *nullSrv = nullptr;
            ctx->PSSetShaderResources(0, 1, &nullSrv);
            bound.srv = nullptr;
        }
        rt->gpuDirty = true;
        {
            float minY = 1e9f, maxY = -1e9f, minX = 1e9f, maxX = -1e9f;
            for (const Vtx &v : pending)
            {
                minY = std::min(minY, v.y);
                maxY = std::max(maxY, v.y);
                minX = std::min(minX, v.x);
                maxX = std::max(maxX, v.x);
            }
            {
                const int xL = std::max<int>(static_cast<int>(c.scissor.x0), static_cast<int>(std::floor(minX)));
                const int xR = std::min<int>({static_cast<int>(c.scissor.x1), static_cast<int>(std::ceil(maxX)), static_cast<int>(rt->w) - 1});
                if (xR >= xL)
                {
                    rt->dirtyX0 = std::min<uint32_t>(rt->dirtyX0, static_cast<uint32_t>(std::max(0, xL)));
                    rt->dirtyX1 = std::max<uint32_t>(rt->dirtyX1, static_cast<uint32_t>(xR));
                }
            }
            // Rows whose pixel centres lie in [minY, maxY): a full 256-line sprite ends at y=256 but
            // only touches rows 0..255 (counting row 256 forced readbacks for CLUTs stored below).
            const int yTop = static_cast<int>(std::floor(minY));
            const int yBottom = std::max(yTop, static_cast<int>(std::ceil(maxY)) - 1);
            const int y0 = std::max<int>(static_cast<int>(c.scissor.y0), yTop);
            const int y1 = std::min<int>({static_cast<int>(c.scissor.y1), yBottom, static_cast<int>(rt->h) - 1});
            if (y1 >= y0)
            {
                rt->dirtyY0 = std::min<uint32_t>(rt->dirtyY0, static_cast<uint32_t>(std::max(0, y0)));
                rt->dirtyY1 = std::max<uint32_t>(rt->dirtyY1, static_cast<uint32_t>(y1));
            }
        }
        ++statDraws;
        if (st.prim.tme)
        {
            const uint32_t k = (static_cast<uint32_t>(c.tex0.psm) << 16) | (static_cast<uint32_t>(c.tex0.cpsm) << 8) | c.tex0.csm;
            ++fmtStats[k];
        }
        pending.clear();
    }

    // ---------------------------------------------------------------- presentation
    void latchDisplay(ID3D11Texture2D *src, uint32_t sx, uint32_t sy, uint32_t w, uint32_t h, bool scaled)
    {
        ++pipeGen;
        const uint32_t s = scaled ? scale : 1u;
        const uint32_t dw = w * s, dh = h * s;
        D3D11_TEXTURE2D_DESC td{};
        if (displayTex)
            displayTex->GetDesc(&td);
        if (!displayTex || td.Width != dw || td.Height != dh)
        {
            displayTex.Reset();
            displaySrv.Reset();
            if (!createTexture(dw, dh, D3D11_BIND_SHADER_RESOURCE, displayTex))
                return;
            dev->CreateShaderResourceView(displayTex.Get(), nullptr, &displaySrv);
        }
        D3D11_TEXTURE2D_DESC sd{};
        src->GetDesc(&sd);
        D3D11_BOX box{sx * s, sy * s, 0, std::min(sd.Width, (sx + w) * s), std::min(sd.Height, (sy + h) * s), 1};
        ctx->CopySubresourceRegion(displayTex.Get(), 0, 0, 0, 0, src, 0, &box);
        displayW = dw;
        displayH = dh;
        hasDisplay = true;
    }

    static bool validCircuit(uint64_t fb, uint64_t disp)
    {
        return ((fb >> 9) & 0x3Fu) != 0u && ((disp >> 32) & 0xFFFu) != 0u && ((disp >> 44) & 0x7FFu) != 0u;
    }
    static bool displayEnabled(const GSPresentationRequest &req)
    {
        return ((req.pmode & 1u) != 0u && validCircuit(req.dispfb1, req.display1)) ||
               ((req.pmode & 2u) != 0u && validCircuit(req.dispfb2, req.display2));
    }

    bool presentRequest(const GSPresentationRequest &req)
    {
        ++pipeGen;
        auto valid = validCircuit;
        const bool en1 = (req.pmode & 1u) != 0u && valid(req.dispfb1, req.display1);
        const bool en2 = (req.pmode & 2u) != 0u && valid(req.dispfb2, req.display2);
        if (!en1 && !en2)
            return false;
        const uint64_t dispfb = en1 ? req.dispfb1 : req.dispfb2;
        const uint64_t display = en1 ? req.display1 : req.display2;
        const uint32_t fbp = static_cast<uint32_t>(dispfb & 0x1FFu);
        const uint32_t fbw = static_cast<uint32_t>((dispfb >> 9) & 0x3Fu);
        const uint32_t dbx = static_cast<uint32_t>((dispfb >> 32) & 0x7FFu);
        const uint32_t dby = static_cast<uint32_t>((dispfb >> 43) & 0x7FFu);
        const uint32_t magh = static_cast<uint32_t>((display >> 23) & 0xFu);
        const uint32_t magv = static_cast<uint32_t>((display >> 27) & 0x3u);
        const uint32_t dw = static_cast<uint32_t>((display >> 32) & 0xFFFu);
        const uint32_t dh = static_cast<uint32_t>((display >> 44) & 0x7FFu);
        uint32_t w = (dw + 1u) / (magh + 1u);
        uint32_t h = (dh + 1u) / (magv + 1u);
        const bool interlaced = (req.smode2 & 1u) != 0u;
        const bool fieldMode = (req.smode2 & 2u) != 0u;
        if (interlaced && fieldMode)
            h /= 2u;
        if (w == 0u || h == 0u || fbw == 0u)
            return false;

        RenderTarget *rt = findRt(framePageBaseToBlock(fbp));
        std::snprintf(lastPresent, sizeof(lastPresent), "circuit=%d fbp=%u fbw=%u dbx=%u dby=%u w=%u h=%u rt=%s",
                      en1 ? 1 : 2, fbp, fbw, dbx, dby, w, h, rt ? "yes" : "no");
        if (!rt || rt->fbw != fbw)
            return false;
        if (rt->needsUpload)
            uploadRtFromVram(*rt);
        w = std::min(w, rt->w - std::min(rt->w, dbx));
        h = std::min(h, rt->h - std::min(rt->h, dby));
        if (w == 0u || h == 0u)
            return false;
        latchDisplay(rt->tex.Get(), dbx, dby, w, h, true);
        return true;
    }

};

// =====================================================================================

GSD3D11Backend::GSD3D11Backend() : m_impl(std::make_unique<Impl>())
{
}

GSD3D11Backend::~GSD3D11Backend()
{
    Impl *expected = m_impl.get();
    g_activeImpl.compare_exchange_strong(expected, nullptr);
}

void GSD3D11Backend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    std::lock_guard lock(ps2host::d3dLock());
    m_impl->vram = vram;
    m_impl->vramSize = vramSize;
    m_impl->mem.Initialize(vram, vramSize);
    if (const char *s = std::getenv("PS2X_GS_SCALE"))
        m_impl->scale = std::clamp(std::atoi(s), 1, 6);
    m_impl->initGpu();
    g_activeImpl.store(m_impl.get());
}

void GSD3D11Backend::Reset()
{
    std::lock_guard lock(ps2host::d3dLock());
    m_impl->pending.clear();
    m_impl->hasPending = false;
    m_impl->rts.clear();
    m_impl->depths.clear();
    m_impl->texCache.clear();
    m_impl->hasDisplay = false;
    m_impl->mem.Reset();
}

void GSD3D11Backend::SubmitNative(const GSDrawState &state, const void *job, uint32_t jobSize)
{
    // HCTP: the slot for native model draws (VU1 program replacements, notes\VU1_PROGRAMS.md); not used yet.
    (void)state;
    (void)job;
    (void)jobSize;
}

void GSD3D11Backend::Submit(const GSPrimitiveBatch &batch)
{
    // Called per primitive batch (~500k/s): no timing scopes here, they cost more than the work.
    std::lock_guard<std::recursive_mutex> lock(ps2host::d3dLock());
    if (!m_impl->initGpu() || batch.vertexCount == 0u)
        return;
    m_impl->submit(batch);
}

void GSD3D11Backend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    std::lock_guard lock(ps2host::d3dLock());
    m_impl->flushPending();
    m_impl->loadClut(tex0, texclut);
    m_impl->mem.LoadClut(tex0, texclut);
}

void GSD3D11Backend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard lock(ps2host::d3dLock());
    Impl &d = *m_impl;
    d.flushPending();
    const auto &bb = command.bitbltbuf;
    const auto &tp = command.trxpos;
    const auto &tr = command.trxreg;
    uint32_t sf = 0, sl = 0, df = 0, dl = 0;
    pageSpanRect(bb.dbp, bb.dbw, bb.dpsm, tp.dsax, tp.dsay, tr.rrw, tr.rrh, df, dl);
    pageSpanRect(bb.sbp, bb.sbw, bb.spsm, tp.ssax, tp.ssay, tr.rrw, tr.rrh, sf, sl);
    {
        static int s_xferTrace = std::getenv("PS2X_XFER_TRACE") ? std::atoi(std::getenv("PS2X_XFER_TRACE")) : 0;
        if (s_xferTrace > 0)
        {
            --s_xferTrace;
            std::printf("[xfer] path=%u dir=%u dbp=%u dbw=%u dpsm=%u sbp=%u sbw=%u spsm=%u rect=%ux%u src=(%u,%u) dst=(%u,%u)\n",
                        static_cast<unsigned>(g_gifCurrentPath), command.direction, bb.dbp, bb.dbw, bb.dpsm, bb.sbp, bb.sbw, bb.spsm, tr.rrw, tr.rrh,
                        tp.ssax, tp.ssay, tp.dsax, tp.dsay);
        }
    }
    {
        // Diagnostics: an image upload that is replaced before all its pixels arrived means data was lost.
        static const bool s_checkXfer = std::getenv("PS2X_XFER_CHECK") != nullptr;
        if (s_checkXfer)
        {
            const GSTransferSnapshot prev = d.mem.GetTransferSnapshot();
            if (prev.direction == 0u && prev.totalPixels != 0u && prev.copiedPixels < prev.totalPixels)
                std::printf("[xfer-check] incomplete upload: %u/%u pixels before new transfer (dbp=%u)\n",
                            prev.copiedPixels, prev.totalPixels, bb.dbp);
        }
    }
    d.lastUploadDbp = bb.dbp;
    d.syncReason = command.direction == 0u ? "upload" : command.direction == 1u ? "readback" : "copy";
    if (d.syncLogEnabled)
        std::snprintf(d.syncDetail, sizeof(d.syncDetail), "xfer dbp=%u dbw=%u dpsm=%u dst=(%u,%u) %ux%u", bb.dbp, bb.dbw, bb.dpsm,
                      tp.dsax, tp.dsay, tr.rrw, tr.rrh);
    d.uploadFirst = 1u;
    d.uploadLast = 0u;
    if (command.direction == 0u)
    {
        // An upload that covers every row the GPU wrote in a live render target replaces that content
        // (RenderWare's texture cache re-uploads an unlocked env-map raster every frame): skip the GPU
        // readback that syncOverlapping would do just to have the upload overwrite it.
        // The upload usually arrives as row chunks, so trim the target's GPU-dirty row range by the rows this
        // chunk rewrites; rows the GPU still owns stay dirty and keep their readback.
        for (auto &rt : d.rts)
        {
            if (!rt.gpuDirty || rt.dirtyY0 == 0xFFFFFFFFu || bb.dbp != rt.fbpBlock || bb.dbw != rt.fbw ||
                bitsPerPixel(bb.dpsm) != bitsPerPixel(rt.psm) || tp.dsax != 0u || tr.rrw < rt.w || tr.rrh == 0u)
                continue;
            const uint32_t y0 = tp.dsay, y1 = tp.dsay + tr.rrh - 1u; // rows this upload replaces (inclusive)
            if (y0 > rt.dirtyY1 || y1 < rt.dirtyY0)
                continue; // no overlap with the dirty rows: syncOverlapping skips it anyway
            if (y0 <= rt.dirtyY0 && y1 >= rt.dirtyY1)
            {
                rt.gpuDirty = false;
                rt.dirtyY0 = 0xFFFFFFFFu;
                rt.dirtyY1 = 0u;
                rt.dirtyX0 = 0xFFFFFFFFu;
                rt.dirtyX1 = 0u;
            }
            else if (y0 <= rt.dirtyY0)
                rt.dirtyY0 = y1 + 1u; // chunk at the top of the dirty range
            else if (y1 >= rt.dirtyY1)
                rt.dirtyY1 = y0 - 1u; // chunk at the bottom
            else
                continue;              // chunk in the middle: keep the readback
            rt.skipNextSync = true;
            ++d.statSkippedSyncs;
        }
        d.syncOverlapping(df, dl);
        d.recordWriter('U', bb.dbp, bb.dbw, bb.dpsm, tp.dsax, tp.dsay, tr.rrw, tr.rrh);
        d.invalidateOverlapping(df, dl);
        d.markPagesDirty(df, dl);
        d.uploadFirst = df;
        d.uploadLast = dl;
        if (d.staleLog)
        {
            ++d.uploadSeq;
            for (uint32_t p = df; p <= dl && p < 512u; ++p)
                d.pageHist[p] = {d.uploadSeq, bb.dbp, bb.dbw, bb.dpsm, tr.rrw, tr.rrh, tp.dsay};
        }
    }
    else if (command.direction == 1u)
    {
        d.syncOverlapping(sf, sl);
    }
    else if (command.direction == 2u)
    {
        d.syncOverlapping(sf, sl);
        d.syncOverlapping(df, dl);
        d.recordWriter('C', bb.dbp, bb.dbw, bb.dpsm, tp.dsax, tp.dsay, tr.rrw, tr.rrh);
        d.invalidateOverlapping(df, dl);
        d.markPagesDirty(df, dl);
    }
    d.mem.BeginTransfer(command);
}

void GSD3D11Backend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    static const char *s_uploadDumpDir = std::getenv("PS2X_UPLOAD_DUMP"); // per packet: no getenv here
    if (const char *dumpDir = s_uploadDumpDir)
    {
        // Append every host->local payload to one file per destination pointer (diagnostics).
        static int s_files = 0;
        // PS2X_UPLOAD_DUMP_SEC=<s> starts the dump later; PS2X_UPLOAD_DUMP_MIN=<pixels> keeps only big transfers.
        static const double s_dumpSec = std::getenv("PS2X_UPLOAD_DUMP_SEC") ? std::atof(std::getenv("PS2X_UPLOAD_DUMP_SEC")) : 0.0;
        static const uint32_t s_dumpMin = std::getenv("PS2X_UPLOAD_DUMP_MIN") ? static_cast<uint32_t>(std::atol(std::getenv("PS2X_UPLOAD_DUMP_MIN"))) : 0u;
        static const auto s_dumpBoot = std::chrono::steady_clock::now();
        const GSTransferSnapshot snap = m_impl->mem.GetTransferSnapshot();
        if (snap.direction == 0u && s_files < 64 && sizeBytes >= 64u && m_impl->lastUploadDbp != 15008u &&
            snap.totalPixels >= s_dumpMin &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - s_dumpBoot).count() >= s_dumpSec)
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/upload_%03d_dbp%u.bin", dumpDir, s_files, m_impl->lastUploadDbp);
            if (snap.copiedPixels == 0u)
                ++s_files;
            if (FILE *f = std::fopen(path, "ab"))
            {
                std::fwrite(data, 1, sizeBytes, f);
                std::fclose(f);
            }
        }
    }
    std::lock_guard lock(ps2host::d3dLock());
    Impl &d = *m_impl;
    d.mem.UploadImage(data, sizeBytes);
    // Texture/CLUT decodes cached from these pages (possibly taken mid-transfer) are now stale.
    if (d.uploadFirst <= d.uploadLast)
    {
        d.invalidateOverlapping(d.uploadFirst, d.uploadLast);
        d.markPagesDirty(d.uploadFirst, d.uploadLast);
    }
}

void GSD3D11Backend::Flush()
{
    std::lock_guard lock(ps2host::d3dLock());
    m_impl->flushPending();
}

void GSD3D11Backend::TextureFlush()
{
    m_impl->mem.TextureFlush();
}

void GSD3D11Backend::Sync(GSSyncReason reason)
{
    std::lock_guard lock(ps2host::d3dLock());
    m_impl->flushPending();
    if (reason == GSSyncReason::DebugReadback)
        for (auto &rt : m_impl->rts)
            m_impl->syncRtToVram(rt);
}

PresentationFrame GSD3D11Backend::Present(const GSPresentationRequest &request)
{
    ps2_host_profile::Scope profScope(ps2_host_profile::GsPresent);
    std::lock_guard lock(ps2host::d3dLock());
    Impl &d = *m_impl;
    d.flushPending();
    ++d.frameCounter;
    {
        // Widescreen: this frame's aspect is the one its 2D was drawn for; the next frame decides from this one.
        static const bool s_hud = Impl::hudWidescreen();
        d.displayWide = !s_hud || d.hudOn;
        d.hudOn = s_hud && d.frame3D;
        d.frame3D = false;
    }
    PresentationFrame result{};
    if (d.gpuReady && d.presentRequest(request))
    {
        result.pixels.assign(4u, 0u);
        result.width = 1u;
        result.height = 1u;
    }
    else if (!Impl::displayEnabled(request))
    {
        // Display not set up yet (e.g. DISPFB 0 around a scene change): nothing to show, and reading
        // every render target back for the CPU path took long enough to stall the frame.
    }
    else
    {
        // No GPU render target is being displayed (FMV / direct uploads): use local memory.
        d.syncReason = "cpu-present";
        for (auto &rt : d.rts)
            d.syncRtToVram(rt);
        result = d.mem.Present(request);
        if (d.gpuReady && result)
        {
            // CPU frames use a 640-wide host stride.
            ComPtr<ID3D11Texture2D> tmp;
            if (d.createTexture(result.width, result.height, D3D11_BIND_SHADER_RESOURCE, tmp))
            {
                d.ctx->UpdateSubresource(tmp.Get(), 0, nullptr, result.pixels.data(), 640u * 4u, 0);
                d.latchDisplay(tmp.Get(), 0, 0, result.width, result.height, false);
            }
        }
    }

    // Evict idle GPU resources.
    if ((d.frameCounter & 255u) == 0u)
    {
        for (auto it = d.texCache.begin(); it != d.texCache.end();)
        {
            if (d.frameCounter - it->second.lastUse > 600u)
                it = d.texCache.erase(it);
            else
                ++it;
        }
        if (std::getenv("PS2X_GS_STATS"))
            std::printf("[gs-d3d11] draws=%llu syncs=%llu skippedSyncs=%llu uploads=%llu texDecodes=%llu rts=%zu tex=%zu\n",
                        static_cast<unsigned long long>(d.statDraws), static_cast<unsigned long long>(d.statSyncs),
                        static_cast<unsigned long long>(d.statSkippedSyncs), static_cast<unsigned long long>(d.statUploads),
                        static_cast<unsigned long long>(d.statTexDecodes), d.rts.size(), d.texCache.size());
        if (std::getenv("PS2X_GS_STATS"))
        {
            std::printf("[gs-d3d11]   present: %s\n", d.lastPresent);
            for (const auto &rt : d.rts)
                std::printf("[gs-d3d11]   rt fbp=%u(page %u) fbw=%u psm=%u %ux%u dirty=%d\n", rt.fbpBlock, rt.fbpBlock / 32u,
                            rt.fbw, rt.psm, rt.w, rt.h, rt.gpuDirty ? 1 : 0);
        }
        if (std::getenv("PS2X_GS_STATS"))
            for (const auto &[k, n] : d.fmtStats)
                std::printf("[gs-d3d11]   tex psm=%u cpsm=%u csm=%u draws=%llu\n", k >> 16, (k >> 8) & 0xFF, k & 0xFF,
                            static_cast<unsigned long long>(n));
        if (d.statsEnabled && !d.decodeHist.empty())
        {
            std::vector<std::pair<uint32_t, uint64_t>> top;
            for (const auto &[k, n] : d.decodeHist)
                top.emplace_back(n, k);
            std::sort(top.rbegin(), top.rend());
            for (size_t i = 0; i < top.size() && i < 6u; ++i)
                std::printf("[gs-d3d11]   decodes tbp=%llu psm=%llu %llux%llu: %u\n", (unsigned long long)(top[i].second >> 40),
                            (unsigned long long)((top[i].second >> 32) & 0xFF), (unsigned long long)((top[i].second >> 16) & 0xFFFF),
                            (unsigned long long)(top[i].second & 0xFFFF), top[i].first);
            std::printf("[gs-d3d11]   decode keys: %zu\n", d.decodeHist.size());
            d.decodeHist.clear();
        }
        if (std::getenv("PS2X_GS_STATS"))
            for (const auto &[r, n] : d.syncReasons)
                std::printf("[gs-d3d11]   sync %s: %llu\n", r.c_str(), static_cast<unsigned long long>(n));
    }
    return result;
}

bool GSD3D11Backend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    std::lock_guard lock(ps2host::d3dLock());
    Impl &d = *m_impl;
    d.flushPending();
    Impl::RenderTarget *rt = d.findRt(framePageBaseToBlock(context.frame.fbp));
    if (rt && d.gpuReady)
    {
        const float a = std::min(1.0f, static_cast<float>((rgba >> 24) & 0xFFu) / 128.0f);
        const float col[4] = {(rgba & 0xFFu) / 255.0f, ((rgba >> 8) & 0xFFu) / 255.0f, ((rgba >> 16) & 0xFFu) / 255.0f, a};
        d.ctx->ClearRenderTargetView(rt->rtv.Get(), col);
        rt->gpuDirty = true;
        rt->dirtyY0 = 0u;
        rt->dirtyY1 = rt->h ? rt->h - 1u : 0u;
        rt->dirtyX0 = 0u;
        rt->dirtyX1 = rt->w ? rt->w - 1u : 0u;
        return true;
    }
    return d.mem.ClearFramebuffer(context, rgba);
}

uint32_t GSD3D11Backend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    return m_impl->mem.ConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GSD3D11Backend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    return m_impl->mem.ReadVram(psm, base, bw, x, y);
}

void GSD3D11Backend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    m_impl->mem.WriteVram(psm, base, bw, x, y, value);
}

void GSD3D11Backend::SnapshotVram(std::vector<uint8_t> &out) const
{
    m_impl->mem.SnapshotVram(out);
}

GSTransferSnapshot GSD3D11Backend::GetTransferSnapshot() const
{
    return m_impl->mem.GetTransferSnapshot();
}

namespace
{
    std::atomic<uint64_t> g_latchedFrames{0};
}

void GSD3D11Backend::PresentToMailbox(const GSPresentationRequest &request, GSPresentTicket ticket, GSPresentMailbox &box)
{
    (void)ticket;
    (void)box;
    Present(request);
    g_latchedFrames.fetch_add(1, std::memory_order_release);
}

void GSD3D11Backend::ResetPresentation(GSPresentMailbox &box, uint64_t seq)
{
    (void)box;
    (void)seq;
    std::lock_guard lock(ps2host::d3dLock());
    m_impl->hasDisplay = false;
}

uint64_t GSD3D11Backend::LatchedFrameCount() { return g_latchedFrames.load(std::memory_order_acquire); }

bool GSD3D11Backend::TakeFreshFrame()
{
    static uint64_t s_taken = 0;
    const uint64_t now = g_latchedFrames.load(std::memory_order_acquire);
    if (now == s_taken)
        return false;
    s_taken = now;
    return true;
}

bool GSD3D11Backend::IsActive()
{
    return g_activeImpl.load() != nullptr;
}

bool GSD3D11Backend::PresentLatchedToHost()
{
    Impl *d = g_activeImpl.load();
    if (!d)
        return false;
    ps2host::beginFrame();
    ps2_host_profile::Scope profScope(ps2_host_profile::GsPresent);
    std::lock_guard lock(ps2host::d3dLock());
    if (!d->hasDisplay || !d->displaySrv)
        return false;
    ps2host::setPresentWide(d->displayWide);
    ps2host::presentTexture(d->displaySrv.Get(), d->displayW, d->displayH, 1.0f, 1.0f);
    return true;
}
