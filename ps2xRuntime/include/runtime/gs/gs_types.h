#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

enum GSPrimType : uint8_t
{
    GS_PRIM_POINT = 0,
    GS_PRIM_LINE = 1,
    GS_PRIM_LINESTRIP = 2,
    GS_PRIM_TRIANGLE = 3,
    GS_PRIM_TRISTRIP = 4,
    GS_PRIM_TRIFAN = 5,
    GS_PRIM_SPRITE = 6,
};

enum GSPsm : uint8_t
{
    GS_PSM_CT32 = 0,
    GS_PSM_CT24 = 1,
    GS_PSM_CT16 = 2,
    GS_PSM_CT16S = 10,
    GS_PSM_T8 = 19,
    GS_PSM_T4 = 20,
    GS_PSM_T8H = 27,
    GS_PSM_T4HL = 36,
    GS_PSM_T4HH = 44,
    GS_PSM_Z32 = 48,
    GS_PSM_Z24 = 49,
    GS_PSM_Z16 = 50,
    GS_PSM_Z16S = 58,
};

enum GSGifFormat : uint8_t
{
    GIF_FMT_PACKED = 0,
    GIF_FMT_REGLIST = 1,
    GIF_FMT_IMAGE = 2,
    GIF_FMT_DISABLED = 3,
};

enum GSRegId : uint8_t
{
    GS_REG_PRIM = 0x00,
    GS_REG_RGBAQ = 0x01,
    GS_REG_ST = 0x02,
    GS_REG_UV = 0x03,
    GS_REG_XYZF2 = 0x04,
    GS_REG_XYZ2 = 0x05,
    GS_REG_TEX0_1 = 0x06,
    GS_REG_TEX0_2 = 0x07,
    GS_REG_CLAMP_1 = 0x08,
    GS_REG_CLAMP_2 = 0x09,
    GS_REG_FOG = 0x0A,
    GS_REG_XYZF3 = 0x0C,
    GS_REG_XYZ3 = 0x0D,
    GS_REG_AD = 0x0F,
    GS_REG_TEX1_1 = 0x14,
    GS_REG_TEX1_2 = 0x15,
    GS_REG_TEX2_1 = 0x16,
    GS_REG_TEX2_2 = 0x17,
    GS_REG_XYOFFSET_1 = 0x18,
    GS_REG_XYOFFSET_2 = 0x19,
    GS_REG_PRMODECONT = 0x1A,
    GS_REG_PRMODE = 0x1B,
    GS_REG_TEXCLUT = 0x1C,
    GS_REG_SCANMSK = 0x22,
    GS_REG_MIPTBP1_1 = 0x34,
    GS_REG_MIPTBP1_2 = 0x35,
    GS_REG_MIPTBP2_1 = 0x36,
    GS_REG_MIPTBP2_2 = 0x37,
    GS_REG_TEXA = 0x3B,
    GS_REG_FOGCOL = 0x3D,
    GS_REG_TEXFLUSH = 0x3F,
    GS_REG_SCISSOR_1 = 0x40,
    GS_REG_SCISSOR_2 = 0x41,
    GS_REG_ALPHA_1 = 0x42,
    GS_REG_ALPHA_2 = 0x43,
    GS_REG_DIMX = 0x44,
    GS_REG_DTHE = 0x45,
    GS_REG_COLCLAMP = 0x46,
    GS_REG_TEST_1 = 0x47,
    GS_REG_TEST_2 = 0x48,
    GS_REG_PABE = 0x49,
    GS_REG_FBA_1 = 0x4A,
    GS_REG_FBA_2 = 0x4B,
    GS_REG_FRAME_1 = 0x4C,
    GS_REG_FRAME_2 = 0x4D,
    GS_REG_ZBUF_1 = 0x4E,
    GS_REG_ZBUF_2 = 0x4F,
    GS_REG_BITBLTBUF = 0x50,
    GS_REG_TRXPOS = 0x51,
    GS_REG_TRXREG = 0x52,
    GS_REG_TRXDIR = 0x53,
    GS_REG_HWREG = 0x54,
    GS_REG_SIGNAL = 0x60,
    GS_REG_FINISH = 0x61,
    GS_REG_LABEL = 0x62,
};

struct GSVertex
{
    float x = 0.0f;
    float y = 0.0f;
    double z = 0.0;
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 0;
    float q = 1.0f;
    float s = 0.0f;
    float t = 0.0f;
    uint16_t u = 0;
    uint16_t v = 0;
    uint8_t fog = 0;
};

struct GSFrameReg
{
    uint32_t fbp = 0;
    uint32_t fbw = 0;
    uint8_t psm = 0;
    uint32_t fbmsk = 0;
};

struct GSZbufReg
{
    uint32_t zbp = 0;
    uint8_t psm = 0;
    bool zmask = false;
};

struct GSScissorReg
{
    uint16_t x0 = 0;
    uint16_t x1 = 0;
    uint16_t y0 = 0;
    uint16_t y1 = 0;
};

struct GSTex0Reg
{
    uint32_t tbp0 = 0;
    uint8_t tbw = 0;
    uint8_t psm = 0;
    uint8_t tw = 0;
    uint8_t th = 0;
    uint8_t tcc = 0;
    uint8_t tfx = 0;
    uint32_t cbp = 0;
    uint8_t cpsm = 0;
    uint8_t csm = 0;
    uint8_t csa = 0;
    uint8_t cld = 0;
};

struct GSXYOffsetReg
{
    uint16_t ofx = 0;
    uint16_t ofy = 0;
};

struct GSTexaReg
{
    uint8_t ta0 = 0;
    bool aem = false;
    uint8_t ta1 = 0;
};

struct GSTexClutReg
{
    uint8_t cbw = 0;
    uint8_t cou = 0;
    uint16_t cov = 0;
};

struct GSContext
{
    GSFrameReg frame;
    GSScissorReg scissor;
    GSTex0Reg tex0;
    GSXYOffsetReg xyoffset;
    GSZbufReg zbuf;
    uint64_t tex1 = 0;
    uint64_t miptbp1 = 0;
    uint64_t miptbp2 = 0;
    uint64_t clamp = 0;
    uint64_t alpha = 0;
    uint64_t test = 0;
    uint64_t fba = 0;
};

struct GSPrimReg
{
    GSPrimType type = GS_PRIM_POINT;
    bool iip = false;
    bool tme = false;
    bool fge = false;
    bool abe = false;
    bool aa1 = false;
    bool fst = false;
    bool ctxt = false;
    bool fix = false;
};

struct GSBitBltBuf
{
    uint32_t sbp = 0;
    uint8_t sbw = 0;
    uint8_t spsm = 0;
    uint32_t dbp = 0;
    uint8_t dbw = 0;
    uint8_t dpsm = 0;
};

struct GSTrxPos
{
    uint16_t ssax = 0;
    uint16_t ssay = 0;
    uint16_t dsax = 0;
    uint16_t dsay = 0;
    uint8_t dir = 0;
};

struct GSTrxReg
{
    uint16_t rrw = 0;
    uint16_t rrh = 0;
};

struct GSDrawState
{
    GSContext context{};
    GSPrimReg prim{};
    GSTexaReg texa{};
    GSTexClutReg texclut{};
    bool pabe = false;
    uint64_t scanmsk = 0;
    uint64_t dimx = 0;
    uint64_t dthe = 0;
    uint64_t colclamp = 0;
    uint8_t fogR = 0;
    uint8_t fogG = 0;
    uint8_t fogB = 0;
    uint16_t textureWidth = 1;
    uint16_t textureHeight = 1;
    bool linearFilter = false;
};

struct GSPrimitiveBatch
{
    std::array<GSVertex, 3> vertices{};
    uint8_t vertexCount = 0;
    GSDrawState state{};
};

// HCTP GPU character path: one stage-2 corner as the VU1 program reads it; the D3D11 vertex shader (vs_native) does
// p08/p18's per-corner maths: RGBA = trunc(min(C1*S1*L + C1*S2, 255)) & 255, STQ = (C0.xy + K.xy, C0.z) * C0.w,
// XYZF2 = FTOI4(P) (x/y 12.4, z >> 4 24-bit, fog = P.w). K.zw = XYOFFSET in pixels (the backend fills it).
struct GSNativeVtx
{
    float p[4];  // screen position + fog (stage-1 output +320)
    float l[4];  // light (+480; .w = 1 for p08)
    float c1[4]; // corner colour
    float c0[4]; // S, T, Q source, and the STQ multiplier (N.w for p08, 1 for p18)
    float s1[4]; // colour scale ([1006], x255 / x128 for p08)
    float s2[4]; // colour add ([1007] * [1019])
    float k[4];  // ST scroll xy, XYOFFSET xy
};

// The chain worker sends a chunk as its corners once (64 B: p, l, c1, c0 of GSNativeVtx) + the chunk's constants
// (s1, s2, k) + a triangle-list index list; the renderer expands it into GSNativeVtx on its own thread.
struct GSNativeCorner
{
    float p[4];
    float l[4];
    float c1[4];
    float c0[4];
};

struct GSNativeConst
{
    float s1[4];
    float s2[4];
    float k[4];
};

// HCTP GPU stage-1 (HCTP_GPU_STAGE1): the corner carries the stage-1 INPUTS (VU1 p01-p04: position, normal, blend
// weights) and vs_native_s1 does the skinning, the light sum and the projection as well. nrm.w = the batch constant
// slot (the renderer fills it), c0.w = mode: 0 p08 (L.w = 1, STQ * q), 1 p18 (L.w = light.w, STQ * 1), 2 p19 (as p18,
// no near clip).
struct GSNativeS1Corner
{
    float pos[4];
    float nrm[4];
    float wt[4];
    float c1[4];
    float c0[4];
};

// one stage-1 batch's constants (+ the stage-2 chunk constants), float4 rows as the VU1 reads them
struct GSNativeS1Const
{
    float m[16][4];  // 4 bones (801 + 4k): rows; unused bones zero
    float pm[4][4];  // projection 1020-1023
    float l[3][4];   // light directions 1012-1014
    float c[3][4];   // light colours 1016-1018
    float misc[4];   // fog a, fog b ([800].zw), normalise (1 / 0), near offset ([1008].x)
    float s1[4];     // colour scale (GSNativeConst::s1)
    float s2[4];     // colour add
    float k[4];      // ST scroll xy, XYOFFSET xy (the renderer fills zw)
};

struct GSNativeChunk
{
    const GSNativeS1Corner *s1Corners = nullptr; // set: a GPU stage-1 chunk (corners / k unused)
    const GSNativeS1Const *s1 = nullptr;         // the group's constants (s1Count, corner nrm.w = index); nullptr = as the last chunk's (s1Serial)
    uint32_t s1Count = 0;
    uint64_t s1Serial = 0;
    uint32_t s1Cull = 0;                         // 1: cull back faces (odd strip triangles already flipped)
    const GSNativeCorner *corners = nullptr;
    uint32_t cornerCount = 0;
    const uint16_t *index = nullptr; // 3 per triangle
    uint32_t indexCount = 0;
    GSNativeConst k{};

    GSNativeVtx vertex(uint32_t i) const
    {
        GSNativeVtx o;
        std::memcpy(&o, &corners[index[i]], sizeof(GSNativeCorner));
        std::memcpy(o.s1, &k, sizeof(GSNativeConst));
        return o;
    }
};

struct GSTransferCommand
{
    GSBitBltBuf bitbltbuf{};
    GSTrxPos trxpos{};
    GSTrxReg trxreg{};
    uint32_t direction = 3;
};

struct GSTransferSnapshot
{
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t totalPixels = 0;
    uint32_t copiedPixels = 0;
    uint32_t direction = 3;
    size_t localToHostPendingBytes = 0;
};

struct GSPresentationRequest
{
    uint64_t pmode = 0;
    uint64_t smode1 = 0; // only CMOD (bits 13-14: 2 NTSC, 3 PAL) is used — the visible lines
    uint64_t smode2 = 0;
    uint64_t dispfb1 = 0;
    uint64_t display1 = 0;
    uint64_t dispfb2 = 0;
    uint64_t display2 = 0;
    uint64_t bgcolor = 0;
    uint64_t vsyncTick = 0;
    GSFrameReg contextFrames[2]{};
    GSFrameReg preferredSource{};
    uint32_t preferredDestFbp = 0;
    bool hasPreferredSource = false;
};

struct PresentationFrame
{
    std::vector<uint8_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t displayFbp = 0;
    uint32_t sourceFbp = 0;
    bool usedPreferred = false;

    explicit operator bool() const
    {
        return !pixels.empty() && width != 0u && height != 0u;
    }
};

enum class GSSyncReason : uint8_t
{
    Finish,
    LocalToHost,
    Presentation,
    DebugReadback,
    Reset,
};
