#ifndef PS2_VU1_COMPILED_H
#define PS2_VU1_COMPILED_H

// VU1 microprograms recompiled ahead of time into C++: a game's VU1 code, generated at build time (e.g.
// def-jam-vendetta-recomp's tools/vu1_recomp.py) and linked into the runner, which registers it with
// vu1RegisterCompiledPieces(). A piece is one block of VU code at a fixed VU address; VU1Interpreter runs it only while
// VU1 code memory holds exactly those bytes (checked whenever the code changes) and only in fast mode with no MAC /
// status flag reads in VU memory. The compiled code keeps the interpreter's fast-mode model exactly: the same stalls
// (the generator's per-pair stall reads / write marks are checked against the interpreter's own decode before a
// piece is used), the same Q / clip / store / XGKICK pipelines and the same cycle count; only the instruction work is
// specialised (registers, lanes, latencies and branch targets as constants, branches as jumps).
// VENDETTA_VU1_VERIFY=1 compares every program with the original model; VENDETTA_VU1_COMPILED=0 turns it off.

#include "runtime/ps2_vu1.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

enum class Vu1CompiledResult : uint8_t // (declared in ps2_vu1.h, with Vu1CompiledFn)
{
    Continue,     // left the compiled code at m_state.pc (not compiled here, or out of budget); the loop goes on there
    ProgramEnded, // ran the E-bit delay slot: the program is over
};

// What the generator assumed about one pair (from its own decode); the piece is used only if every pair matches the
// interpreter's decode of the same words.
struct Vu1PairShape
{
    uint8_t stallVf[4][2]; // VF reads that can stall: (register, lanes)
    uint8_t stallVfCount;
    uint16_t stallVi;      // VI registers 1-15 read
    uint8_t stallAcc;      // ACC lanes read
    uint8_t lowerPipeline; // VU1Interpreter's Pipeline value of the lower instruction
    uint8_t flags;         // kShape* below
    uint8_t markLowerVf[2];
    uint8_t markLowerVfLatency;
    uint8_t markUpperVf[2];
    uint8_t markUpperVfLatency;
    uint16_t markVi;
    uint8_t markViLatency;
    uint8_t markAcc;
    uint8_t lowerViWriteReg;
    uint8_t upperVfShadowReg;
    uint8_t suppressedLowerVf;
};
constexpr uint8_t kShapeWaitQ = 0x01u, kShapeWaitP = 0x02u, kShapeIBit = 0x04u, kShapeEBit = 0x08u,
                  kShapeDBit = 0x10u, kShapeTBit = 0x20u, kShapeReserved = 0x40u, kShapeDelaysBranchRead = 0x80u;

struct Vu1CompiledPiece
{
    uint32_t vuAddress;         // byte address in VU1 code memory
    uint32_t pairCount;         // 64-bit instruction pairs
    const uint32_t *code;       // the pairs as they sit in VU memory (lower word, upper word)
    const Vu1CompiledFn *fns;   // per pair: the function to enter there (nullptr = not compiled, the interpreter runs it)
    const Vu1PairShape *shapes; // per pair
    const char *name;
};

// Called by the generated code's static initialiser.
void vu1RegisterCompiledPieces(const Vu1CompiledPiece *pieces, size_t count);

// The VU float rules of the fast path, shared by the interpreter (execUpperFast) and the compiled code. Forced
// inline: the generated functions are huge and the compiler's own inlining gave up on them (a call per operand).
#if defined(_MSC_VER)
#define VU1_FORCEINLINE __forceinline
#else
#define VU1_FORCEINLINE inline __attribute__((always_inline))
#endif
namespace vu1fast
{
    // = VU1Interpreter::normalizeOperand: denormals -> signed zero, Inf / NaN -> signed max
    VU1_FORCEINLINE float operand(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t exponent = bits & 0x7F800000u;
        if (exponent == 0u)
            bits &= 0x80000000u;
        else if (exponent == 0x7F800000u)
            bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    // A sum / difference / product of two VU floats: clamping the float result gives the same value as the
    // interpreter's exact (long double) check — the result is zero / denormal / overflowing exactly when the exact
    // one is, with the same sign (round toward zero).
    VU1_FORCEINLINE float plainResult(float value)
    {
        return operand(value);
    }

    // acc ± a·b: the float result can differ from the exact one near zero, so decide as
    // normalizeFmacExactResult does, on the exact value (long double = double here).
    VU1_FORCEINLINE float exactResult(float value, double exact)
    {
        uint32_t bits;
        const double magnitude = exact < 0.0 ? -exact : exact;
        uint64_t exactBits;
        std::memcpy(&exactBits, &exact, sizeof(exactBits));
        const uint32_t sign = (exactBits >> 63u) != 0u ? 0x80000000u : 0u; // = std::signbit, without a library call
        if (magnitude == 0.0 || magnitude < static_cast<double>(std::numeric_limits<float>::min()))
            bits = sign;
        else if (magnitude > static_cast<double>(std::numeric_limits<float>::max()))
            bits = sign | 0x7F7FFFFFu;
        else
            return value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    // = VU1Interpreter::normalizeResult without the flags
    VU1_FORCEINLINE float result(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if ((bits & 0x7FFFFFFFu) != 0u && exponent == 0u)
            bits &= 0x80000000u;
        else if (exponent == 0xFFu)
            bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    VU1_FORCEINLINE float fromBits(uint32_t bits)
    {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    VU1_FORCEINLINE int32_t floatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }

    // A register's four ready cycles (a row of VU1Interpreter::m_vfReady, or m_accReady): which of the Lanes (bit c =
    // component c, x = bit 0) are not ready at cycle, and the write marks. With AVX2 a row is one 256-bit load /
    // compare / store; a partly marked row is blended and stored whole, so the next whole-row load is forwarded from a
    // single store (a load spanning several narrower stores stalls).
    template <unsigned Lanes>
    VU1_FORCEINLINE unsigned lanesAfter(const uint64_t *ready, uint64_t cycle)
    {
#if defined(__AVX2__)
        const __m256i late = _mm256_cmpgt_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ready)),
                                                _mm256_set1_epi64x(static_cast<int64_t>(cycle)));
        return static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(late))) & Lanes;
#else
        unsigned late = 0u;
        for (unsigned component = 0; component < 4u; ++component)
        {
            if ((Lanes & (1u << component)) != 0u && ready[component] > cycle)
                late |= 1u << component;
        }
        return late;
#endif
    }

    template <unsigned Lanes>
    VU1_FORCEINLINE void markReady(uint64_t *ready, uint64_t value)
    {
#if defined(__AVX2__)
        const __m256i marked = _mm256_set1_epi64x(static_cast<int64_t>(value));
        if constexpr (Lanes == 0xFu)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(ready), marked);
        else
        {
            constexpr int dwords = ((Lanes & 1u) ? 0x03 : 0) | ((Lanes & 2u) ? 0x0C : 0) |
                                   ((Lanes & 4u) ? 0x30 : 0) | ((Lanes & 8u) ? 0xC0 : 0);
            const __m256i old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ready));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(ready), _mm256_blend_epi32(old, marked, dwords));
        }
#else
        for (unsigned component = 0; component < 4u; ++component)
        {
            if ((Lanes & (1u << component)) != 0u)
                ready[component] = value;
        }
#endif
    }

    // CLIP's flags (x+ x- y+ y- z+ z-) for vs against |vt.w|, exactly as VU1Interpreter::execUpper computes them
    VU1_FORCEINLINE uint32_t clipFlags(const float *vs, const float *vt)
    {
        uint32_t wBits = 0u;
        std::memcpy(&wBits, &vt[3], sizeof(wBits));
        const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
        const auto exceeds = [limit](float value, uint32_t signMask)
        {
            uint32_t bits = 0u;
            std::memcpy(&bits, &value, sizeof(bits));
            bits ^= signMask;
            int32_t ordered = 0;
            std::memcpy(&ordered, &bits, sizeof(ordered));
            return ordered > limit;
        };
        uint32_t flags = 0u;
        if (exceeds(vs[0], 0x00000000u))
            flags |= 0x01u;
        if (exceeds(vs[0], 0x80000000u))
            flags |= 0x02u;
        if (exceeds(vs[1], 0x00000000u))
            flags |= 0x04u;
        if (exceeds(vs[1], 0x80000000u))
            flags |= 0x08u;
        if (exceeds(vs[2], 0x00000000u))
            flags |= 0x10u;
        if (exceeds(vs[2], 0x80000000u))
            flags |= 0x20u;
        return flags;
    }
}

// The same rules on all four lanes of a register at once (SSE / AVX2): per lane exactly the operations of the
// vu1fast helpers above, in the same order. Results are stored as whole registers (store<Dest> blends the lanes that
// aren't written), so a later whole-register load is forwarded from one store. Dest / Lanes: bit c = component c
// (x = bit 0). Without AVX2, V is four floats and each helper runs the vu1fast helper per lane.
namespace vu1v
{
#if defined(__AVX2__)
    using V = __m128;
    using VI = __m128i;

    VU1_FORCEINLINE V load(const float *p) { return _mm_loadu_ps(p); }
    VU1_FORCEINLINE V loadBytes(const uint8_t *p) { return _mm_loadu_ps(reinterpret_cast<const float *>(p)); }
    VU1_FORCEINLINE V splat(float value) { return _mm_set1_ps(value); }
    VU1_FORCEINLINE V add(V a, V b) { return _mm_add_ps(a, b); }
    VU1_FORCEINLINE V sub(V a, V b) { return _mm_sub_ps(a, b); }
    VU1_FORCEINLINE V mul(V a, V b) { return _mm_mul_ps(a, b); }
    // (a > b) ? a : b and (a < b) ? a : b per lane — MAXPS / MINPS return the second operand when equal (no NaN
    // reaches them: operands are clamped)
    VU1_FORCEINLINE V max(V a, V b) { return _mm_max_ps(a, b); }
    VU1_FORCEINLINE V min(V a, V b) { return _mm_min_ps(a, b); }

    // = vu1fast::operand per lane: denormals -> signed zero, Inf / NaN -> signed max
    VU1_FORCEINLINE V operand(V value)
    {
        const __m128i bits = _mm_castps_si128(value);
        const __m128i exponentMask = _mm_set1_epi32(0x7F800000);
        const __m128i exponent = _mm_and_si128(bits, exponentMask);
        const __m128i sign = _mm_and_si128(bits, _mm_set1_epi32(static_cast<int>(0x80000000u)));
        __m128i out = _mm_blendv_epi8(bits, sign, _mm_cmpeq_epi32(exponent, _mm_setzero_si128()));
        out = _mm_blendv_epi8(out, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF)), _mm_cmpeq_epi32(exponent, exponentMask));
        return _mm_castsi128_ps(out);
    }

    VU1_FORCEINLINE V abs(V value) { return _mm_andnot_ps(_mm_set1_ps(-0.0f), value); }

    // acc ± a·b = vu1fast::exactResult(c ± a * b, (double)c ± (double)a * (double)b) per written lane: the float
    // result unless the exact one is zero / below the smallest normal / above the largest float, which take the
    // scalar rule (rare)
    // First a test in float lanes: a result in [2^-100, 2^126] with |fl(a * b)| <= 2^20 |result| has its exact value
    // within ~7% of it (the float result is off by at most 2^-24 of |c ± fl(a·b)| plus 2^-24 of |a·b|), so the exact
    // value is in [FLT_MIN, FLT_MAX] and not zero — the double test below would keep the float result too. Only the
    // other lanes need the double test (a standalone check against the double-only version: 12 billion lane results
    // from random and edge-case operands — near zero, the bottom and the top of the range, zero operands — 0 differences).
    template <unsigned Dest, bool Subtract>
    VU1_FORCEINLINE V fused(V c, V a, V b)
    {
        const V floatProduct = _mm_mul_ps(a, b);
        const V value = Subtract ? _mm_sub_ps(c, floatProduct) : _mm_add_ps(c, floatProduct);
        {
            const V sign = _mm_set1_ps(-0.0f);
            const V magnitude = _mm_andnot_ps(sign, value);
            const V inRange = _mm_and_ps(_mm_and_ps(_mm_cmp_ps(magnitude, _mm_set1_ps(0x1p-100f), _CMP_GE_OQ),
                                                    _mm_cmp_ps(magnitude, _mm_set1_ps(0x1p126f), _CMP_LE_OQ)),
                                         _mm_cmp_ps(_mm_andnot_ps(sign, floatProduct),
                                                    _mm_mul_ps(magnitude, _mm_set1_ps(0x1p20f)), _CMP_LE_OQ));
            // a zero operand (common: matrix entries, w = 0): the product is an exact signed zero, so the exact sum is
            // c (clean: normal or zero) or a signed zero by the same IEEE rules as the float sum — the float result too
            const V zeroOperand = _mm_or_ps(_mm_cmp_ps(a, _mm_setzero_ps(), _CMP_EQ_OQ), _mm_cmp_ps(b, _mm_setzero_ps(), _CMP_EQ_OQ));
            if ((~static_cast<unsigned>(_mm_movemask_ps(_mm_or_ps(inRange, zeroOperand))) & Dest) == 0u)
                return value;
        }
        const __m256d product = _mm256_mul_pd(_mm256_cvtps_pd(a), _mm256_cvtps_pd(b));
        const __m256d exact = Subtract ? _mm256_sub_pd(_mm256_cvtps_pd(c), product)
                                       : _mm256_add_pd(_mm256_cvtps_pd(c), product);
        const __m256d magnitude = _mm256_andnot_pd(_mm256_set1_pd(-0.0), exact);
        const __m256d special = _mm256_or_pd(
            _mm256_cmp_pd(magnitude, _mm256_set1_pd(static_cast<double>(std::numeric_limits<float>::min())), _CMP_LT_OQ),
            _mm256_cmp_pd(magnitude, _mm256_set1_pd(static_cast<double>(std::numeric_limits<float>::max())), _CMP_GT_OQ));
        const unsigned lanes = static_cast<unsigned>(_mm256_movemask_pd(special)) & Dest;
        if (lanes == 0u)
            return value;
        alignas(16) float values[4];
        alignas(32) double exacts[4];
        _mm_store_ps(values, value);
        _mm256_store_pd(exacts, exact);
        for (unsigned component = 0; component < 4u; ++component)
        {
            if ((lanes & (1u << component)) != 0u)
                values[component] = vu1fast::exactResult(values[component], exacts[component]);
        }
        return _mm_load_ps(values);
    }

    // = vu1fast::floatToInt(value, scale) per lane: (double)value * scale, clamped to int32, truncated
    VU1_FORCEINLINE V floatToInt(V value, float scale)
    {
        __m256d scaled = _mm256_mul_pd(_mm256_cvtps_pd(value), _mm256_set1_pd(static_cast<double>(scale)));
        scaled = _mm256_min_pd(_mm256_max_pd(scaled, _mm256_set1_pd(static_cast<double>(std::numeric_limits<int32_t>::min()))),
                               _mm256_set1_pd(static_cast<double>(std::numeric_limits<int32_t>::max())));
        return _mm_castsi128_ps(_mm256_cvttpd_epi32(scaled));
    }

    // static_cast<float>(int32 lane bits) / divisor (ITOF4 / 12 / 15; ITOF0 = intToFloat(v))
    VU1_FORCEINLINE V intToFloat(V bits) { return _mm_cvtepi32_ps(_mm_castps_si128(bits)); }
    VU1_FORCEINLINE V intToFloat(V bits, float divisor) { return _mm_div_ps(_mm_cvtepi32_ps(_mm_castps_si128(bits)), _mm_set1_ps(divisor)); }

    // MR32: (y, z, w, x)
    VU1_FORCEINLINE V rotate(V value) { return _mm_shuffle_ps(value, value, _MM_SHUFFLE(0, 3, 2, 1)); }
    // an integer's bits in every lane (MFIR)
    VU1_FORCEINLINE V splatBits(int32_t bits) { return _mm_castsi128_ps(_mm_set1_epi32(bits)); }

    template <unsigned Dest>
    VU1_FORCEINLINE void store(float *p, V value)
    {
        if constexpr (Dest == 0xFu)
            _mm_storeu_ps(p, value);
        else
            _mm_storeu_ps(p, _mm_blend_ps(_mm_loadu_ps(p), value, static_cast<int>(Dest)));
    }
#else
    struct V
    {
        float f[4];
    };

    VU1_FORCEINLINE V load(const float *p) { V out; std::memcpy(out.f, p, 16); return out; }
    VU1_FORCEINLINE V loadBytes(const uint8_t *p) { V out; std::memcpy(out.f, p, 16); return out; }
    VU1_FORCEINLINE V splat(float value) { return V{{value, value, value, value}}; }
    template <typename F>
    VU1_FORCEINLINE V each(V a, V b, F f) { return V{{f(a.f[0], b.f[0]), f(a.f[1], b.f[1]), f(a.f[2], b.f[2]), f(a.f[3], b.f[3])}}; }
    VU1_FORCEINLINE V add(V a, V b) { return each(a, b, [](float x, float y) { return x + y; }); }
    VU1_FORCEINLINE V sub(V a, V b) { return each(a, b, [](float x, float y) { return x - y; }); }
    VU1_FORCEINLINE V mul(V a, V b) { return each(a, b, [](float x, float y) { return x * y; }); }
    VU1_FORCEINLINE V max(V a, V b) { return each(a, b, [](float x, float y) { return x > y ? x : y; }); }
    VU1_FORCEINLINE V min(V a, V b) { return each(a, b, [](float x, float y) { return x < y ? x : y; }); }
    VU1_FORCEINLINE V operand(V v) { return V{{vu1fast::operand(v.f[0]), vu1fast::operand(v.f[1]), vu1fast::operand(v.f[2]), vu1fast::operand(v.f[3])}}; }
    VU1_FORCEINLINE V abs(V v) { return V{{std::fabs(v.f[0]), std::fabs(v.f[1]), std::fabs(v.f[2]), std::fabs(v.f[3])}}; }

    template <unsigned Dest, bool Subtract>
    VU1_FORCEINLINE V fused(V c, V a, V b)
    {
        V out;
        for (unsigned k = 0; k < 4u; ++k)
        {
            const double exact = Subtract ? static_cast<double>(c.f[k]) - static_cast<double>(a.f[k]) * static_cast<double>(b.f[k])
                                          : static_cast<double>(c.f[k]) + static_cast<double>(a.f[k]) * static_cast<double>(b.f[k]);
            out.f[k] = vu1fast::exactResult(Subtract ? c.f[k] - a.f[k] * b.f[k] : c.f[k] + a.f[k] * b.f[k], exact);
        }
        return out;
    }

    VU1_FORCEINLINE V floatToInt(V v, float scale)
    {
        V out;
        for (unsigned k = 0; k < 4u; ++k)
        {
            const int32_t i = vu1fast::floatToInt(v.f[k], scale);
            std::memcpy(&out.f[k], &i, 4);
        }
        return out;
    }

    VU1_FORCEINLINE V intToFloat(V bits)
    {
        V out;
        for (unsigned k = 0; k < 4u; ++k)
        {
            int32_t i;
            std::memcpy(&i, &bits.f[k], 4);
            out.f[k] = static_cast<float>(i);
        }
        return out;
    }

    VU1_FORCEINLINE V intToFloat(V bits, float divisor)
    {
        V out = intToFloat(bits);
        for (unsigned k = 0; k < 4u; ++k)
            out.f[k] = out.f[k] / divisor;
        return out;
    }

    VU1_FORCEINLINE V rotate(V v) { return V{{v.f[1], v.f[2], v.f[3], v.f[0]}}; }
    VU1_FORCEINLINE V splatBits(int32_t bits) { float f; std::memcpy(&f, &bits, 4); return splat(f); }

    template <unsigned Dest>
    VU1_FORCEINLINE void store(float *p, V value)
    {
        for (unsigned k = 0; k < 4u; ++k)
        {
            if ((Dest & (1u << k)) != 0u)
                p[k] = value.f[k];
        }
    }
#endif
}

// The compiled code's access to the interpreter (a friend of VU1Interpreter). The generated functions take
// references to these once and then use them as plain variables.
struct Vu1Compiled
{
    static VU1State &state(VU1Interpreter &vu) { return vu.m_state; }
    static uint64_t &cycle(VU1Interpreter &vu) { return vu.m_cycle; }
    static const uint64_t &nextCommitCycle(VU1Interpreter &vu) { return vu.m_nextCommitCycle; }
    static const bool &xgkickActive(VU1Interpreter &vu) { return vu.m_xgkick.active; }
    static const bool &stopRequested(VU1Interpreter &vu) { return vu.m_stopRequested; }
    static auto &vfReady(VU1Interpreter &vu) { return vu.m_vfReady; }
    static auto &viReady(VU1Interpreter &vu) { return vu.m_viReady; }
    static auto &accReady(VU1Interpreter &vu) { return vu.m_accReady; }
    // the same as plain arrays (register r's row at + r * 4): no std::array calls left in the huge generated functions
    static uint64_t *vfReadyData(VU1Interpreter &vu) { return vu.m_vfReady[0].data(); }
    static uint64_t *viReadyData(VU1Interpreter &vu) { return vu.m_viReady.data(); }
    static uint64_t *accReadyData(VU1Interpreter &vu) { return vu.m_accReady.data(); }
    static uint64_t &settleCycle(VU1Interpreter &vu) { return vu.m_registerWritesSettleCycle; }
    static bool &branchBackupValid(VU1Interpreter &vu) { return vu.m_viBranchBackupValid; }
    static int32_t &branchBackupValue(VU1Interpreter &vu) { return vu.m_viBranchBackupValue; }
    static uint8_t &branchBackupReg(VU1Interpreter &vu) { return vu.m_viBranchBackupReg; }
    static const auto &fdiv(VU1Interpreter &vu) { return vu.m_fdiv; }
    static const auto &efu(VU1Interpreter &vu) { return vu.m_efu; }
    static const uint64_t &efuResourceReady(VU1Interpreter &vu) { return vu.m_efuResourceReady; }
    static uint8_t *vuData(VU1Interpreter &vu) { return vu.m_activeVuData; }
    static const auto *decodedPairs(VU1Interpreter &vu) { return vu.m_decodedCodeCache.data(); }

    static void commit(VU1Interpreter &vu) { vu.commitReadyPipelines(); }
    static void progressXgkick(VU1Interpreter &vu) { vu.progressXgkick(); }

    // the interpreter's stall loop for a pair; false = the budget ran out first (the pair does not run)
    template <typename Pair>
    static bool stall(VU1Interpreter &vu, const Pair &decoded, uint64_t budgetEnd)
    {
        uint64_t ready = vu.calculatePairReadyCycle(decoded);
        while (ready > vu.m_cycle)
        {
            if (ready >= budgetEnd)
            {
                vu.advanceTo(budgetEnd);
                break;
            }
            vu.advanceTo(ready);
            ready = vu.calculatePairReadyCycle(decoded);
        }
        return vu.m_cycle < budgetEnd;
    }

    static void queueStore(VU1Interpreter &vu, uint32_t address, const uint32_t words[4], uint8_t laneMask)
    {
        vu.queueStore(address, words, laneMask);
    }
    static void queueClip(VU1Interpreter &vu, uint32_t clip) { vu.queueClip(clip); }
    static void queueFcset(VU1Interpreter &vu, uint32_t clip) { vu.queueFcset(clip); }
    static void startXgkick(VU1Interpreter &vu, uint32_t qwordAddress) { vu.startXgkick(qwordAddress); }

    // DIV / SQRT / RSQRT as VU1Interpreter::execLower does them
    static void div(VU1Interpreter &vu, float numRaw, float denRaw)
    {
        const float num = vu1fast::operand(numRaw);
        const float den = vu1fast::operand(denRaw);
        uint32_t statusDi = 0u;
        float value = 0.0f;
        if (den == 0.0f)
        {
            statusDi = num == 0.0f ? 0x10u : 0x20u;
            value = std::signbit(num) != std::signbit(den) ? -std::numeric_limits<float>::max()
                                                           : std::numeric_limits<float>::max();
        }
        else
        {
            value = num / den;
        }
        vu.queueQ(vu1fast::result(value), 7u, statusDi);
    }
    static void sqrt(VU1Interpreter &vu, float valueRaw)
    {
        const float value = vu1fast::operand(valueRaw);
        vu.queueQ(std::sqrt(std::fabs(value)), 7u, value < 0.0f ? 0x10u : 0u);
    }
    static void rsqrt(VU1Interpreter &vu, float numRaw, float radicandRaw)
    {
        const float num = vu1fast::operand(numRaw);
        const float radicand = vu1fast::operand(radicandRaw);
        const float den = std::sqrt(std::fabs(radicand));
        uint32_t statusDi = radicand < 0.0f ? 0x10u : 0u;
        float value = 0.0f;
        if (den != 0.0f)
            value = num / den;
        else
        {
            statusDi = num == 0.0f ? 0x10u : 0x20u;
            value = std::signbit(num) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        }
        vu.queueQ(vu1fast::result(value), 13u, statusDi);
    }

    // the instructions the generator leaves to the interpreter
    static void runUpper(VU1Interpreter &vu, uint32_t instr) { vu.runUpper(instr); }
    static void execLower(VU1Interpreter &vu, uint32_t instr, uint32_t upper)
    {
        vu.execLower(instr, vu.m_activeVuData, vu.m_activeVuDataSize, *vu.m_activeGs, vu.m_activeMemory, upper);
    }
};

#endif
