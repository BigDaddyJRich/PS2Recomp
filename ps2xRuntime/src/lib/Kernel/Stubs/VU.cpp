#include "Common.h"
#include "VU.h"
//TODO use glm
#include <iostream>
#include <atomic>
#include <cmath>
#define VCNT(name) do{ static std::atomic<int> c_{0}; int v_=c_.fetch_add(1)+1; if(v_==1||v_==10||v_==100||v_==1000||v_==10000||v_==100000) std::cerr << "[vu0cnt] " << name << " n=" << v_ << std::endl; }while(0)
static void vlogM(const char* tag, const float* m){ std::cerr << " " << tag << "="; for(int i=0;i<16;++i) std::cerr << m[i] << (i==15?"":","); }
static void vlogV(const char* tag, const float* v){ std::cerr << " " << tag << "=" << v[0] << "," << v[1] << "," << v[2] << "," << v[3]; }


namespace ps2_stubs
{
    void sceVu0ecossin(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ecossin");
        TODO_NAMED("sceVu0ecossin", rdram, ctx, runtime);
    }

    namespace
    {
        bool readVuVec4f(uint8_t *rdram, uint32_t addr, float (&out)[4])
        {
            const uint8_t *ptr = getConstMemPtr(rdram, addr);
            if (!ptr)
            {
                return false;
            }
            std::memcpy(out, ptr, sizeof(out));
            return true;
        }

        bool writeVuVec4f(uint8_t *rdram, uint32_t addr, const float (&in)[4])
        {
            uint8_t *ptr = getMemPtr(rdram, addr);
            if (!ptr)
            {
                return false;
            }
            std::memcpy(ptr, in, sizeof(in));
            return true;
        }

        bool readVuVec4i(uint8_t *rdram, uint32_t addr, int32_t (&out)[4])
        {
            const uint8_t *ptr = getConstMemPtr(rdram, addr);
            if (!ptr)
            {
                return false;
            }
            std::memcpy(out, ptr, sizeof(out));
            return true;
        }

        bool writeVuVec4i(uint8_t *rdram, uint32_t addr, const int32_t (&in)[4])
        {
            uint8_t *ptr = getMemPtr(rdram, addr);
            if (!ptr)
            {
                return false;
            }
            std::memcpy(ptr, in, sizeof(in));
            return true;
        }

        bool readVuMatrix4f(uint8_t *rdram, uint32_t addr, float (&out)[16])
        {
            const uint8_t *ptr = getConstMemPtr(rdram, addr);
            if (!ptr)
            {
                return false;
            }
            std::memcpy(out, ptr, sizeof(out));
            return true;
        }

        bool writeVuMatrix4f(uint8_t *rdram, uint32_t addr, const float (&in)[16])
        {
            uint8_t *ptr = getMemPtr(rdram, addr);
            if (!ptr)
            {
                return false;
            }
            std::memcpy(ptr, in, sizeof(in));
            return true;
        }

        void mulVuMatrix(const float (&lhs)[16], const float (&rhs)[16], float (&out)[16])
        {
            std::fill(std::begin(out), std::end(out), 0.0f);
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    for (int k = 0; k < 4; ++k)
                    {
                        out[4 * i + j] += rhs[4 * k + j] * lhs[4 * i + k];
                    }
                }
            }
        }

        void makeIdentityMatrix(float (&out)[16])
        {
            std::fill(std::begin(out), std::end(out), 0.0f);
            out[0] = 1.0f;
            out[5] = 1.0f;
            out[10] = 1.0f;
            out[15] = 1.0f;
        }

        void vuApply(const float (&m)[16], const float (&v)[4], float (&o)[4])
        {
            for (int j = 0; j < 4; ++j)
                o[j] = m[j] * v[0] + m[4 + j] * v[1] + m[8 + j] * v[2] + m[12 + j] * v[3];
        }
        void vuAffineInverse(const float (&m)[16], float (&o)[16])
        {
            // Exact libvu0 sceVu0InversMatrix behaviour (verified vs ELF code in emulator):
            // 3x3 = transpose of input 3x3 (NOT a true inverse for scaled matrices),
            // t' = -(R^T * t), column w = 0, o[15] = m[15].
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) o[i*4+j] = m[j*4+i];
            o[3] = o[7] = o[11] = 0.0f;
            for (int j = 0; j < 3; ++j) o[12+j] = -(o[0*4+j]*m[12] + o[1*4+j]*m[13] + o[2*4+j]*m[14]);
            o[15] = m[15];
        }
        void vuNorm3(float (&v)[4])
        {
            float l = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
            if (l > 0.0f) { v[0]/=l; v[1]/=l; v[2]/=l; }
        }
        void vuCross(const float (&a)[4], const float (&b)[4], float (&o)[4])
        {
            o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0]; o[3]=0.0f;
        }
        void vuRtp(const float (&m)[16], const float (&v)[4], int32_t (&o)[4], uint32_t mode = 0)
        {
            float r[4]; vuApply(m, v, r);
            float q = (r[3] != 0.0f) ? 1.0f / r[3] : 0.0f;
            for (int i = 0; i < 2; ++i) o[i] = static_cast<int32_t>(r[i] * q * 16.0f);
            if (mode == 0) { o[2] = static_cast<int32_t>(r[2] * q * 16.0f); o[3] = static_cast<int32_t>(r[3] * 16.0f); }
            else { o[2] = static_cast<int32_t>(r[2] * q); o[3] = static_cast<int32_t>(r[3]); } // FTOI0.zw
        }

    }

    void sceVpu0Reset(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceVu0AddVector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0AddVector");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t lhsAddr = getRegU32(ctx, 5);
        const uint32_t rhsAddr = getRegU32(ctx, 6);
        float lhs[4]{}, rhs[4]{}, out[4]{};
        if (readVuVec4f(rdram, lhsAddr, lhs) && readVuVec4f(rdram, rhsAddr, rhs))
        {
            for (int i = 0; i < 4; ++i)
            {
                out[i] = lhs[i] + rhs[i];
            }
            (void)writeVuVec4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0ApplyMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ApplyMatrix");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t matrixAddr = getRegU32(ctx, 5);
        const uint32_t srcAddr = getRegU32(ctx, 6);
        float matrix[16]{};
        float src[4]{};
        float out[4]{};
        if (readVuMatrix4f(rdram, matrixAddr, matrix) && readVuVec4f(rdram, srcAddr, src))
        {
            // Match libvux VuxApplyMatrix math while honoring the imported EE ABI:
            // a0=out, a1=matrix, a2=vector.
            out[0] = (matrix[0] * src[0]) + (matrix[4] * src[1]) + (matrix[8] * src[2]) + (matrix[12] * src[3]);
            out[1] = (matrix[1] * src[0]) + (matrix[5] * src[1]) + (matrix[9] * src[2]) + (matrix[13] * src[3]);
            out[2] = (matrix[2] * src[0]) + (matrix[6] * src[1]) + (matrix[10] * src[2]) + (matrix[14] * src[3]);
            out[3] = (matrix[3] * src[0]) + (matrix[7] * src[1]) + (matrix[11] * src[2]) + (matrix[15] * src[3]);
            (void)writeVuVec4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0CameraMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0CameraMatrix");
        const uint32_t d = getRegU32(ctx, 4), pa = getRegU32(ctx, 5), za = getRegU32(ctx, 6), ya = getRegU32(ctx, 7);
        float p[4]{}, z[4]{}, y[4]{}, x[4]{};
        if (readVuVec4f(rdram, pa, p) && readVuVec4f(rdram, za, z) && readVuVec4f(rdram, ya, y))
        {
            vuNorm3(z); vuCross(y, z, x); vuNorm3(x); vuCross(z, x, y);
            float c[16]{}, o[16]{};
            for (int i=0;i<3;++i){ c[i]=x[i]; c[4+i]=y[i]; c[8+i]=z[i]; c[12+i]=p[i]; }
            c[15]=1.0f;
            vuAffineInverse(c, o); writeVuMatrix4f(rdram, d, o);
            { static std::atomic<int> n{0}; static std::atomic<int> nn{0}; const int k = n.fetch_add(1); if (std::isnan(p[0]) || std::isnan(z[0])) nn.fetch_add(1); if (k < 8 || (k % 500) == 0) { std::cerr << "[vu0stub] CameraMatrix k=" << k << " nan=" << nn.load() << " ra=0x" << std::hex << getRegU32(ctx, 31) << std::dec; { const uint32_t self = getRegU32(ctx, 18); std::cerr << " s2=0x" << std::hex << self << " raw:"; for (uint32_t o = 0x30; o < 0xA8; o += 4) { uint32_t w = 0; if (const uint8_t *q = getConstMemPtr(rdram, self + o)) std::memcpy(&w, q, 4); std::cerr << " " << o << "=" << w; } std::cerr << std::dec; } vlogV("p",p); vlogV("zd_in",z); vlogV("yd",y); vlogM("out",o); std::cerr << std::endl; } }
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0ClampVector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ClampVector");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5);
        const float lo = ctx ? ctx->f[12] : 0.0f, hi = ctx ? ctx->f[13] : 0.0f;
        float x[4]{};
        if (readVuVec4f(rdram, a, x)) { for (int i=0;i<4;++i) x[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]); writeVuVec4f(rdram, d, x); }
        setReturnS32(ctx, 0);
    }

    void sceVu0ClipAll(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ClipAll");
        setReturnS32(ctx, 0);
    }

    void sceVu0ClipScreen(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ClipScreen");
        setReturnS32(ctx, 0);
    }

    void sceVu0ClipScreen3(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ClipScreen3");
        setReturnS32(ctx, 0);
    }

    void sceVu0CopyMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0CopyMatrix");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        uint8_t *dst = getMemPtr(rdram, dstAddr);
        const uint8_t *src = getConstMemPtr(rdram, srcAddr);
        if (dst && src)
        {
            std::memcpy(dst, src, sizeof(float) * 16u);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0CopyVector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0CopyVector");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        uint8_t *dst = getMemPtr(rdram, dstAddr);
        const uint8_t *src = getConstMemPtr(rdram, srcAddr);
        if (dst && src)
        {
            std::memcpy(dst, src, sizeof(float) * 4u);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0CopyVectorXYZ(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0CopyVectorXYZ");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        uint8_t *dst = getMemPtr(rdram, dstAddr);
        const uint8_t *src = getConstMemPtr(rdram, srcAddr);
        if (dst && src)
        {
            std::memcpy(dst, src, sizeof(float) * 3u);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0DivVector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0DivVector");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5);
        const float q = ctx ? ctx->f[12] : 1.0f;
        float x[4]{};
        if (readVuVec4f(rdram, a, x) && q != 0.0f) { for (int i=0;i<4;++i) x[i]/=q; writeVuVec4f(rdram, d, x); }
        setReturnS32(ctx, 0);
    }

    void sceVu0DivVectorXYZ(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0DivVectorXYZ");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5);
        const float q = ctx ? ctx->f[12] : 1.0f;
        float x[4]{};
        if (readVuVec4f(rdram, a, x) && q != 0.0f) { for (int i=0;i<3;++i) x[i]/=q; writeVuVec4f(rdram, d, x); }
        setReturnS32(ctx, 0);
    }

    void sceVu0DropShadowMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0DropShadowMatrix");
        // Hand translation of the game's own libvu0 code at 0x104188 (SLUS_207.87).
        // void sceVu0DropShadowMatrix(FMATRIX m, FVECTOR lp, float a, float b, float c, int mode)
        // a0=m a1=lp f12=a f13=b f14=c a2=mode. Plane a*x+b*y+c*z=1 (as the code implies).
        const uint32_t d = getRegU32(ctx, 4), lpAddr = getRegU32(ctx, 5);
        const int32_t mode = (int32_t)getRegU32(ctx, 6);
        const float a = ctx->f[12], b = ctx->f[13], c = ctx->f[14];
        float lp[4]{};
        readVuVec4f(rdram, lpAddr, lp);
        const float l0 = lp[0], l1 = lp[1], l2 = lp[2];
        float m[16]{};
        const float dd = a * l0 + b * l1 + c * l2;
        if (mode != 0)
        {   // point light at lp
            const float k = 1.0f - dd;
            m[0] = a * l0 + k; m[1] = a * l1; m[2] = a * l2; m[3] = a;
            m[4] = b * l0; m[5] = b * l1 + k; m[6] = b * l2; m[7] = b;
            m[8] = c * l0; m[9] = c * l1; m[10] = c * l2 + k; m[11] = c;
            m[12] = -l0; m[13] = -l1; m[14] = -l2; m[15] = k - 1.0f;
        }
        else
        {   // parallel light, direction lp
            float s; // -1/dd with PS2 x/0 = +-Fmax
            if (dd == 0.0f) s = std::signbit(dd) ? 3.402823466e+38f : -3.402823466e+38f;
            else s = -1.0f / dd;
            m[0] = s * (a * l0 - dd); m[1] = s * (a * l1); m[2] = s * (a * l2); m[3] = 0.0f;
            m[4] = s * (b * l0); m[5] = s * (b * l1 - dd); m[6] = s * (b * l2); m[7] = 0.0f;
            m[8] = s * (c * l0); m[9] = s * (c * l1); m[10] = s * (c * l2 - dd); m[11] = 0.0f;
            m[12] = s * (-l0); m[13] = s * (-l1); m[14] = s * (-l2); m[15] = s * (-dd);
        }
        writeVuMatrix4f(rdram, d, m);
        { static std::atomic<int> n{0}; if (n.fetch_add(1) < 8) { std::cerr << "[vu0stub] DropShadow mode=" << mode << " abc=" << a << "," << b << "," << c; vlogV("lp", lp); vlogM("out", m); std::cerr << std::endl; } }
        setReturnS32(ctx, 0);
    }

    void sceVu0FTOI0Vector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0FTOI0Vector");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        float src[4]{};
        int32_t out[4]{};
        if (readVuVec4f(rdram, srcAddr, src))
        {
            for (int i = 0; i < 4; ++i)
            {
                out[i] = static_cast<int32_t>(src[i]);
            }
            (void)writeVuVec4i(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0FTOI4Vector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0FTOI4Vector");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        float src[4]{};
        int32_t out[4]{};
        if (readVuVec4f(rdram, srcAddr, src))
        {
            for (int i = 0; i < 4; ++i)
            {
                out[i] = static_cast<int32_t>(src[i] * 16.0f);
            }
            (void)writeVuVec4i(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0InnerProduct(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0InnerProduct");
        const uint32_t lhsAddr = getRegU32(ctx, 4);
        const uint32_t rhsAddr = getRegU32(ctx, 5);
        float lhs[4]{}, rhs[4]{};
        float dot = 0.0f;
        if (readVuVec4f(rdram, lhsAddr, lhs) && readVuVec4f(rdram, rhsAddr, rhs))
        {
            dot = (lhs[0] * rhs[0]) + (lhs[1] * rhs[1]) + (lhs[2] * rhs[2]) + (lhs[3] * rhs[3]);
        }

        if (ctx)
        {
            ctx->f[0] = dot;
        }
        uint32_t raw = 0u;
        std::memcpy(&raw, &dot, sizeof(raw));
        setReturnU32(ctx, raw);
    }

    void sceVu0InterVector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0InterVector");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5), b = getRegU32(ctx, 6);
        const float r = ctx ? ctx->f[12] : 0.0f;
        float x[4]{}, y[4]{}, o[4]{};
        if (readVuVec4f(rdram, a, x) && readVuVec4f(rdram, b, y)) { for (int i=0;i<4;++i) o[i]=x[i]*r+y[i]*(1.0f-r); writeVuVec4f(rdram, d, o); }
        setReturnS32(ctx, 0);
    }

    void sceVu0InterVectorXYZ(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0InterVectorXYZ");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5), b = getRegU32(ctx, 6);
        const float r = ctx ? ctx->f[12] : 0.0f;
        float x[4]{}, y[4]{}, o[4]{};
        if (readVuVec4f(rdram, a, x) && readVuVec4f(rdram, b, y)) { for (int i=0;i<3;++i) o[i]=x[i]*r+y[i]*(1.0f-r); o[3]=x[3]; writeVuVec4f(rdram, d, o); }
        setReturnS32(ctx, 0);
    }

    void sceVu0InversMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0InversMatrix");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5);
        float m[16]{}, o[16]{};
        if (readVuMatrix4f(rdram, a, m)) { vuAffineInverse(m, o); writeVuMatrix4f(rdram, d, o);
            { static std::atomic<int> n{0}; if (n.fetch_add(1) < 40) { std::cerr << "[vu0stub] InversMatrix"; vlogM("in",m); vlogM("out",o); std::cerr << std::endl; } } }
        setReturnS32(ctx, 0);
    }

    void sceVu0ITOF0Vector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ITOF0Vector");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        int32_t src[4]{};
        float out[4]{};
        if (readVuVec4i(rdram, srcAddr, src))
        {
            for (int i = 0; i < 4; ++i)
            {
                out[i] = static_cast<float>(src[i]);
            }
            (void)writeVuVec4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0ITOF12Vector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ITOF12Vector");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        int32_t src[4]{};
        float out[4]{};
        if (readVuVec4i(rdram, srcAddr, src))
        {
            for (int i = 0; i < 4; ++i)
            {
                out[i] = static_cast<float>(src[i]) / 4096.0f;
            }
            (void)writeVuVec4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0ITOF4Vector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ITOF4Vector");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        int32_t src[4]{};
        float out[4]{};
        if (readVuVec4i(rdram, srcAddr, src))
        {
            for (int i = 0; i < 4; ++i)
            {
                out[i] = static_cast<float>(src[i]) / 16.0f;
            }
            (void)writeVuVec4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0LightColorMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0LightColorMatrix");
        const uint32_t d = getRegU32(ctx, 4);
        float o[16]{};
        for (int i = 0; i < 4; ++i) { float v[4]{}; readVuVec4f(rdram, getRegU32(ctx, 5 + i), v); for (int j=0;j<4;++j) o[i*4+j]=v[j]; }
        writeVuMatrix4f(rdram, d, o);
        setReturnS32(ctx, 0);
    }

    void sceVu0MulMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0MulMatrix");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5), b = getRegU32(ctx, 6);
        float x[16]{}, y[16]{}, o[16]{};
        if (readVuMatrix4f(rdram, a, x) && readVuMatrix4f(rdram, b, y)) { mulVuMatrix(x, y, o); writeVuMatrix4f(rdram, d, o);
            { static std::atomic<int> n{0}; if (n.fetch_add(1) < 60) { std::cerr << "[vu0stub] MulMatrix"; vlogM("a",x); vlogM("b",y); vlogM("out",o); std::cerr << std::endl; } } }
        setReturnS32(ctx, 0);
    }

    void sceVu0MulVector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0MulVector");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5), b = getRegU32(ctx, 6);
        float x[4]{}, y[4]{}, o[4]{};
        if (readVuVec4f(rdram, a, x) && readVuVec4f(rdram, b, y)) { for (int i=0;i<4;++i) o[i]=x[i]*y[i]; writeVuVec4f(rdram, d, o); }
        setReturnS32(ctx, 0);
    }

    void sceVu0Normalize(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0Normalize");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        float src[4]{}, out[4]{};
        if (readVuVec4f(rdram, srcAddr, src))
        {
            // libvu0: length from xyz only, result w = 0 (verified against ELF code via emulator)
            const float len = std::sqrt((src[0] * src[0]) + (src[1] * src[1]) + (src[2] * src[2]));
            if (len > 1.0e-6f)
            {
                const float invLen = 1.0f / len;
                for (int i = 0; i < 3; ++i)
                {
                    out[i] = src[i] * invLen;
                }
            }
            (void)writeVuVec4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0NormalLightMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0NormalLightMatrix");
        const uint32_t d = getRegU32(ctx, 4);
        float o[16]{};
        for (int i = 0; i < 3; ++i) { float v[4]{}; readVuVec4f(rdram, getRegU32(ctx, 5 + i), v); vuNorm3(v); o[0*4+i]=-v[0]; o[1*4+i]=-v[1]; o[2*4+i]=-v[2]; }
        o[15] = 1.0f; // libvu0: m[3] = {0,0,0,1} before the transpose
        writeVuMatrix4f(rdram, d, o);
        setReturnS32(ctx, 0);
    }

    void sceVu0OuterProduct(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0OuterProduct");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t lhsAddr = getRegU32(ctx, 5);
        const uint32_t rhsAddr = getRegU32(ctx, 6);
        float lhs[4]{}, rhs[4]{}, out[4]{};
        if (readVuVec4f(rdram, lhsAddr, lhs) && readVuVec4f(rdram, rhsAddr, rhs))
        {
            out[0] = (lhs[1] * rhs[2]) - (lhs[2] * rhs[1]);
            out[1] = (lhs[2] * rhs[0]) - (lhs[0] * rhs[2]);
            out[2] = (lhs[0] * rhs[1]) - (lhs[1] * rhs[0]);
            out[3] = 0.0f;
            (void)writeVuVec4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0RotMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0RotMatrix");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5), r = getRegU32(ctx, 6);
        float rot[4]{}; float m[16]{};
        if (readVuVec4f(rdram, r, rot) && readVuMatrix4f(rdram, a, m))
        {
            float tmp[16]{}, t2[16]{}, rm[16];
            auto mk = [&](int axis, float ang, const float (&src)[16], float (&out)[16]) {
                makeIdentityMatrix(rm); float cs=std::cos(ang), sn=std::sin(ang);
                if (axis==0){rm[5]=cs;rm[6]=sn;rm[9]=-sn;rm[10]=cs;}
                else if (axis==1){rm[0]=cs;rm[2]=-sn;rm[8]=sn;rm[10]=cs;}
                else {rm[0]=cs;rm[1]=sn;rm[4]=-sn;rm[5]=cs;}
                mulVuMatrix(src, rm, out); };
            mk(2, rot[2], m, tmp); mk(1, rot[1], tmp, t2); mk(0, rot[0], t2, tmp);
            writeVuMatrix4f(rdram, d, tmp);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0RotMatrixX(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0RotMatrixX");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        const float angle = ctx ? ctx->f[12] : 0.0f;
        float src[16]{}, rot[16]{}, out[16]{};
        if (readVuMatrix4f(rdram, srcAddr, src))
        {
            makeIdentityMatrix(rot);
            const float cs = std::cos(angle);
            const float sn = std::sin(angle);
            rot[5] = cs;
            rot[6] = sn;
            rot[9] = -sn;
            rot[10] = cs;
            mulVuMatrix(src, rot, out);
            (void)writeVuMatrix4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0RotMatrixY(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0RotMatrixY");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        const float angle = ctx ? ctx->f[12] : 0.0f;
        float src[16]{}, rot[16]{}, out[16]{};
        if (readVuMatrix4f(rdram, srcAddr, src))
        {
            makeIdentityMatrix(rot);
            const float cs = std::cos(angle);
            const float sn = std::sin(angle);
            rot[0] = cs;
            rot[2] = -sn;
            rot[8] = sn;
            rot[10] = cs;
            mulVuMatrix(src, rot, out);
            (void)writeVuMatrix4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0RotMatrixZ(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0RotMatrixZ");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        const float angle = ctx ? ctx->f[12] : 0.0f;
        float src[16]{}, rot[16]{}, out[16]{};
        if (readVuMatrix4f(rdram, srcAddr, src))
        {
            makeIdentityMatrix(rot);
            const float cs = std::cos(angle);
            const float sn = std::sin(angle);
            rot[0] = cs;
            rot[1] = sn;
            rot[4] = -sn;
            rot[5] = cs;
            mulVuMatrix(src, rot, out);
            (void)writeVuMatrix4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0RotTransPers(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0RotTransPers");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5), b = getRegU32(ctx, 6);
        float m[16]{}, v[4]{}; int32_t o[4]{};
        if (readVuMatrix4f(rdram, a, m) && readVuVec4f(rdram, b, v)) { vuRtp(m, v, o, getRegU32(ctx, 7)); writeVuVec4i(rdram, d, o); }
        setReturnS32(ctx, 0);
    }

    void sceVu0RotTransPersN(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0RotTransPersN");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5), b = getRegU32(ctx, 6);
        const uint32_t n = getRegU32(ctx, 7);
        float m[16]{};
        if (readVuMatrix4f(rdram, a, m))
            for (uint32_t i = 0; i < n; ++i) { float v[4]{}; int32_t o[4]{}; if (readVuVec4f(rdram, b + i*16, v)) { vuRtp(m, v, o, getRegU32(ctx, 8)); writeVuVec4i(rdram, d + i*16, o); } }
        setReturnS32(ctx, 0);
    }

    void sceVu0ScaleVector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ScaleVector");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        float src[4]{}, out[4]{};
        float scale = ctx ? ctx->f[12] : 0.0f;
        if (scale == 0.0f)
        {
            uint32_t raw = getRegU32(ctx, 6);
            std::memcpy(&scale, &raw, sizeof(scale));
            if (scale == 0.0f)
            {
                scale = static_cast<float>(getRegU32(ctx, 6));
            }
        }

        if (readVuVec4f(rdram, srcAddr, src))
        {
            for (int i = 0; i < 4; ++i)
            {
                out[i] = src[i] * scale;
            }
            (void)writeVuVec4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0ScaleVectorXYZ(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ScaleVectorXYZ");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5);
        const float r = ctx ? ctx->f[12] : 0.0f;
        float x[4]{};
        if (readVuVec4f(rdram, a, x)) { for (int i=0;i<3;++i) x[i]*=r; writeVuVec4f(rdram, d, x); }
        setReturnS32(ctx, 0);
    }

    void sceVu0SubVector(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0SubVector");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t lhsAddr = getRegU32(ctx, 5);
        const uint32_t rhsAddr = getRegU32(ctx, 6);
        float lhs[4]{}, rhs[4]{}, out[4]{};
        if (readVuVec4f(rdram, lhsAddr, lhs) && readVuVec4f(rdram, rhsAddr, rhs))
        {
            for (int i = 0; i < 4; ++i)
            {
                out[i] = lhs[i] - rhs[i];
            }
            (void)writeVuVec4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0TransMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0TransMatrix");
        const uint32_t d = getRegU32(ctx, 4), a = getRegU32(ctx, 5), b = getRegU32(ctx, 6);
        float m[16]{}, t[4]{};
        if (readVuMatrix4f(rdram, a, m) && readVuVec4f(rdram, b, t)) { for (int i=0;i<3;++i) m[12+i]+=t[i]; /* libvu0: w of translation row unchanged */ writeVuMatrix4f(rdram, d, m); }
        setReturnS32(ctx, 0);
    }

    void sceVu0TransposeMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0TransposeMatrix");
        const uint32_t dstAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        float src[16]{};
        float out[16]{};
        if (readVuMatrix4f(rdram, srcAddr, src))
        {
            for (int row = 0; row < 4; ++row)
            {
                for (int col = 0; col < 4; ++col)
                {
                    out[4 * row + col] = src[4 * col + row];
                }
            }
            (void)writeVuMatrix4f(rdram, dstAddr, out);
        }
        setReturnS32(ctx, 0);
    }

    void sceVu0UnitMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0UnitMatrix");
        const uint32_t dstAddr = getRegU32(ctx, 4); // sceVu0FMATRIX dst
        alignas(16) const float identity[16] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};

        if (!writeGuestBytes(rdram, runtime, dstAddr, reinterpret_cast<const uint8_t *>(identity), sizeof(identity)))
        {
            static uint32_t warnCount = 0;
            if (warnCount < 8)
            {
                std::cerr << "sceVu0UnitMatrix: failed to write matrix at 0x"
                          << std::hex << dstAddr << std::dec << std::endl;
                ++warnCount;
            }
        }

        setReturnS32(ctx, 0);
    }

    void sceVu0ViewScreenMatrix(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        VCNT("sceVu0ViewScreenMatrix");
        const uint32_t d = getRegU32(ctx, 4);
        // HCTP widescreen (ElHecht's PCSX2 16:9 patch): ax x 0.75, and the game camera's width constant at 0x7F00
        // (hctp.toml patches CCamGame5::Action 0x206a5c to read it): 0.53125 at 16:9, the stock 1.0 otherwise
        static const bool wide = [] { const char *w = std::getenv("PS2X_WIDESCREEN"); return w && *w && *w != '0'; }();
        { const float camW = wide ? 0.53125f : 1.0f; if (uint8_t *p = getMemPtr(rdram, 0x7F00u)) std::memcpy(p, &camW, 4); }
        const float scrz = ctx->f[12], ax = ctx->f[13] * (wide ? 0.75f : 1.0f), ay = ctx->f[14], cx = ctx->f[15], cy = ctx->f[16];
        const float zmin = ctx->f[17], zmax = ctx->f[18], nearz = ctx->f[19];
        float farz = 0.0f;
        { const uint8_t *sp = getConstMemPtr(rdram, getRegU32(ctx, 29)); if (sp) std::memcpy(&farz, sp, 4); }
        float o[16]{};
        const float dz = farz - nearz;
        const float az = (dz != 0.0f) ? farz * nearz * (zmax - zmin) / dz : 0.0f; // libvu0: near->zmax, far->zmin
        const float cz = (dz != 0.0f) ? (zmin * farz - zmax * nearz) / dz : 0.0f;
        o[0] = scrz * ax; o[5] = scrz * ay;
        o[8] = cx; o[9] = cy; o[10] = cz; o[11] = 1.0f;
        o[14] = az;
        writeVuMatrix4f(rdram, d, o);
        { static float ls=-1,lcx=-1,lcy=-1,lay=-1; static std::atomic<int> n{0}; bool chg=(ls!=scrz||lcx!=cx||lcy!=cy||lay!=ay); ls=scrz;lcx=cx;lcy=cy;lay=ay; if (chg && n.fetch_add(1) < 60) { std::cerr << "[vu0stub] ViewScreen scrz=" << scrz << " ax=" << ax << " ay=" << ay << " cx=" << cx << " cy=" << cy << " zmin=" << zmin << " zmax=" << zmax << " near=" << nearz << " far=" << farz; vlogM("out",o); std::cerr << std::endl; } }
        setReturnS32(ctx, 0);
    }
}
