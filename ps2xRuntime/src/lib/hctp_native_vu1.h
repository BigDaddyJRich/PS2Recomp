#pragma once

// HCTP native model draws (notes\VU1_PROGRAMS.md): the VU1 programs' math outside the VU1 emulation.
#include <cstdint>

#include "runtime/ps2_vu1.h"

class GS;

void hctpNativeVu1Init();
// HCTP_NATIVE_P08=1: a p08 chunk that needs no clipping is drawn from the spec (packet straight to the GS);
// true = done, the VU1 resume is skipped
bool hctpNativeVu1TryDraw(const uint8_t *code, uint8_t *data, const VU1State &st, uint32_t top, GS &gs);
// around each VU1 MSCNT (HCTP_NATIVE_VERIFY=1): code / data = VU1 memories, st = VU1 state before the resume,
// top = TOP (qwords)
void hctpNativeVu1BeforeResume(const uint8_t *code, const uint8_t *data, const VU1State &st, uint32_t top);
void hctpNativeVu1AfterResume(const uint8_t *data);
// p01 at MSCAL 0x00a0: HCTP_NATIVE_P01=1 -> computed natively, true = skip the VU1; verify mode compares after the run
bool hctpNativeVu1Mscal(const uint8_t *code, uint8_t *data, uint32_t startPC);
void hctpNativeVu1AfterMscal(const uint8_t *data);
