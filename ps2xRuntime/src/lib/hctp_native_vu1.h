#pragma once

// HCTP native model draws (notes\VU1_PROGRAMS.md): the VU1 programs' math outside the VU1 emulation.
#include <cstdint>

void hctpNativeVu1Init();
// around each VU1 MSCNT: code / data = VU1 memories, pc = where the VU1 resumes, top = TOP (qwords)
void hctpNativeVu1BeforeResume(const uint8_t *code, const uint8_t *data, uint32_t pc, uint32_t top);
void hctpNativeVu1AfterResume(const uint8_t *data);
