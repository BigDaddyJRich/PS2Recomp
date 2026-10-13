#pragma once

#include <cstdint>

class PS2Memory;

// The writes into VU1 code memory, by generation (PS2Memory::getVU1CodeGeneration), so that the VU1 decode cache can
// re-decode only the pairs written since it was built. Kept out of ps2_memory.h, which every recompiled function
// includes. The writers are serialised: VIF1 MPG uploads on the chain worker, EE stores after a chain fence.
namespace ps2_vu_code_log
{
    // a write of the bytes [begin, end) of the memory's VU1 code memory, recorded as the generation it brings
    void record(const PS2Memory *memory, uint64_t generation, uint32_t begin, uint32_t end);

    // the bytes written between two generations as one range [begin, end) (begin == end: none); false when the log
    // no longer reaches back that far (then anything may have changed)
    bool writtenSince(const PS2Memory *memory, uint64_t since, uint64_t now, uint32_t &begin, uint32_t &end);
}
