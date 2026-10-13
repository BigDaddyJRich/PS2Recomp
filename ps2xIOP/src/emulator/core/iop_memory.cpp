#include "iop_memory.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kDmaSpu0Chcr = 0x1F8010C8u;
        constexpr uint32_t kDmaSpu1Chcr = 0x1F801508u;
        constexpr uint32_t kDmaStart = 1u << 24u;
        constexpr int kDmaSpu0Irq = 0x24;
        constexpr int kDmaSpu1Irq = 0x28;

        // the SPU2's registers (0x1F900000-0x1F9007FF), handled by Spu2 in 16-bit units
        bool isSpu2Register(uint32_t phys)
        {
            return phys >= IopMemory::Spu2Base && phys < IopMemory::Spu2Base + Spu2::kRegisterBytes;
        }

        uint32_t alignUp(uint32_t value, uint32_t alignment)
        {
            return (value + alignment - 1u) & ~(alignment - 1u);
        }
    }

    IopMemory::IopMemory()
        : m_ram(RamSize), m_owned(RamSize), m_scratch(ScratchSize)
    {
        reset();
    }

    void IopMemory::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint8_t{0});
        std::fill(m_owned.begin(), m_owned.end(), uint8_t{0});
        std::fill(m_scratch.begin(), m_scratch.end(), uint8_t{0});
        m_hardware.clear();
        m_allocations.clear();
        m_heapCursor = HeapBase;
        m_interruptStatus = 0;
        m_interruptMask = 0;
        m_interruptControl = 1;
        m_dmaStart.reset();
        m_spu2.reset();
    }

    uint32_t IopMemory::physicalAddress(uint32_t address) noexcept
    {
        return address & 0x1FFFFFFFu;
    }

    uint8_t IopMemory::read8(uint32_t address) const
    {
        const uint32_t phys = physicalAddress(address);
        if (phys < RamSize)
            return m_ram[phys];
        if (phys >= ScratchBase && phys < ScratchBase + ScratchSize)
            return m_scratch[phys - ScratchBase];
        if (isSpu2Register(phys))
            return static_cast<uint8_t>(const_cast<Spu2 &>(m_spu2).read16((phys - Spu2Base) & ~1u) >> ((phys & 1u) * 8u));
        const uint32_t value = readHardware32(phys & ~3u);
        return static_cast<uint8_t>(value >> ((phys & 3u) * 8u));
    }

    uint16_t IopMemory::read16(uint32_t address) const
    {
        const uint32_t phys = physicalAddress(address);
        if (phys + 1u < RamSize)
        {
            uint16_t value;
            std::memcpy(&value, m_ram.data() + phys, sizeof(value));
            return value;
        }
        if ((phys & 1u) == 0u && isSpu2Register(phys))
            return const_cast<Spu2 &>(m_spu2).read16(phys - Spu2Base);
        return static_cast<uint16_t>(read8(address) | (static_cast<uint16_t>(read8(address + 1u)) << 8u));
    }

    uint32_t IopMemory::read32(uint32_t address) const
    {
        const uint32_t phys = physicalAddress(address);
        if ((phys & 3u) == 0u && phys + 3u < RamSize)
        {
            uint32_t value;
            std::memcpy(&value, m_ram.data() + phys, sizeof(value));
            return value;
        }
        if ((phys & 3u) == 0u && phys >= ScratchBase && phys + 3u < ScratchBase + ScratchSize)
        {
            uint32_t value;
            std::memcpy(&value, m_scratch.data() + (phys - ScratchBase), sizeof(value));
            return value;
        }
        if ((phys & 3u) == 0u && isSpu2Register(phys))
            return static_cast<uint32_t>(read16(address)) | (static_cast<uint32_t>(read16(address + 2u)) << 16u);
        if ((phys & 3u) == 0u && isHardwareAddress(phys))
            return readHardware32(phys);

        return static_cast<uint32_t>(read8(address)) |
               (static_cast<uint32_t>(read8(address + 1u)) << 8u) |
               (static_cast<uint32_t>(read8(address + 2u)) << 16u) |
               (static_cast<uint32_t>(read8(address + 3u)) << 24u);
    }

    void IopMemory::write8(uint32_t address, uint8_t value)
    {
        const uint32_t phys = physicalAddress(address);
        if (phys < RamSize)
        {
            m_ram[phys] = value;
            markOwned(phys, sizeof(value));
            return;
        }
        if (phys >= ScratchBase && phys < ScratchBase + ScratchSize)
        {
            m_scratch[phys - ScratchBase] = value;
            return;
        }
        if (isSpu2Register(phys))
        {
            const uint32_t half = (phys - Spu2Base) & ~1u;
            const uint32_t shift = (phys & 1u) * 8u;
            const uint16_t current = m_spu2.read16(half);
            m_spu2.write16(half, static_cast<uint16_t>((current & ~(0xFFu << shift)) | (static_cast<uint32_t>(value) << shift)));
            return;
        }
        const uint32_t aligned = phys & ~3u;
        uint32_t current = readHardware32(aligned);
        const uint32_t shift = (phys & 3u) * 8u;
        current = (current & ~(0xFFu << shift)) | (static_cast<uint32_t>(value) << shift);
        writeHardware32(aligned, current);
    }

    void IopMemory::write16(uint32_t address, uint16_t value)
    {
        const uint32_t phys = physicalAddress(address);
        if (phys + 1u < RamSize)
        {
            std::memcpy(m_ram.data() + phys, &value, sizeof(value));
            markOwned(phys, sizeof(value));
            return;
        }
        if (phys >= Spu2Base && phys < Spu2Base + 0x800u && std::getenv("VENDETTA_SPUREGLOG"))
        {
            // VENDETTA_SPUREGLOG=core: only the cores' own registers (mixing, reverb, volumes, DMA), not the voices'
            // parameters (0x000-0x17F) and addresses (0x1C0-0x2DF) of each core, which are most of the writes
            static const bool coreOnly = std::string(std::getenv("VENDETTA_SPUREGLOG")) == "core";
            const uint32_t offset = phys & 0x7FFu;
            const uint32_t inCore = offset & 0x3FFu;
            const bool voiceRegister = offset < 0x760u && (inCore < 0x180u || (inCore >= 0x1C0u && inCore < 0x2E0u));
            static uint32_t logged = 0u;
            if (!(coreOnly && voiceRegister) && logged++ < 20000u && offset != 0x1ACu && offset != 0x5ACu)
            {
                static const auto start = std::chrono::steady_clock::now();
                const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
                std::cerr << "[spureg] " << ms << " 0x" << std::hex << (phys & 0x7FFu) << " = 0x" << value << std::dec << std::endl;
            }
        }
        if ((phys & 1u) == 0u && isSpu2Register(phys))
        {
            m_spu2.write16(phys - Spu2Base, value);
            return;
        }
        write8(address, static_cast<uint8_t>(value));
        write8(address + 1u, static_cast<uint8_t>(value >> 8u));
    }

    void IopMemory::write32(uint32_t address, uint32_t value)
    {
        const uint32_t phys = physicalAddress(address);
        if ((phys & 3u) == 0u && phys + 3u < RamSize)
        {
            std::memcpy(m_ram.data() + phys, &value, sizeof(value));
            markOwned(phys, sizeof(value));
            return;
        }
        if ((phys & 3u) == 0u && phys >= ScratchBase && phys + 3u < ScratchBase + ScratchSize)
        {
            std::memcpy(m_scratch.data() + (phys - ScratchBase), &value, sizeof(value));
            return;
        }
        if ((phys & 3u) == 0u && isSpu2Register(phys))
        {
            write16(address, static_cast<uint16_t>(value));
            write16(address + 2u, static_cast<uint16_t>(value >> 16u));
            return;
        }
        if ((phys & 3u) == 0u)
        {
            writeHardware32(phys, value);
            return;
        }
        write8(address, static_cast<uint8_t>(value));
        write8(address + 1u, static_cast<uint8_t>(value >> 8u));
        write8(address + 2u, static_cast<uint8_t>(value >> 16u));
        write8(address + 3u, static_cast<uint8_t>(value >> 24u));
    }

    bool IopMemory::readRam(uint32_t address, void *destination, size_t size) const
    {
        const uint32_t phys = physicalAddress(address);
        if ((!destination && size != 0u) || phys > RamSize || size > RamSize - phys)
            return false;
        if (size != 0u)
            std::memcpy(destination, m_ram.data() + phys, size);
        return true;
    }

    bool IopMemory::writeRam(uint32_t address, const void *source, size_t size)
    {
        const uint32_t phys = physicalAddress(address);
        if ((!source && size != 0u) || phys > RamSize || size > RamSize - phys)
            return false;
        if (size != 0u)
        {
            std::memcpy(m_ram.data() + phys, source, size);
            markOwned(phys, size);
        }
        return true;
    }

    bool IopMemory::zeroRam(uint32_t address, size_t size)
    {
        const uint32_t phys = physicalAddress(address);
        if (phys > RamSize || size > RamSize - phys)
            return false;
        if (size != 0u)
        {
            std::memset(m_ram.data() + phys, 0, size);
            markOwned(phys, size);
        }
        return true;
    }

    bool IopMemory::ownsRamRange(uint32_t address, size_t size) const
    {
        const uint32_t phys = physicalAddress(address);
        if (phys > RamSize || size > RamSize - phys)
            return false;
        return std::all_of(m_owned.begin() + phys, m_owned.begin() + phys + size,
                           [](uint8_t value)
                           { return value != 0u; });
    }

    bool IopMemory::isRamRange(uint32_t address, size_t size) const
    {
        const uint32_t phys = physicalAddress(address);
        return phys <= RamSize && size <= RamSize - phys;
    }

    void IopMemory::markOwned(uint32_t address, size_t size)
    {
        if (address > RamSize || size > RamSize - address)
            return;
        std::fill(m_owned.begin() + address, m_owned.begin() + address + size, uint8_t{1});
    }

    bool IopMemory::isHardwareAddress(uint32_t address) const
    {
        const uint32_t phys = physicalAddress(address);
        return (phys >= HardwareBase && phys < HardwareEnd) ||
               (phys >= Spu2Base && phys < Spu2End) ||
               (phys >= SifBase && phys < SifEnd);
    }

    uint32_t IopMemory::readHardware32(uint32_t address) const
    {
        const auto value = m_hardware.find(address);
        if (value != m_hardware.end())
            return value->second;
        switch (address)
        {
        case 0x1F801070u:
            return m_interruptStatus;
        case 0x1F801074u:
            return m_interruptMask;
        case 0x1F801078u:
            return m_interruptControl;
        default:
            return 0u;
        }
    }

    void IopMemory::writeHardware32(uint32_t address, uint32_t value)
    {
        switch (address)
        {
        case 0x1F801070u:
            m_interruptStatus &= value;
            return;
        case 0x1F801074u:
            m_interruptMask = value;
            return;
        case 0x1F801078u:
            m_interruptControl = value & 1u;
            return;
        default:
            break;
        }

        m_hardware[address] = value;
        if ((address != kDmaSpu0Chcr && address != kDmaSpu1Chcr) || (value & kDmaStart) == 0u)
            return;

        const bool secondCore = address == kDmaSpu1Chcr;
        const int core = secondCore ? 1 : 0;
        m_hardware[address] = value & ~kDmaStart;
        m_spu2.markDmaReady(core);

        const uint32_t blockControlAddress = address - sizeof(uint32_t);
        uint32_t blockControl = 0u;
        if (const auto current = m_hardware.find(blockControlAddress); current != m_hardware.end())
            blockControl = current->second;
        const uint32_t wordsPerBlock = std::max<uint32_t>(blockControl & 0xFFFFu, 1u);
        const uint32_t blockCount = std::max<uint32_t>(blockControl >> 16u, 1u);
        const uint64_t transferWords = static_cast<uint64_t>(wordsPerBlock) * blockCount;
        if (std::getenv("VENDETTA_SPULOG"))
        {
            // the SPU2 DMA a sound driver starts: source, size, the core's AutoDMA control (0x1F9001B0 + core×0x400)
            static uint32_t logged = 0u;
            if (logged++ < 400u)
            {
                const auto reg = [this](uint32_t a)
                { return static_cast<uint32_t>(m_spu2.read16(a - Spu2Base)); };
                const uint32_t madr = m_hardware.count(address - 8u) ? m_hardware[address - 8u] : 0u;
                std::cerr << "[spulog] core" << (secondCore ? 1 : 0) << " DMA chcr=0x" << std::hex << value
                          << " madr=0x" << madr << " bcr=0x" << blockControl << std::dec << " bytes=" << transferWords * 4u
                          << std::hex << " admas=0x" << reg(0x1F9001B0u + (secondCore ? 0x400u : 0u))
                          << " attr=0x" << reg(0x1F90019Au + (secondCore ? 0x400u : 0u)) << std::dec << std::endl;
            }
        }
        // AutoDMA (the core's ADMA control at 0x1F9001B0 + core×0x400 is set): the SPU2 does not take the data
        // into its RAM but plays it as a 48 kHz 16-bit stereo stream (blocks of 512 bytes left, 512 bytes right),
        // pulling it as it plays — so the transfer ends after the time the samples take to play (768 IOP cycles
        // per frame at 36.864 MHz). Finishing it at once made a sound driver that refills on this interrupt mix
        // hundreds of times faster than real time.
        constexpr uint64_t kIopCyclesPerSpuFrame = 768u;
        const bool autoDma = m_spu2.autoDmaEnabled(core);
        DmaStart start{
            secondCore ? kDmaSpu1Irq : kDmaSpu0Irq,
            core,
            std::max<uint64_t>(transferWords * 2u, 64u),
        };
        if (!autoDma)
        {
            // a plain transfer to / from sound memory at the core's transfer address (CHCR bit 0: 1 = to the SPU2)
            const auto madr = m_hardware.find(address - 8u);
            const uint32_t memoryAddress = physicalAddress(madr == m_hardware.end() ? 0u : madr->second);
            const uint32_t bytes = static_cast<uint32_t>(transferWords * 4u);
            if (memoryAddress < RamSize && bytes <= RamSize - memoryAddress)
            {
                if ((value & 1u) != 0u)
                    m_spu2.dmaWrite(core, m_ram.data() + memoryAddress, bytes);
                else
                {
                    m_spu2.dmaRead(core, m_ram.data() + memoryAddress, bytes);
                    markOwned(memoryAddress, bytes);
                }
            }
        }
        if (autoDma)
        {
            const uint32_t madrAddress = address - 8u;
            const auto madr = m_hardware.find(madrAddress);
            const uint32_t source = physicalAddress(madr == m_hardware.end() ? 0u : madr->second);
            const uint32_t bytes = static_cast<uint32_t>(transferWords * 4u) & ~0x3FFu; // whole L+R blocks
            if (bytes != 0u && source < RamSize && bytes <= RamSize - source)
            {
                constexpr uint32_t kBlockSamples = 256u; // 512 bytes per channel per block
                start.pcm.resize(static_cast<size_t>(bytes / 2u));
                size_t out = 0u;
                for (uint32_t block = 0u; block < bytes; block += 0x400u)
                {
                    const uint8_t *left = m_ram.data() + source + block;
                    const uint8_t *right = left + 0x200u;
                    for (uint32_t i = 0u; i < kBlockSamples; ++i)
                    {
                        int16_t l, r;
                        std::memcpy(&l, left + i * 2u, sizeof(l));
                        std::memcpy(&r, right + i * 2u, sizeof(r));
                        start.pcm[out++] = l;
                        start.pcm[out++] = r;
                    }
                }
                start.delayCycles = static_cast<uint64_t>(bytes / 4u) * kIopCyclesPerSpuFrame;
            }
        }
        m_dmaStart = std::move(start);
    }

    std::optional<IopMemory::DmaStart> IopMemory::takeDmaStart() noexcept
    {
        std::optional<DmaStart> result = std::move(m_dmaStart);
        m_dmaStart.reset();
        return result;
    }

    uint32_t IopMemory::allocate(uint32_t size, uint32_t alignment, std::optional<uint32_t> fixed)
    {
        size = alignUp(std::max(size, 1u), 16u);
        alignment = std::max<uint32_t>(alignment, 4u);
        if (fixed)
        {
            const uint32_t address = *fixed;
            if (address < HeapBase || address + size > HeapLimit)
                return 0u;
            for (const auto &block : m_allocations)
                if (address < block.address + block.size && block.address < address + size)
                    return 0u;
            m_allocations.push_back({address, size});
            markOwned(address, size);
            return address;
        }

        // first fit from the bottom, as the IOP's sysmem (AllocSysMemory, smallest address): starting at the highest
        // address ever used never reused freed blocks - HCTP stages each IRX through a heap block, so the cursor ended
        // near the top and CRI ADX's 260 KB sceSifAllocIopHeap failed (game halted on the first loading screen).
        uint32_t candidate = alignUp(HeapBase, alignment);
        for (;;)
        {
            bool overlap = false;
            for (const auto &block : m_allocations)
            {
                if (candidate < block.address + block.size && block.address < candidate + size)
                {
                    candidate = alignUp(block.address + block.size, alignment);
                    overlap = true;
                    break;
                }
            }
            if (!overlap)
                break;
        }
        if (candidate > HeapLimit || size > HeapLimit - candidate)
            return 0u;
        m_allocations.push_back({candidate, size});
        markOwned(candidate, size);
        m_heapCursor = std::max(m_heapCursor, candidate + size);
        return candidate;
    }

    bool IopMemory::freeAllocation(uint32_t address)
    {
        const auto block = std::find_if(m_allocations.begin(), m_allocations.end(),
                                        [&](const Allocation &candidate)
                                        { return candidate.address == address; });
        if (block == m_allocations.end())
            return false;
        std::fill(m_owned.begin() + block->address,
                  m_owned.begin() + block->address + block->size,
                  uint8_t{0});
        m_allocations.erase(block);
        return true;
    }

    // The free gaps of the heap, lowest first (sysmem's free list).
    static std::vector<IopMemory::Allocation> freeGaps(const std::vector<IopMemory::Allocation> &allocations)
    {
        std::vector<IopMemory::Allocation> used = allocations;
        std::sort(used.begin(), used.end(), [](const auto &a, const auto &b) { return a.address < b.address; });
        std::vector<IopMemory::Allocation> gaps;
        uint32_t cursor = IopMemory::HeapBase;
        for (const auto &block : used)
        {
            if (block.address + block.size <= cursor)
                continue;
            if (block.address > cursor)
                gaps.push_back({cursor, std::min(block.address, IopMemory::HeapLimit) - cursor});
            cursor = std::max(cursor, block.address + block.size);
            if (cursor >= IopMemory::HeapLimit)
                break;
        }
        if (cursor < IopMemory::HeapLimit)
            gaps.push_back({cursor, IopMemory::HeapLimit - cursor});
        return gaps;
    }

    // ALLOC_LAST: the highest free block that fits (sysmem takes the top of the last fitting gap).
    uint32_t IopMemory::allocateHigh(uint32_t size, uint32_t alignment)
    {
        size = alignUp(std::max(size, 1u), 16u);
        alignment = std::max<uint32_t>(alignment, 4u);
        const auto gaps = freeGaps(m_allocations);
        for (auto gap = gaps.rbegin(); gap != gaps.rend(); ++gap)
        {
            if (gap->size < size)
                continue;
            const uint32_t address = (gap->address + gap->size - size) & ~(alignment - 1u);
            if (address < gap->address)
                continue;
            m_allocations.push_back({address, size});
            markOwned(address, size);
            m_heapCursor = std::max(m_heapCursor, address + size);
            return address;
        }
        return 0u;
    }

    // QueryMaxFreeMemSize = the largest free block (was "top minus the highest address ever used", which counted none of
    // the blocks freed below it: after boot -> menus -> loads HCTP's sound driver was told the IOP was full and its bank
    // buffer came back 0 - the season match load stalled with 32 KB of bank copied to IOP address 0).
    uint32_t IopMemory::maxFreeMemory() const
    {
        uint32_t best = 0u;
        for (const auto &gap : freeGaps(m_allocations))
            best = std::max(best, gap.size);
        return best;
    }

    uint32_t IopMemory::totalFreeMemory() const
    {
        uint32_t total = 0u;
        for (const auto &gap : freeGaps(m_allocations))
            total += gap.size;
        return total;
    }

    std::optional<IopMemory::Allocation> IopMemory::allocationContaining(uint32_t address) const
    {
        const auto block = std::find_if(m_allocations.begin(), m_allocations.end(),
                                        [&](const Allocation &candidate)
                                        {
                                            return address >= candidate.address &&
                                                   address < candidate.address + candidate.size;
                                        });
        if (block == m_allocations.end())
            return std::nullopt;
        return *block;
    }

    std::string IopMemory::readString(uint32_t address, size_t limit) const
    {
        std::string result;
        result.reserve(std::min<size_t>(limit, 64u));
        for (size_t i = 0; i < limit; ++i)
        {
            const char ch = static_cast<char>(read8(address + static_cast<uint32_t>(i)));
            if (ch == '\0')
                break;
            result.push_back(ch);
        }
        return result;
    }
}
