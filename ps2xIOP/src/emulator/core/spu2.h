#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps2x::iop::detail
{
    // The PS2's sound chip (SPU2): two cores of 24 voices that play PS-ADPCM samples from its 2 MB sound memory, each
    // core mixing its voices with its inputs (the AutoDMA sound-data stream; on core 1 also core 0's output) into a
    // 48 kHz stereo output (core 1's is what the console plays). Registers at 0x1F900000-0x1F9007FF, 16 bits each,
    // core 1's at +0x400; the per-core volumes at 0x760 (core 0) / 0x788 (core 1). Each core's reverb runs on the
    // effects area of sound memory (ESA .. EEA) at half rate, as the hardware does.
    class Spu2
    {
    public:
        static constexpr uint32_t kRegisterBytes = 0x800u;
        static constexpr uint32_t kRamHalfwords = 1u << 20u; // 2 MB
        static constexpr uint64_t kIopCyclesPerSample = 768u; // 36.864 MHz / 48 kHz
        static constexpr int kVoices = 24;

        Spu2();
        void reset();

        // register access; offset = address - 0x1F900000 (0 .. 0x7FF), halfword aligned
        [[nodiscard]] uint16_t read16(uint32_t offset) const;
        void write16(uint32_t offset, uint16_t value);

        // a DMA (not AutoDMA) to / from sound memory at the core's transfer address (TSA), which then moves on
        void dmaWrite(int core, const uint8_t *data, size_t bytes);
        void dmaRead(int core, uint8_t *data, size_t bytes);
        void markDmaReady(int core); // STATX bit 7, what a sound library waits for after a transfer
        [[nodiscard]] bool autoDmaEnabled(int core) const;
        // AutoDMA: the core's sound-data input, 48 kHz stereo frames (L, R interleaved)
        void queueInput(int core, const int16_t *frames, size_t frameCount);

        // plays up to the IOP cycle count, appending the output frames (L, R interleaved)
        void advanceTo(uint64_t iopCycle, std::vector<int16_t> &out);
        [[nodiscard]] uint64_t nextSampleCycle() const noexcept { return m_nextSampleCycle; }

    private:
        enum class Phase : uint8_t
        {
            Off,
            Attack,
            Decay,
            Sustain,
            Release,
        };

        // a volume register: fixed (bit 15 clear) or a sweep (bit 15 set) that moves the level each sample
        struct Volume
        {
            uint16_t reg = 0;
            int32_t level = 0; // 0 .. 0x7FFF (a sweep's magnitude) or the fixed value (signed)
            uint32_t counter = 0;
            void set(uint16_t value);
            void step();
            [[nodiscard]] int32_t value() const;
        };

        struct Voice
        {
            Phase phase = Phase::Off;
            int32_t envelope = 0;
            uint32_t envelopeCounter = 0;
            uint32_t nextAddress = 0; // halfwords: the next block to decode
            uint32_t loopAddress = 0;
            uint32_t blockAddress = 0;
            bool loopLocked = false;   // the loop address came from the register, not a block's loop-start flag
            bool loopWritten = false;  // LSAX written since the last key-on
            bool blockLoaded = false;
            uint8_t blockFlags = 0;
            uint8_t position = 28;     // the next of the 28 decoded samples
            std::array<int16_t, 28> samples{};
            int32_t history1 = 0;
            int32_t history2 = 0;
            std::array<int32_t, 4> recent{}; // the last four decoded samples, oldest first, for interpolation
            uint32_t counter = 0;      // the pitch counter's 12-bit fraction
            int32_t output = 0;        // after the envelope, before the volume (pitch modulation of the next voice)
            uint32_t playedSamples = 0; // VENDETTA_SPULOG: how long it played and its loudest output
            int32_t peak = 0;
            Volume left;
            Volume right;
        };

        struct Core
        {
            std::array<Voice, kVoices> voices{};
            uint32_t endFlags = 0;          // ENDX
            uint32_t transferAddress = 0;   // halfwords, where the data port / a DMA writes next
            std::vector<int16_t> input;     // AutoDMA frames not yet played
            size_t inputRead = 0;
            bool inputPlaying = false;
            Volume masterLeft;
            Volume masterRight;
            int32_t outLeft = 0;
            int32_t outRight = 0;
            int32_t noiseLevel = 1;
            int32_t noiseTimer = 0;
            // reverb: the wet input and output around the half-rate effect, each a 64-sample ring stored twice (so a
            // filter reads 39 samples without wrapping); the position in the effects area; which channel is next
            std::array<std::array<int32_t, 128>, 2> reverbDown{};
            std::array<std::array<int32_t, 128>, 2> reverbUp{};
            uint32_t reverbRing = 0;
            uint32_t reverbPosition = 0;
            bool reverbRight = false;
            int32_t reverbInPeak = 0; // VENDETTA_SPULOG: the loudest wet input / effect output in the last second
            int32_t reverbOutPeak = 0;
        };

        [[nodiscard]] uint16_t reg(uint32_t offset) const { return m_regs[(offset & 0x7FFu) >> 1u]; }
        void keyOn(int core, uint32_t mask);
        void keyOff(int core, uint32_t mask);
        void decodeBlock(Voice &voice);
        int32_t nextSample(Core &core, int index, Voice &voice, uint32_t pitch);
        void stepEnvelope(int core, int index, Voice &voice);
        void mixCore(int core, int32_t extLeft, int32_t extRight);
        void reverb(int core, int32_t wetLeft, int32_t wetRight, int32_t &outLeft, int32_t &outRight);

        std::vector<uint16_t> m_ram;
        std::array<uint16_t, kRegisterBytes / 2u> m_regs{};
        std::array<Core, 2> m_cores{};
        uint64_t m_nextSampleCycle = kIopCyclesPerSample;
        uint32_t m_samplesSinceReport = 0;
    };
}
